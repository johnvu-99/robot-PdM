"""
The simulation worker.

Owns the PyBullet client and runs the fixed timestep accumulator loop on its
own thread. It never touches a QWidget: everything leaves through Qt signals at
a few tens of hertz, or through the frame mailbox.

Commands arrive on a plain deque instead of queued slot invocations, because
the loop below blocks the thread and therefore never runs a Qt event loop of
its own. Signals emitted from here are still delivered normally, since queuing
happens on the receiving thread.
"""

import collections
import platform
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

import config
from rendering.pybullet_renderer import PyBulletRenderer
from simulator.camera import OrbitCamera
from simulator.joint_controller import CONTROL_MODES, JointController
from simulator.robot import Robot, RobotLoadError
from simulator.trajectory import TrajectoryGenerator, MODE_HOME, MODE_MANUAL

STATE_STOPPED = "STOPPED"
STATE_RUNNING = "RUNNING"
STATE_PAUSED = "PAUSED"
STATE_EMERGENCY_STOP = "EMERGENCY_STOP"


class SimulationWorker(QObject):

    ready = pyqtSignal(object)          # static robot/renderer description
    telemetry_ready = pyqtSignal(object)
    stats_ready = pyqtSignal(object)
    state_changed = pyqtSignal(str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, mailbox, parent=None):
        QObject.__init__(self, parent)
        self.mailbox = mailbox

        self.client_id = -1
        self.robot = None
        self.controller = None
        self.trajectory = None
        self.camera = None
        self.renderer = None

        self.state = STATE_STOPPED
        self.sim_time = 0.0

        # Settings that survive a robot swap.
        self._control_mode = config.DEFAULT_CONTROL_MODE
        self._speed = 1.0
        self._amplitude = 0.6
        self._position_gain = config.DEFAULT_POSITION_GAIN
        self._velocity_gain = config.DEFAULT_VELOCITY_GAIN
        self._joint_enabled = np.ones(0, dtype=bool)
        self._renderer_note = ""

        self._commands = collections.deque()
        self._wake = threading.Event()
        self._stop_event = threading.Event()

        self._accumulator = 0.0
        self._clamped_deltas = 0
        self._dropped_backlog = 0

        self._step_durations = collections.deque(maxlen=240)
        self._render_durations = collections.deque(maxlen=120)
        self._steps_in_window = 0
        self._frames_in_window = 0
        self._total_frames = 0

        buffer_length = int(config.TELEMETRY_BUFFER_SECONDS * config.TELEMETRY_SAMPLE_HZ)
        self._telemetry_buffer = collections.deque(maxlen=buffer_length)
        self._telemetry_period = 1.0 / max(config.TELEMETRY_SAMPLE_HZ, 1.0)
        self._telemetry_accumulator = 0.0

        self._positions = None
        self._velocities = None
        self._torques = None
        self._targets = None
        self._target_velocities = None
        self._target_accelerations = None

        self._timer_period_set = False
        self._last_render_wall = 0.0

        self._handlers = {
            "start": self._cmd_start,
            "pause": self._cmd_pause,
            "stop": self._cmd_stop,
            "reset": self._cmd_reset,
            "emergency_stop": self._cmd_emergency_stop,
            "set_mode": self._cmd_set_mode,
            "set_speed": self._cmd_set_speed,
            "set_amplitude": self._cmd_set_amplitude,
            "set_manual_targets": self._cmd_set_manual_targets,
            "set_gains": self._cmd_set_gains,
            "camera_orbit": self._cmd_camera_orbit,
            "camera_pan": self._cmd_camera_pan,
            "camera_zoom": self._cmd_camera_zoom,
            "camera_preset": self._cmd_camera_preset,
            "set_viewport": self._cmd_set_viewport,
            "set_quality": self._cmd_set_quality,
            "load_robot": self._cmd_load_robot,
            "set_control_mode": self._cmd_set_control_mode,
            "set_joint_enabled": self._cmd_set_joint_enabled,
        }

    # -- external API (thread safe) -----------------------------------------

    def post(self, name, payload=None):
        """Called from the GUI thread. Never blocks."""
        self._commands.append((name, payload or {}))
        self._wake.set()

    def request_stop(self):
        self._stop_event.set()
        self._wake.set()

    # -- lifecycle ----------------------------------------------------------

    @pyqtSlot()
    def run(self):
        if not self._setup():
            self.finished.emit()
            return
        self._request_timer_resolution()
        try:
            self._loop()
        finally:
            self._release_timer_resolution()
            self._teardown()
            self.finished.emit()

    def _setup(self):
        try:
            self.client_id = p.connect(p.DIRECT)
            if self.client_id < 0:
                self.log_message.emit("Could not create a PyBullet client.")
                return False

            p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                      physicsClientId=self.client_id)
            p.resetSimulation(physicsClientId=self.client_id)
            p.setGravity(0.0, 0.0, config.GRAVITY, physicsClientId=self.client_id)
            p.setTimeStep(config.FIXED_DT, physicsClientId=self.client_id)
            p.setRealTimeSimulation(0, physicsClientId=self.client_id)
            p.setPhysicsEngineParameter(
                fixedTimeStep=config.FIXED_DT,
                numSolverIterations=config.SOLVER_ITERATIONS,
                numSubSteps=1,
                deterministicOverlappingPairs=1,
                physicsClientId=self.client_id,
            )

            if config.LOAD_GROUND_PLANE:
                p.loadURDF("plane.urdf", physicsClientId=self.client_id)

            try:
                self._install_robot(config.URDF_PATH)
            except RobotLoadError as exc:
                self.log_message.emit("Could not load %s: %s" % (config.URDF_PATH, exc))
                return False

            self.camera = OrbitCamera()
            self.renderer = PyBulletRenderer(self.client_id)
            renderer_name, tiny_ms, hardware_ms = self.renderer.select_renderer()

            self._emit_ready()
            self.log_message.emit(
                "Renderer: %s at %dx%d (256px probe: software %s, hardware %s)."
                % (renderer_name, self.renderer.width, self.renderer.height,
                   "n/a" if tiny_ms is None else "%.1f ms" % tiny_ms,
                   "unavailable" if hardware_ms is None else "%.1f ms" % hardware_ms))
            self._set_state(STATE_STOPPED)
            return True
        except Exception as exc:  # startup failures must reach the UI
            self.log_message.emit("Simulation setup failed: %s" % exc)
            return False

    def _install_robot(self, path):
        """
        Load a model and build its controller and trajectory generator. The
        new robot is fully built before the old one is removed, so a model
        that fails to load leaves the running robot untouched.
        """
        robot = Robot(self.client_id, path, config.ROBOT_BASE_POSITION,
                      config.USE_FIXED_BASE)
        try:
            robot.load()
            if robot.n == 0:
                raise RobotLoadError("model has no movable joints")
            controller = JointController(self.client_id, robot)
            controller.set_gains(self._position_gain, self._velocity_gain)
            if not controller.set_control_mode(self._control_mode):
                controller.pop_warnings()
                self._control_mode = controller.control_mode
            trajectory = TrajectoryGenerator(
                robot.n, robot.home_pose, robot.lower_limits, robot.upper_limits,
                controller.velocity_limits)
            trajectory.set_speed(self._speed)
            trajectory.set_amplitude(self._amplitude)
        except RobotLoadError:
            robot.unload()
            raise
        except (p.error, ValueError, IndexError) as exc:
            robot.unload()
            raise RobotLoadError(str(exc))

        if self.robot is not None:
            for problem in self.robot.unload():
                self.log_message.emit("Unloading previous robot: %s" % problem)

        self.robot = robot
        self.controller = controller
        self.trajectory = trajectory
        n = robot.n
        self._joint_enabled = np.ones(n, dtype=bool)
        self._positions = robot.home_pose.copy()
        self._velocities = np.zeros(n)
        self._torques = np.zeros(n)
        self._targets = robot.home_pose.copy()
        self._target_velocities = np.zeros(n)
        self._target_accelerations = np.zeros(n)
        self._telemetry_buffer.clear()
        self._telemetry_accumulator = 0.0
        controller.release_motors()

        self.log_message.emit(
            "Loaded %s: %d joints, %d movable, %s base, inverse dynamics %s."
            % (path, len(robot.all_joints), n,
               "fixed" if robot.fixed_base else "free",
               "available" if robot.supports_inverse_dynamics else "unavailable"))

    def _emit_ready(self):
        description = self.robot.describe()
        description.update(self.renderer.describe())
        description["physics_hz"] = config.PHYSICS_HZ
        description["control_mode"] = self.controller.control_mode
        description["enabled"] = self._joint_enabled.tolist()
        self.ready.emit(description)

    def _teardown(self):
        if self.client_id >= 0:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except p.error as exc:
                self.log_message.emit("PyBullet disconnect failed: %s" % exc)
            self.client_id = -1

    # -- Windows timer resolution -------------------------------------------

    def _request_timer_resolution(self):
        """
        Default Windows sleep granularity is around 15 ms, which would make the
        accumulator constantly catch up in bursts. 1 ms makes sleeps accurate
        enough that the loop lands close to every deadline.
        """
        if platform.system() != "Windows":
            return
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(config.WINDOWS_TIMER_RESOLUTION_MS)
            self._timer_period_set = True
        except Exception:
            self._timer_period_set = False

    def _release_timer_resolution(self):
        if not self._timer_period_set:
            return
        try:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(config.WINDOWS_TIMER_RESOLUTION_MS)
        except (AttributeError, OSError) as exc:
            self.log_message.emit("timeEndPeriod failed: %s" % exc)
        self._timer_period_set = False

    # -- main loop ----------------------------------------------------------

    def _loop(self):
        last = time.perf_counter()
        self._last_render_wall = last
        window_start = last
        next_render = last
        next_telemetry = last
        next_stats = last

        telemetry_period = 1.0 / config.TELEMETRY_UI_HZ
        stats_period = 1.0 / config.STATS_UI_HZ

        while not self._stop_event.is_set():
            iteration_start = time.perf_counter()

            self._wake.clear()
            self._drain_commands()

            delta = iteration_start - last
            last = iteration_start
            if delta > config.MAX_FRAME_DELTA:
                self._clamped_deltas += 1
                delta = config.MAX_FRAME_DELTA

            if self.state in (STATE_RUNNING, STATE_EMERGENCY_STOP):
                self._accumulator += delta
                steps = 0
                step_start = time.perf_counter()
                while (self._accumulator >= config.FIXED_DT
                       and steps < config.MAX_STEPS_PER_ITERATION):
                    self._physics_step()
                    self._accumulator -= config.FIXED_DT
                    steps += 1
                if steps > 0:
                    elapsed = time.perf_counter() - step_start
                    self._step_durations.append(elapsed / steps)
                    self._steps_in_window += steps
                if self._accumulator >= config.FIXED_DT:
                    # Backlog we refuse to replay. Dropping it is what keeps a
                    # hitch from turning into a burst of fast motion.
                    self._accumulator = 0.0
                    self._dropped_backlog += 1
            else:
                self._accumulator = 0.0

            # The camera follows wall clock time so it stays smooth even while
            # the simulation is paused.
            self.camera.update(delta)

            now = time.perf_counter()

            if now >= next_render:
                self._render_frame(now)
                period = 1.0 / max(self.renderer.target_fps, 1.0)
                next_render += period
                if next_render <= now:
                    next_render = now + period

            if now >= next_telemetry:
                self._emit_telemetry()
                next_telemetry = now + telemetry_period

            if now >= next_stats:
                self._emit_stats(now - window_start)
                window_start = now
                next_stats = now + stats_period

            self._sleep(iteration_start, next_render)

    def _sleep(self, iteration_start, next_render):
        """
        Wait on an Event rather than time.sleep, so an incoming command wakes
        the loop immediately instead of after the remaining timeslice. Small
        timing errors are harmless: the accumulator absorbs them and the
        average physics rate stays exact.
        """
        if self.state in (STATE_RUNNING, STATE_EMERGENCY_STOP):
            deadline = iteration_start + config.FIXED_DT
        else:
            deadline = min(next_render, iteration_start + 0.02)
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            self._wake.wait(remaining)

    def _physics_step(self):
        if self.state == STATE_EMERGENCY_STOP:
            self.controller.engage_brakes()
        else:
            targets, velocities, accelerations = self.trajectory.step(config.FIXED_DT)
            self.controller.apply(targets, velocities, accelerations,
                                  self._positions, self._velocities)
            self._targets = self.controller.commanded_positions
            self._target_velocities = self.controller.commanded_velocities
            self._target_accelerations = accelerations

        p.stepSimulation(physicsClientId=self.client_id)
        self.sim_time += config.FIXED_DT

        positions, velocities, torques = self.robot.get_states()
        applied = self.controller.applied_torques
        if applied is not None:
            torques = applied
        self._positions = positions
        self._velocities = velocities
        self._torques = torques

        self._telemetry_accumulator += config.FIXED_DT
        if self._telemetry_accumulator >= self._telemetry_period:
            self._telemetry_accumulator -= self._telemetry_period
            self._telemetry_buffer.append((
                self.sim_time,
                positions.copy(),
                velocities.copy(),
                torques.copy(),
                np.array(self._targets, dtype=np.float64),
            ))

    def _render_frame(self, now):
        elapsed = now - self._last_render_wall if self._last_render_wall else 0.0
        self._last_render_wall = now
        data, width, height, duration = self.renderer.render(self.camera)
        self._render_durations.append(duration)
        if data is not None:
            self.mailbox.put(data, width, height)
            self._frames_in_window += 1
            self._total_frames += 1
        message = self.renderer.note_render_duration(duration, elapsed)
        if message:
            self.log_message.emit(message)

    # -- outbound data ------------------------------------------------------

    def _emit_telemetry(self):
        if self._positions is None:
            return
        positions = np.asarray(self._positions)
        targets = np.asarray(self._targets)
        velocities = np.asarray(self._velocities)
        torques = np.asarray(self._torques)
        error = targets - positions

        self.telemetry_ready.emit({
            "sim_time": self.sim_time,
            "state": self.state,
            "mode": self.trajectory.mode if self.trajectory else MODE_HOME,
            "position": positions.tolist(),
            "target": targets.tolist(),
            "error": error.tolist(),
            "velocity": velocities.tolist(),
            "acceleration": np.asarray(self._target_accelerations).tolist(),
            "torque": torques.tolist(),
            "power": (torques * velocities).tolist(),
            "max_error": float(np.max(np.abs(error))) if error.size else 0.0,
            "enabled": self._joint_enabled.tolist(),
            "control_mode": self.controller.control_mode,
            "model": self.robot.urdf_path,
            "samples": len(self._telemetry_buffer),
        })

    def _emit_stats(self, elapsed):
        if elapsed <= 0.0:
            return
        steps = self._steps_in_window
        frames = self._frames_in_window
        self._steps_in_window = 0
        self._frames_in_window = 0

        step_ms = 0.0
        if self._step_durations:
            step_ms = 1000.0 * sum(self._step_durations) / len(self._step_durations)
        render_ms = 0.0
        if self._render_durations:
            render_ms = 1000.0 * sum(self._render_durations) / len(self._render_durations)

        self.stats_ready.emit({
            "physics_hz": steps / elapsed,
            "render_fps": frames / elapsed,
            "real_time_factor": (steps * config.FIXED_DT) / elapsed,
            "step_ms": step_ms,
            "render_ms": render_ms,
            "sim_time": self.sim_time,
            "state": self.state,
            "resolution": "%dx%d" % (self.renderer.width, self.renderer.height),
            "renderer": self.renderer.renderer_name,
            "quality": self.renderer.quality,
            "target_fps": self.renderer.target_fps,
            "clamped_deltas": self._clamped_deltas,
            "dropped_backlog": self._dropped_backlog,
            "model": self.robot.urdf_path,
        })
        for warning in self.controller.pop_warnings():
            self.log_message.emit(warning)

    def _set_state(self, state):
        if state == self.state:
            return
        self.state = state
        self.state_changed.emit(state)

    # -- command dispatch ---------------------------------------------------

    def _drain_commands(self):
        while True:
            try:
                name, payload = self._commands.popleft()
            except IndexError:
                return
            handler = self._handlers.get(name)
            if handler is None:
                continue
            try:
                handler(payload)
            except Exception as exc:
                self.log_message.emit("Command %s failed: %s" % (name, exc))

    def _cmd_start(self, payload):
        if self.state == STATE_EMERGENCY_STOP:
            self.log_message.emit("Emergency stop is latched. Press Reset first.")
            return
        if self.state == STATE_RUNNING:
            return
        if self.state == STATE_STOPPED:
            self.trajectory.resync(self._positions)
            self.trajectory.set_mode(self.trajectory.mode, transition_time=0.5)
        self._accumulator = 0.0
        self._set_state(STATE_RUNNING)
        self.log_message.emit("Running.")

    def _cmd_pause(self, payload):
        if self.state != STATE_RUNNING:
            return
        # Holding the last commanded pose means the arm does not sag while the
        # solver is idle, and resuming does not fight gravity.
        self.controller.apply(self.controller.commanded_positions,
                              np.zeros(self.robot.n), None,
                              self._positions, self._velocities)
        self._accumulator = 0.0
        self._set_state(STATE_PAUSED)
        self.log_message.emit("Paused.")

    def _cmd_stop(self, payload):
        self._halt_and_home()
        self.log_message.emit("Stopped and returned to home pose.")

    def _cmd_reset(self, payload):
        self._halt_and_home()
        self.camera.apply_preset("Home")
        self.renderer.reset_adaptive()
        self._clamped_deltas = 0
        self._dropped_backlog = 0
        self.log_message.emit("Reset. Emergency stop cleared.")

    def _halt_and_home(self):
        self.robot.reset_to(self.robot.home_pose)
        self.controller.release_motors()
        self.trajectory.resync(self.robot.home_pose)
        self.trajectory.set_mode(MODE_HOME, transition_time=0.1)
        self.trajectory.set_hold(~self._joint_enabled)
        self._telemetry_buffer.clear()
        self._telemetry_accumulator = 0.0
        self.sim_time = 0.0
        self._accumulator = 0.0
        self._positions = self.robot.home_pose.copy()
        self._velocities = np.zeros(self.robot.n)
        self._torques = np.zeros(self.robot.n)
        self._targets = self.robot.home_pose.copy()
        self._set_state(STATE_STOPPED)

    def _cmd_emergency_stop(self, payload):
        if self.state == STATE_EMERGENCY_STOP:
            return
        self.controller.engage_brakes()
        self._set_state(STATE_EMERGENCY_STOP)
        self.log_message.emit("EMERGENCY STOP. Motors braking, motion disabled.")

    def _cmd_set_mode(self, payload):
        mode = payload.get("mode")
        if self.trajectory.set_mode(mode):
            self.log_message.emit("Motion mode: %s" % mode)

    def _cmd_set_speed(self, payload):
        self._speed = float(payload.get("value", 1.0))
        self.trajectory.set_speed(self._speed)

    def _cmd_set_amplitude(self, payload):
        self._amplitude = float(payload.get("value", 0.6))
        self.trajectory.set_amplitude(self._amplitude)

    def _cmd_set_manual_targets(self, payload):
        if self.trajectory.mode != MODE_MANUAL:
            return
        self.trajectory.set_manual_targets(payload.get("targets", []))

    def _cmd_set_gains(self, payload):
        if payload.get("position_gain") is not None:
            self._position_gain = float(payload["position_gain"])
        if payload.get("velocity_gain") is not None:
            self._velocity_gain = float(payload["velocity_gain"])
        self.controller.set_gains(self._position_gain, self._velocity_gain)

    def _cmd_load_robot(self, payload):
        path = payload.get("path") or ""
        if not path:
            self.log_message.emit("No model path given.")
            return
        if self.state == STATE_RUNNING:
            self.log_message.emit("Stopping motion to swap the robot model.")
        try:
            self._install_robot(path)
        except RobotLoadError as exc:
            self.log_message.emit("Could not load %s: %s. Keeping %s."
                                  % (path, exc, self.robot.urdf_path))
            return
        self._halt_and_home()
        self._emit_ready()

    def _cmd_set_control_mode(self, payload):
        mode = payload.get("mode")
        if mode not in CONTROL_MODES:
            self.log_message.emit("Unknown control mode: %s" % mode)
            return
        if mode == self.controller.control_mode:
            return
        if not self.controller.set_control_mode(mode):
            for warning in self.controller.pop_warnings():
                self.log_message.emit(warning)
            self._emit_ready()
            return
        self._control_mode = mode
        if self.state == STATE_EMERGENCY_STOP:
            # Keep braking; the new mode takes effect after Reset.
            self.controller.engage_brakes()
        elif self.state == STATE_PAUSED:
            self.controller.apply(self.controller.commanded_positions,
                                  np.zeros(self.robot.n), None,
                                  self._positions, self._velocities)
        self.log_message.emit("Control mode: %s" % mode)

    def _cmd_set_joint_enabled(self, payload):
        index = int(payload.get("index", -1))
        enabled = bool(payload.get("enabled", True))
        if not 0 <= index < self.robot.n:
            return
        if bool(self._joint_enabled[index]) == enabled:
            return
        self._joint_enabled[index] = enabled
        self.trajectory.set_hold(~self._joint_enabled)
        if enabled:
            # Re-enter the running pattern through a blend, so the joint that
            # was frozen joins the motion without a jump.
            self.trajectory.set_mode(self.trajectory.mode,
                                     transition_time=config.REENABLE_TRANSITION_TIME,
                                     regenerate=False)
        self.log_message.emit("Joint %d (%s) %s."
                              % (index + 1, self.robot.joint_names[index],
                                 "enabled" if enabled else "disabled, holding position"))

    def _cmd_camera_orbit(self, payload):
        self.camera.orbit(payload.get("dx", 0.0), payload.get("dy", 0.0))

    def _cmd_camera_pan(self, payload):
        self.camera.pan(payload.get("dx", 0.0), payload.get("dy", 0.0))

    def _cmd_camera_zoom(self, payload):
        self.camera.zoom(payload.get("notches", 0.0))

    def _cmd_camera_preset(self, payload):
        self.camera.apply_preset(payload.get("name", "Home"))

    def _cmd_set_viewport(self, payload):
        self.renderer.set_viewport(payload.get("width", 960),
                                   payload.get("height", 540))
        self.mailbox.clear()

    def _cmd_set_quality(self, payload):
        quality = payload.get("quality")
        if self.renderer.set_quality(quality):
            self.log_message.emit(
                "Render quality: %s (%dx%d, target %.0f FPS)"
                % (quality, self.renderer.width, self.renderer.height,
                   self.renderer.target_fps))
