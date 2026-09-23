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
import datetime
import os
import platform
import threading
import time

import numpy as np
import pybullet as p
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

import config
from monitoring.telemetry import TelemetryRecorder, CH_ACCELERATION, CH_FORCE_MAGNITUDE
from storage.csv_logger import CsvTelemetryWriter
from rendering.pybullet_renderer import PyBulletRenderer
from simulator.camera import OrbitCamera
from simulator.fault_injector import FAULT_TYPES, FaultInjector, FAULT_CAPTIONS
from simulator.step_pipeline import StepPipeline, configure_client
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
    chart_data = pyqtSignal(object)
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

        # 240 Hz ring buffer of every movable joint, plus decimated views for
        # charts and CSV. See monitoring/telemetry.py.
        self.recorder = TelemetryRecorder(config.FIXED_DT,
                                          config.TELEMETRY_BUFFER_SECONDS,
                                          config.CHART_SAMPLE_HZ)
        self._csv_writer = None
        self._csv_rate = 0
        self._finishing_writers = []
        # Callable (records, mode, transitioning, state) installed by the UI
        # before the thread starts. Must not block; the monitoring service
        # only enqueues. Physics never imports monitoring or ML code.
        self.monitor_sink = None

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
            "inject_fault": self._cmd_inject_fault,
            "start_degradation": self._cmd_start_degradation,
            "remove_fault": self._cmd_remove_fault,
            "clear_faults": self._cmd_clear_faults,
            "start_logging": self._cmd_start_logging,
            "stop_logging": self._cmd_stop_logging,
            "export_csv": self._cmd_export_csv,
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
            self._close_writers(wait=True)
            self._teardown()
            self.finished.emit()

    def _setup(self):
        try:
            self.client_id = p.connect(p.DIRECT)
            if self.client_id < 0:
                self.log_message.emit("Could not create a PyBullet client.")
                return False

            configure_client(self.client_id)

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
        self.injector = FaultInjector(self.client_id, robot, controller, config.FIXED_DT)
        self.pipeline = StepPipeline(self.client_id, robot, controller, trajectory,
                                     self.injector, self.recorder)
        n = robot.n
        self._joint_enabled = np.ones(n, dtype=bool)
        self._positions = robot.home_pose.copy()
        self._velocities = np.zeros(n)
        self._torques = np.zeros(n)
        self._targets = robot.home_pose.copy()
        self._target_velocities = np.zeros(n)
        self._target_accelerations = np.zeros(n)
        self.recorder.reset()
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
        next_chart = last
        next_stats = last
        chart_period = 1.0 / config.CHART_UI_HZ

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

            if now >= next_chart:
                self._emit_chart_and_csv()
                next_chart = now + chart_period

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
        self.sim_time = self.pipeline.step(self.sim_time,
                                           braking=self.state == STATE_EMERGENCY_STOP)
        pipeline = self.pipeline
        self._positions = pipeline.positions
        self._velocities = pipeline.velocities
        self._torques = pipeline.torques
        self._targets = pipeline.targets
        self._target_velocities = pipeline.target_velocities
        self._target_accelerations = pipeline.target_accelerations

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
        latest = self.recorder.latest
        if latest is not None and latest[1].shape[0] == positions.shape[0]:
            row = latest[1]
            acceleration = row[:, CH_ACCELERATION].tolist()
            force = row[:, CH_FORCE_MAGNITUDE].tolist()
            moment = np.sqrt(np.sum(row[:, 9:12] ** 2, axis=1)).tolist()
            reactions = row[:, 6:12].tolist()
        else:
            zeros = [0.0] * positions.shape[0]
            acceleration, force, moment = zeros, zeros, zeros
            reactions = [[0.0] * 6 for _ in zeros]

        self.telemetry_ready.emit({
            "sim_time": self.sim_time,
            "state": self.state,
            "mode": self.trajectory.mode if self.trajectory else MODE_HOME,
            "position": positions.tolist(),
            "target": targets.tolist(),
            "error": error.tolist(),
            "velocity": velocities.tolist(),
            "acceleration": acceleration,
            "force_magnitude": force,
            "moment_magnitude": moment,
            "reactions": reactions,
            "torque": torques.tolist(),
            "power": (torques * velocities).tolist(),
            "max_error": float(np.max(np.abs(error))) if error.size else 0.0,
            "enabled": self._joint_enabled.tolist(),
            "control_mode": self.controller.control_mode,
            "model": self.robot.urdf_path,
            "samples": len(self.recorder),
            "labels": [list(label) for label in self.injector.labels],
            "faults": self.injector.describe(self.sim_time),
        })
        for joint, old, new in self.injector.pop_events():
            self.log_message.emit("t=%.2f s  Joint %d: %s -> %s"
                                  % (self.sim_time, joint + 1, old, new))

    def _emit_chart_and_csv(self):
        if self.monitor_sink is not None:
            self.recorder.monitor_enabled = True
            records = self.recorder.take_monitor_records()
            if records:
                self.monitor_sink(records, self.trajectory.mode,
                                  self.trajectory.transitioning, self.state)
        chunk = self.recorder.take_chart_chunk()
        if chunk is not None:
            self.chart_data.emit(chunk)
        records = self.recorder.take_csv_records()
        if records and self._csv_writer is not None:
            self._csv_writer.submit(records)

    def _logging_status(self):
        self._reap_writers()
        writer = self._csv_writer
        if writer is None:
            return {"active": False}
        if writer.error:
            message = writer.error
            self.log_message.emit("CSV logging stopped: %s" % message)
            self._stop_logging_internal("error")
            return {"active": False, "error": message}
        return {
            "active": True,
            "path": writer.path,
            "rows": writer.rows_written,
            "bytes": writer.bytes_written,
            "rate": self._csv_rate,
        }

    def _reap_writers(self):
        still_running = []
        for writer, kind in self._finishing_writers:
            if writer.is_alive():
                still_running.append((writer, kind))
                continue
            if writer.error:
                self.log_message.emit("%s failed: %s" % (kind, writer.error))
            else:
                self.log_message.emit("%s complete: %s (%d rows)."
                                      % (kind, writer.path, writer.rows_written))
        self._finishing_writers = still_running

    def _close_writers(self, wait):
        if self._csv_writer is not None:
            self._stop_logging_internal("shutdown")
        for writer, _ in self._finishing_writers:
            if wait:
                writer.join(3.0)
        self._finishing_writers = []

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
            "logging": self._logging_status(),
            "samples": len(self.recorder),
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
        if self.injector.has_items:
            self.log_message.emit("Faults and degradation cleared (simulated time restarts).")
        self.injector.clear()
        if self._csv_writer is not None:
            # Simulated time restarts at zero, so the current file ends here.
            self._stop_logging_internal("simulation reset")
        self.robot.reset_to(self.robot.home_pose)
        self.controller.release_motors()
        self.trajectory.resync(self.robot.home_pose)
        self.trajectory.set_mode(MODE_HOME, transition_time=0.1)
        self.trajectory.set_hold(~self._joint_enabled)
        self.recorder.reset()
        self.sim_time = 0.0
        self._accumulator = 0.0
        self.pipeline.reset_state()
        self._positions = self.pipeline.positions
        self._velocities = self.pipeline.velocities
        self._torques = self.pipeline.torques
        self._targets = self.pipeline.targets
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

    # -- faults -------------------------------------------------------------

    def _cmd_inject_fault(self, payload):
        fault_type = payload.get("type")
        if fault_type not in FAULT_TYPES:
            self.log_message.emit("Unknown fault type: %s" % fault_type)
            return
        start = self.sim_time + max(0.0, float(payload.get("start_delay", 0.0)))
        try:
            fault = self.injector.add_fault(int(payload.get("joint", 0)), fault_type,
                                            float(payload.get("severity", 0.5)), start,
                                            float(payload.get("duration", 0.0)))
        except ValueError as exc:
            self.log_message.emit("Fault rejected: %s" % exc)
            return
        info = fault.describe()
        self.log_message.emit(
            "Fault #%d scheduled: joint %d %s, %s, start t=%.2f s, %s."
            % (fault.id, fault.joint + 1, FAULT_CAPTIONS[fault_type], info["meaning"],
               fault.start, "permanent" if fault.duration <= 0 else "%.1f s" % fault.duration))

    def _cmd_start_degradation(self, payload):
        start = self.sim_time + max(0.0, float(payload.get("start_delay", 0.0)))
        try:
            scenario = self.injector.add_scenario(
                payload.get("scenario"), int(payload.get("joint", 0)), start,
                float(payload.get("duration", config.DEGRADATION_DEFAULT_DURATION_S)),
                float(payload.get("final_health", config.DEGRADATION_DEFAULT_FINAL_HEALTH)))
        except ValueError as exc:
            self.log_message.emit("Degradation rejected: %s" % exc)
            return
        self.log_message.emit("Degradation #%d: joint %d %s, %s, start t=%.2f s."
                              % (scenario.id, scenario.joint + 1, scenario.name,
                                 scenario.describe()["meaning"], start))

    def _cmd_remove_fault(self, payload):
        if self.injector.remove(int(payload.get("id", -1))):
            self.log_message.emit("Removed fault #%s." % payload.get("id"))

    def _cmd_clear_faults(self, payload):
        if self.injector.has_items:
            self.injector.clear()
            self.log_message.emit("All faults and degradation cleared.")

    # -- logging ------------------------------------------------------------

    def _csv_meta(self):
        return {
            "robot_model": self.robot.urdf_path.replace("\\", "/"),
            "robot_id": self.robot.body_id,
            "joint_ids": list(self.robot.joint_indices),
            "joint_names": list(self.robot.joint_names),
            "fault_type": config.CSV_DEFAULT_FAULT_TYPE,
            "health_status": config.CSV_DEFAULT_HEALTH_STATUS,
        }

    @staticmethod
    def _timestamped_path(prefix):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(config.LOG_DIR, "%s_%s.csv" % (prefix, stamp))

    def _cmd_start_logging(self, payload):
        rate = int(payload.get("rate", config.CSV_DEFAULT_RATE_HZ))
        if rate not in config.CSV_RATES_HZ:
            self.log_message.emit("Unsupported CSV rate: %s Hz" % rate)
            return
        if self._csv_writer is not None:
            self._stop_logging_internal("restart")
        path = payload.get("path") or self._timestamped_path("telemetry")
        writer = CsvTelemetryWriter(path, self._csv_meta(),
                                    flush_interval=config.CSV_FLUSH_INTERVAL_S)
        try:
            writer.start()
        except (IOError, OSError) as exc:
            self.log_message.emit("Could not open CSV file %s: %s" % (path, exc))
            return
        self._csv_writer = writer
        self._csv_rate = rate
        self.recorder.set_csv_rate(rate)
        self.log_message.emit("Logging %d Hz to %s" % (rate, path))

    def _cmd_stop_logging(self, payload):
        if self._csv_writer is None:
            return
        self._stop_logging_internal("user")

    def _stop_logging_internal(self, reason):
        writer = self._csv_writer
        # Hand over samples already selected, then close in the background.
        writer.submit(self.recorder.take_csv_records())
        self.recorder.set_csv_rate(0)
        writer.finish()
        self._csv_writer = None
        self._csv_rate = 0
        self._finishing_writers.append((writer, "Logging (%s)" % reason))

    def _cmd_export_csv(self, payload):
        records = self.recorder.snapshot()
        if not records:
            self.log_message.emit("Nothing to export yet: the telemetry buffer is empty.")
            return
        path = payload.get("path") or self._timestamped_path("export")
        writer = CsvTelemetryWriter(path, self._csv_meta(),
                                    flush_interval=config.CSV_FLUSH_INTERVAL_S)
        try:
            writer.start()
        except (IOError, OSError) as exc:
            self.log_message.emit("Could not open export file %s: %s" % (path, exc))
            return
        writer.submit(records)
        writer.finish()
        self._finishing_writers.append((writer, "Export"))
        self.log_message.emit("Exporting %.1f s of 240 Hz telemetry (%d samples)..."
                              % (records[-1][0] - records[0][0] + config.FIXED_DT,
                                 len(records)))

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
