"""
Joint space trajectory generation.

Every motion mode produces continuous position, velocity and acceleration.
Mode changes are blended with a quintic segment, slider changes are filtered,
and the sine envelope is ramped, so the commanded target never steps between
two physics frames.
"""

import numpy as np

import config

MODE_HOME = "HOME"
MODE_ZERO = "ZERO"
MODE_RANDOM = "RANDOM"
MODE_SINE = "SINE"
MODE_P2P = "POINT_TO_POINT"
MODE_STRESS = "STRESS"
MODE_MANUAL = "MANUAL"

MODES = (MODE_ZERO, MODE_HOME, MODE_RANDOM, MODE_SINE, MODE_P2P, MODE_STRESS,
         MODE_MANUAL)

MODE_LABELS = {
    MODE_ZERO: "Zero All",
    MODE_HOME: "Home",
    MODE_RANDOM: "Random Pose",
    MODE_SINE: "Sine Motion",
    MODE_P2P: "Trajectory Motion",
    MODE_STRESS: "Stress Motion",
    MODE_MANUAL: "Manual Sliders",
}

# Modes that move to one pose and hold it.
_HOLD_MODES = (MODE_HOME, MODE_ZERO, MODE_RANDOM)


def smoothstep5(s):
    """
    Quintic smoothstep h(s) = 10s^3 - 15s^4 + 6s^5 with its first and second
    derivatives with respect to s.

    h(0) = 0, h(1) = 1, and both derivatives vanish at the endpoints, which is
    what removes the velocity and acceleration discontinuities.
    """
    if s <= 0.0:
        return 0.0, 0.0, 0.0
    if s >= 1.0:
        return 1.0, 0.0, 0.0
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    h = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    hd = 30.0 * s2 - 60.0 * s3 + 30.0 * s4
    hdd = 60.0 * s - 180.0 * s2 + 120.0 * s3
    return h, hd, hdd


class QuinticSegment(object):
    """
    Minimum jerk interpolation that starts from an arbitrary state and ends at
    rest.

    The plain 10s^3 - 15s^4 + 6s^5 form assumes the motion starts from a
    standstill. Mode changes do not: leaving a sine sweep mid stroke means
    leaving it at speed, and forcing a zero start velocity there would put a
    step in the velocity command, which is exactly what produces a visible
    jolt. Solving the quintic for the real initial velocity and acceleration
    keeps all three continuous.
    """

    def __init__(self, q_start, q_end, duration, v_start=None, a_start=None):
        self.q_start = np.array(q_start, dtype=np.float64)
        self.q_end = np.array(q_end, dtype=np.float64)
        n = self.q_start.shape[0]
        v0 = np.zeros(n) if v_start is None else np.array(v_start, dtype=np.float64)
        a0 = np.zeros(n) if a_start is None else np.array(a_start, dtype=np.float64)

        self.duration = max(float(duration), 1e-3)
        self.t = 0.0

        t1 = self.duration
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t3 * t1
        t5 = t4 * t1
        delta = self.q_end - self.q_start

        # Boundary conditions: (q0, v0, a0) at t = 0, (q1, 0, 0) at t = T.
        self._c0 = self.q_start
        self._c1 = v0
        self._c2 = 0.5 * a0
        self._c3 = (20.0 * delta - 12.0 * v0 * t1 - 3.0 * a0 * t2) / (2.0 * t3)
        self._c4 = (-30.0 * delta + 16.0 * v0 * t1 + 3.0 * a0 * t2) / (2.0 * t4)
        self._c5 = (12.0 * delta - 6.0 * v0 * t1 - a0 * t2) / (2.0 * t5)

    @property
    def finished(self):
        return self.t >= self.duration

    def advance(self, dt):
        self.t += dt
        t = self.t
        if t > self.duration:
            t = self.duration
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        q = (self._c0 + self._c1 * t + self._c2 * t2
             + self._c3 * t3 + self._c4 * t4 + self._c5 * t5)
        qd = (self._c1 + 2.0 * self._c2 * t + 3.0 * self._c3 * t2
              + 4.0 * self._c4 * t3 + 5.0 * self._c5 * t4)
        qdd = (2.0 * self._c2 + 6.0 * self._c3 * t
               + 12.0 * self._c4 * t2 + 20.0 * self._c5 * t3)
        return q, qd, qdd


class TrajectoryGenerator(object):
    """
    Stateful generator advanced once per physics step with the fixed timestep.

    Because dt is constant the output is deterministic: the same command
    sequence always produces the same motion.
    """

    def __init__(self, n_joints, home_pose, lower_limits, upper_limits,
                 velocity_limits=None):
        self.n = int(n_joints)
        self.home = np.array(home_pose, dtype=np.float64)
        self.lower = np.array(lower_limits, dtype=np.float64)
        self.upper = np.array(upper_limits, dtype=np.float64)
        if velocity_limits is None:
            velocity_limits = np.full(self.n, config.FALLBACK_MAX_VELOCITY
                                      * config.VELOCITY_SAFETY_FACTOR)
        self.velocity_limits = np.array(velocity_limits, dtype=np.float64)
        self._safe_lower = self.lower + config.JOINT_LIMIT_MARGIN
        self._safe_upper = self.upper - config.JOINT_LIMIT_MARGIN
        self._half_range = 0.5 * np.maximum(self._safe_upper - self._safe_lower, 0.0)

        self.q = self.home.copy()
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)

        self.mode = MODE_HOME

        self._transition = None

        # Sine state.
        self._sine_phase = np.zeros(self.n)
        self._sine_time = 0.0
        self._amplitude_scale = self._sized(config.SINE_JOINT_AMPLITUDE, 0.6)
        self._frequency_scale = self._sized(config.SINE_JOINT_FREQ_SCALE, 0.5)

        # Stress state: amplitude around home, bounded by the limits.
        stress_amp = self._sized(config.STRESS_JOINT_AMPLITUDE, 0.0)
        default_amp = config.STRESS_RANGE_FRACTION * self._half_range
        stress_amp[len(config.STRESS_JOINT_AMPLITUDE):] = \
            default_amp[len(config.STRESS_JOINT_AMPLITUDE):]
        stress_amp = np.minimum(stress_amp, default_amp)
        stress_amp = np.minimum(stress_amp, self._safe_upper - self.home)
        stress_amp = np.minimum(stress_amp, self.home - self._safe_lower)
        self._stress_amplitude = np.maximum(stress_amp, 0.0)
        self._stress_freq_scale = self._sized(config.STRESS_JOINT_FREQ_SCALE, 1.0)

        # Hold state for HOME, ZERO and RANDOM.
        self._hold_target = self.home.copy()
        self._random_pose = self.home.copy()
        self._rng = np.random.RandomState()

        # Joints disabled by the operator are pinned to this pose.
        self._held = np.zeros(self.n, dtype=bool)
        self._held_pose = self.home.copy()

        # Point to point state. The configured waypoints only fit the 7 axis
        # KUKA; other robots get a deterministic spread within their limits.
        if self.n == len(config.P2P_WAYPOINTS[0]):
            waypoints = [self._sized(w, 0.0) for w in config.P2P_WAYPOINTS]
        else:
            joints = np.arange(self.n)
            waypoints = [self.home + 0.45 * self._half_range
                         * np.sin(1.7 * k + 0.9 * joints)
                         for k in range(len(config.P2P_WAYPOINTS))]
        self._waypoints = [self._clamped(w) for w in waypoints]
        self._p2p_index = 0
        self._p2p_segment = None
        self._p2p_dwell = 0.0

        # Manual state.
        self._manual_target = self.home.copy()

        # Filtered slider values.
        self._speed_command = 1.0
        self._speed = 1.0
        self._amplitude_command = 0.6
        self._amplitude = 0.6

    # -- helpers ------------------------------------------------------------

    def _sized(self, values, fill):
        """Coerce a config tuple to the actual joint count of the robot."""
        out = np.full(self.n, float(fill), dtype=np.float64)
        count = min(self.n, len(values))
        out[:count] = np.asarray(values[:count], dtype=np.float64)
        return out

    def _clamped(self, q):
        return np.clip(q, self.lower + config.JOINT_LIMIT_MARGIN,
                       self.upper - config.JOINT_LIMIT_MARGIN)

    # -- commands -----------------------------------------------------------

    def resync(self, q_measured):
        """
        Align the generator with the real robot state, used after reset,
        emergency stop release or a teleport.
        """
        self.q = np.array(q_measured, dtype=np.float64)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)
        self._transition = None
        self._p2p_segment = None
        self._p2p_dwell = 0.0
        self._sine_time = 0.0
        self._sine_phase[:] = 0.0
        self._manual_target = self.q.copy()
        self._hold_target = self.q.copy()
        self._held_pose[self._held] = self.q[self._held]

    def set_mode(self, mode, transition_time=None, regenerate=True):
        """
        Enter a mode through a quintic blend. regenerate=False re-enters the
        current mode without drawing a new random pose.
        """
        if mode not in MODES:
            return False
        if mode == MODE_RANDOM and regenerate:
            self._random_pose = self._make_random_pose()
        entry = self._entry_pose(mode)
        if mode in _HOLD_MODES:
            self._hold_target = entry.copy()
        duration = config.TRANSITION_TIME if transition_time is None else transition_time
        self.mode = mode
        self._reset_mode_state()
        at_rest = (np.max(np.abs(self.qd)) < 1e-4
                   and np.max(np.abs(self.qdd)) < 1e-4)
        if np.max(np.abs(entry - self.q)) < 1e-4 and at_rest:
            self._transition = None
        else:
            self._transition = QuinticSegment(self.q, entry, duration,
                                              self.qd, self.qdd)
        return True

    def _entry_pose(self, mode):
        if mode == MODE_HOME:
            return self.home.copy()
        if mode == MODE_ZERO:
            return self._clamped(np.zeros(self.n))
        if mode == MODE_RANDOM:
            return self._random_pose.copy()
        if mode in (MODE_SINE, MODE_STRESS):
            # The sine starts at its centre with a zero amplitude envelope, so
            # the centre is the only pose it can enter from smoothly.
            return self.home.copy()
        if mode == MODE_P2P:
            return np.array(self._waypoints[0], dtype=np.float64)
        return self.q.copy()

    def _reset_mode_state(self):
        self._sine_time = 0.0
        self._sine_phase[:] = 0.0
        self._p2p_index = 0
        self._p2p_segment = None
        self._p2p_dwell = 0.0
        if self.mode == MODE_MANUAL:
            self._manual_target = self.q.copy()

    def _make_random_pose(self):
        offsets = self._rng.uniform(-1.0, 1.0, self.n)
        pose = self.home + offsets * config.RANDOM_POSE_RANGE_FRACTION * self._half_range
        return self._clamped(pose)

    def set_hold(self, held_mask):
        """
        Pin disabled joints at their current commanded position. Newly held
        joints freeze where the reference is now, so position stays continuous.
        """
        mask = np.asarray(held_mask, dtype=bool)
        if mask.shape[0] != self.n:
            return
        newly_held = mask & ~self._held
        self._held_pose[newly_held] = self.q[newly_held]
        self._held = mask.copy()

    def set_speed(self, value):
        low, high = config.SPEED_RANGE
        self._speed_command = config.clamp(float(value), low, high)

    def set_amplitude(self, value):
        low, high = config.AMPLITUDE_RANGE
        self._amplitude_command = config.clamp(float(value), low, high)

    def set_manual_targets(self, targets):
        values = np.asarray(targets, dtype=np.float64)
        if values.shape[0] != self.n:
            return
        self._manual_target = self._clamped(values)

    def manual_targets(self):
        return self._manual_target.copy()

    # -- stepping -----------------------------------------------------------

    def step(self, dt):
        """Advance one fixed timestep and return (position, velocity, acceleration)."""
        alpha_speed = config.frame_rate_independent_alpha(dt, config.SPEED_FILTER_TAU)
        self._speed += (self._speed_command - self._speed) * alpha_speed
        alpha_amp = config.frame_rate_independent_alpha(dt, config.AMPLITUDE_FILTER_TAU)
        self._amplitude += (self._amplitude_command - self._amplitude) * alpha_amp

        if self._transition is not None:
            q, qd, qdd = self._transition.advance(dt)
            if self._transition.finished:
                self._transition = None
                self._sine_time = 0.0
                self._sine_phase[:] = 0.0
                if self.mode == MODE_MANUAL:
                    self._manual_target = q.copy()
        elif self.mode == MODE_SINE:
            q, qd, qdd = self._step_sine(dt)
        elif self.mode == MODE_STRESS:
            q, qd, qdd = self._step_stress(dt)
        elif self.mode == MODE_P2P:
            q, qd, qdd = self._step_p2p(dt)
        elif self.mode == MODE_MANUAL:
            q, qd, qdd = self._step_manual(dt)
        else:
            q, qd, qdd = self._hold_target.copy(), np.zeros(self.n), np.zeros(self.n)

        if self._held.any():
            q = np.array(q, dtype=np.float64)
            qd = np.array(qd, dtype=np.float64)
            qdd = np.array(qdd, dtype=np.float64)
            q[self._held] = self._held_pose[self._held]
            qd[self._held] = 0.0
            qdd[self._held] = 0.0

        self.q = self._clamped(q)
        self.qd = qd
        self.qdd = qdd
        return self.q, self.qd, self.qdd

    def _step_sine(self, dt):
        omega = 2.0 * np.pi * config.SINE_BASE_FREQUENCY * self._frequency_scale * self._speed
        amp = self._amplitude_scale * self._amplitude
        return self._step_periodic(dt, amp, omega)

    def _step_stress(self, dt):
        """
        Large amplitude, high speed sweep. Frequency is capped per joint so
        the peak velocity A * omega stays inside the controller velocity limit.
        """
        amp = self._stress_amplitude
        omega = (2.0 * np.pi * config.STRESS_BASE_FREQUENCY
                 * self._stress_freq_scale * self._speed)
        cap = (config.STRESS_VELOCITY_FRACTION * self.velocity_limits
               / np.maximum(amp, 1e-6))
        omega = np.minimum(omega, cap)
        return self._step_periodic(dt, amp, omega)

    def _step_periodic(self, dt, amp, omega):
        self._sine_time += dt

        ramp_time = max(config.SINE_RAMP_TIME, 1e-3)
        r, rd_s, rdd_s = smoothstep5(self._sine_time / ramp_time)
        rd = rd_s / ramp_time
        rdd = rdd_s / (ramp_time * ramp_time)

        # Integrating the phase rather than evaluating sin(w * t) means a
        # changing speed stays phase continuous.
        self._sine_phase += omega * dt

        sin_p = np.sin(self._sine_phase)
        cos_p = np.cos(self._sine_phase)

        q = self.home + amp * r * sin_p
        qd = amp * (rd * sin_p + r * omega * cos_p)
        qdd = amp * (rdd * sin_p + 2.0 * rd * omega * cos_p - r * omega * omega * sin_p)
        return q, qd, qdd

    def _step_p2p(self, dt):
        if self._p2p_segment is None:
            self._start_next_p2p_segment(first=True)

        q, qd, qdd = self._p2p_segment.advance(dt)
        if self._p2p_segment.finished:
            self._p2p_dwell += dt
            if self._p2p_dwell >= config.P2P_DWELL_TIME:
                self._p2p_dwell = 0.0
                self._p2p_index = (self._p2p_index + 1) % len(self._waypoints)
                self._start_next_p2p_segment(first=False)
        return q, qd, qdd

    def _start_next_p2p_segment(self, first):
        target = np.array(self._waypoints[self._p2p_index], dtype=np.float64)
        if first and np.max(np.abs(target - self.q)) < 1e-4:
            self._p2p_index = (self._p2p_index + 1) % len(self._waypoints)
            target = np.array(self._waypoints[self._p2p_index], dtype=np.float64)
        duration = config.P2P_SEGMENT_TIME / max(self._speed, 0.1)
        self._p2p_segment = QuinticSegment(self.q, target, duration,
                                           self.qd, self.qdd)

    def _step_manual(self, dt):
        """
        Third order critically damped filter (three poles at -w). Position,
        velocity and acceleration all stay continuous even when a slider is
        dragged in steps, which a plain low pass filter cannot promise.
        """
        w = config.MANUAL_BANDWIDTH
        error = self._manual_target - self.q
        jerk = (w ** 3) * error - 3.0 * (w ** 2) * self.qd - 3.0 * w * self.qdd
        qdd = self.qdd + jerk * dt
        qd = self.qd + qdd * dt
        q = self.q + qd * dt
        return q, qd, qdd

    # -- introspection ------------------------------------------------------

    @property
    def speed(self):
        return self._speed

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def held(self):
        return self._held.copy()

    @property
    def transitioning(self):
        return self._transition is not None
