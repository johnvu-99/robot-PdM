"""
Fault injection and degradation simulation.

Every fault acts on the simulated plant, the actuator, or the sensor path.
Nothing edits telemetry after the fact:

MOTOR_WEAKNESS      available motor force scaled to (1 - severity)
FRICTION            extra viscous joint friction (changeDynamics jointDamping,
                    or subtracted from the motor torque in torque mode)
JOINT_STUCK         joint limits narrowed around the seizure position
BACKLASH            hysteresis (dead band) between motor command and joint
SENSOR_NOISE        Gaussian noise on position, velocity and torque sensors
SENSOR_BIAS         constant encoder offset
SENSOR_DRIFT        encoder offset growing linearly with time
OVERLOAD            unmodelled payload: external downward force on the link
TORQUE_DEGRADATION  motor force capacity falling linearly over the fault window
INTERMITTENT_FAULT  random bursts of motor power loss

Sensor faults feed back into control: every control mode servos on the
measured signal, so a biased encoder really moves the arm to the wrong place.

Runs on the simulation thread (or inside the dataset generator process). No Qt.
"""

import math

import numpy as np
import pybullet as p

import config

LABEL_NORMAL = "NORMAL"

MOTOR_WEAKNESS = "MOTOR_WEAKNESS"
FRICTION = "FRICTION"
JOINT_STUCK = "JOINT_STUCK"
BACKLASH = "BACKLASH"
SENSOR_NOISE = "SENSOR_NOISE"
SENSOR_BIAS = "SENSOR_BIAS"
SENSOR_DRIFT = "SENSOR_DRIFT"
OVERLOAD = "OVERLOAD"
TORQUE_DEGRADATION = "TORQUE_DEGRADATION"
INTERMITTENT_FAULT = "INTERMITTENT_FAULT"

FAULT_TYPES = (MOTOR_WEAKNESS, FRICTION, JOINT_STUCK, BACKLASH, SENSOR_NOISE,
               SENSOR_BIAS, SENSOR_DRIFT, OVERLOAD, TORQUE_DEGRADATION,
               INTERMITTENT_FAULT)

LABELS = (LABEL_NORMAL,) + FAULT_TYPES

FAULT_CAPTIONS = {
    MOTOR_WEAKNESS: "Motor weakness",
    FRICTION: "Increased friction",
    JOINT_STUCK: "Joint stuck",
    BACKLASH: "Backlash",
    SENSOR_NOISE: "Sensor noise",
    SENSOR_BIAS: "Sensor bias",
    SENSOR_DRIFT: "Sensor drift",
    OVERLOAD: "Overload",
    TORQUE_DEGRADATION: "Torque degradation",
    INTERMITTENT_FAULT: "Intermittent fault",
}


def describe_severity(fault_type, severity):
    """Human readable physical meaning of a severity in [0, 1]."""
    s = float(severity)
    if fault_type == MOTOR_WEAKNESS:
        return "%.0f%% torque available" % (100.0 * (1.0 - s))
    if fault_type == FRICTION:
        return "+%.0f%% of rated torque at rated speed" % (100.0 * s * config.FAULT_FRICTION_MAX_FRACTION)
    if fault_type == JOINT_STUCK:
        if s >= 0.999:
            return "fully seized"
        return "free travel +/-%.3f rad" % max((1.0 - s) * config.FAULT_STUCK_MAX_BAND_RAD, config.FAULT_STUCK_MIN_BAND_RAD)
    if fault_type == BACKLASH:
        return "%.3f rad dead band" % (s * config.FAULT_BACKLASH_MAX_RAD)
    if fault_type == SENSOR_NOISE:
        return "sigma %.4f rad" % (s * config.FAULT_NOISE_MAX_STD_RAD)
    if fault_type == SENSOR_BIAS:
        return "%+.3f rad offset" % (s * config.FAULT_BIAS_MAX_RAD)
    if fault_type == SENSOR_DRIFT:
        return "%.4f rad/s drift" % (s * config.FAULT_DRIFT_MAX_RAD_PER_S)
    if fault_type == OVERLOAD:
        return "+%.1f kg on link" % (s * config.FAULT_OVERLOAD_MAX_KG)
    if fault_type == TORQUE_DEGRADATION:
        return "capacity falls to %.0f%%" % (100.0 * (1.0 - s))
    if fault_type == INTERMITTENT_FAULT:
        return "bursts at %.0f%% torque" % (100.0 * (1.0 - s))
    return "%.0f%%" % (100.0 * s)


def health_status(health):
    for threshold, name in config.HEALTH_STATUS_BANDS:
        if health >= threshold:
            return name
    return config.HEALTH_STATUS_BANDS[-1][1]


class Fault(object):

    def __init__(self, fault_id, joint, fault_type, severity, start, duration, seed):
        self.id = int(fault_id)
        self.joint = int(joint)
        self.type = fault_type
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.start = float(start)
        self.duration = max(0.0, float(duration))
        self.rng = np.random.RandomState(seed)
        # Intermittent burst schedule, generated lazily in simulated time.
        self._burst_on = False
        self._burst_switch = self.start

    @property
    def end(self):
        return self.start + self.duration if self.duration > 0.0 else float("inf")

    def active(self, t):
        return self.start <= t < self.end

    def expired(self, t):
        return t >= self.end

    def progress(self, t):
        """0 at onset, 1 at the end of the window (or after the ramp time)."""
        span = self.duration if self.duration > 0.0 else config.FAULT_DEGRADATION_RAMP_S
        return float(np.clip((t - self.start) / max(span, 1e-6), 0.0, 1.0))

    def burst(self, t):
        while t >= self._burst_switch:
            self._burst_on = not self._burst_on
            low, high = (config.FAULT_BURST_ON_S if self._burst_on
                         else config.FAULT_BURST_OFF_S)
            self._burst_switch += self.rng.uniform(low, high)
        return self._burst_on

    def describe(self):
        return {
            "id": self.id,
            "kind": "fault",
            "joint": self.joint,
            "type": self.type,
            "severity": self.severity,
            "start": self.start,
            "duration": self.duration,
            "meaning": describe_severity(self.type, self.severity),
        }


class DegradationScenario(object):
    """
    Gradual wear. Health falls from 100 to final_health over the duration
    along (t / T) ** shape, and every mechanism's severity follows it up to its
    configured maximum.
    """

    def __init__(self, scenario_id, name, joint, mechanisms, start, duration,
                 final_health, shape):
        self.id = int(scenario_id)
        self.name = name
        self.joint = int(joint)
        self.mechanisms = dict(mechanisms)
        self.start = float(start)
        self.duration = max(1e-3, float(duration))
        self.final_health = float(np.clip(final_health, 0.0, 99.0))
        self.shape = float(shape)

    def health(self, t):
        if t < self.start:
            return 100.0
        x = min(1.0, (t - self.start) / self.duration)
        return 100.0 - (100.0 - self.final_health) * (x ** self.shape)

    def severities(self, t):
        loss = (100.0 - self.health(t)) / max(100.0 - self.final_health, 1e-6)
        return dict((kind, weight * loss) for kind, weight in self.mechanisms.items())

    def describe(self):
        return {
            "id": self.id,
            "kind": "degradation",
            "joint": self.joint,
            "type": self.name,
            "severity": 1.0 - self.final_health / 100.0,
            "start": self.start,
            "duration": self.duration,
            "meaning": "health 100%% -> %.0f%% over %.0f s (%s)" % (
                self.final_health, self.duration,
                ", ".join(sorted(self.mechanisms.keys()))),
        }


class FaultInjector(object):

    def __init__(self, client_id, robot, controller, dt, seed=None):
        self.client_id = client_id
        self.robot = robot
        self.controller = controller
        self.dt = float(dt)
        self.n = robot.n
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._next_id = 1

        self.faults = []
        self.scenarios = []

        n = self.n
        self._urdf_lower = np.array([j["lower"] for j in robot.all_joints if j["movable"]])
        self._urdf_upper = np.array([j["upper"] for j in robot.all_joints if j["movable"]])
        self._half_range = 0.5 * (robot.upper_limits - robot.lower_limits)

        # Link masses of each joint's child link, for sizing nothing but display.
        self._stuck_center = np.full(n, np.nan)
        self._backlash_output = None
        self._drift_start = np.zeros(n)

        # Last applied plant state, so PyBullet is only touched on change.
        self._applied_damping = np.zeros(n)
        self._applied_band = np.full(n, np.nan)
        self._applied_mode = None
        self._force_scales = np.ones(n)

        # Per step effect vectors.
        self.sensor_offset = np.zeros(n)
        self._noise_std = np.zeros(n)
        self._bias = np.zeros(n)
        self._load_kg = np.zeros(n)
        self._backlash_width = np.zeros(n)

        self._labels = self._normal_labels()
        self._label_key = None
        self.events = []

    # -- configuration ------------------------------------------------------

    def add_fault(self, joint, fault_type, severity, start, duration):
        if fault_type not in FAULT_TYPES:
            raise ValueError("unknown fault type %s" % fault_type)
        if not 0 <= int(joint) < self.n:
            raise ValueError("joint %s out of range" % joint)
        seed = self._rng.randint(0, 2 ** 31 - 1)
        fault = Fault(self._next_id, joint, fault_type, severity, start, duration, seed)
        self._next_id += 1
        self.faults.append(fault)
        return fault

    def add_scenario(self, name, joint, start, duration, final_health):
        preset = config.DEGRADATION_SCENARIOS.get(name)
        if preset is None:
            raise ValueError("unknown degradation scenario %s" % name)
        if not 0 <= int(joint) < self.n:
            raise ValueError("joint %s out of range" % joint)
        scenario = DegradationScenario(self._next_id, name, joint, preset["mechanisms"],
                                       start, duration, final_health,
                                       preset.get("shape", 1.5))
        self._next_id += 1
        self.scenarios.append(scenario)
        return scenario

    def remove(self, item_id):
        before = len(self.faults) + len(self.scenarios)
        self.faults = [f for f in self.faults if f.id != item_id]
        self.scenarios = [s for s in self.scenarios if s.id != item_id]
        return before != len(self.faults) + len(self.scenarios)

    def clear(self):
        self.faults = []
        self.scenarios = []
        self._restore_plant()
        self._labels = self._normal_labels()
        self._label_key = None

    def describe(self, t):
        items = []
        for item in list(self.faults) + list(self.scenarios):
            info = item.describe()
            if t < item.start:
                info["status"] = "scheduled"
            elif isinstance(item, Fault) and item.expired(t):
                info["status"] = "ended"
            else:
                info["status"] = "active"
            if isinstance(item, DegradationScenario):
                info["health"] = item.health(t)
            items.append(info)
        return items

    @property
    def has_items(self):
        return bool(self.faults or self.scenarios)

    # -- per step: actuator and plant ---------------------------------------

    def _collect(self, t):
        """Combine faults and scenarios into per joint mechanism severities."""
        n = self.n
        sev = dict((kind, np.zeros(n)) for kind in FAULT_TYPES)
        force_scale = np.ones(n)
        drift_elapsed = np.zeros(n)
        label_types = [[] for _ in range(n)]
        label_severity = np.zeros(n)
        health = np.full(n, 100.0)
        scenario_names = [[] for _ in range(n)]

        for fault in self.faults:
            if not fault.active(t):
                continue
            j = fault.joint
            s = fault.severity
            label_types[j].append(fault.type)
            label_severity[j] = max(label_severity[j], s)
            health[j] = min(health[j], 100.0 * (1.0 - s))
            if fault.type == TORQUE_DEGRADATION:
                force_scale[j] *= 1.0 - s * fault.progress(t)
            elif fault.type == INTERMITTENT_FAULT:
                if fault.burst(t):
                    force_scale[j] *= 1.0 - s
            elif fault.type == MOTOR_WEAKNESS:
                force_scale[j] *= 1.0 - s
            elif fault.type == SENSOR_DRIFT:
                sev[SENSOR_DRIFT][j] = max(sev[SENSOR_DRIFT][j], s)
                drift_elapsed[j] = max(drift_elapsed[j], t - fault.start)
            else:
                sev[fault.type][j] = max(sev[fault.type][j], s)

        for scenario in self.scenarios:
            if t < scenario.start:
                continue
            j = scenario.joint
            h = scenario.health(t)
            health[j] = min(health[j], h)
            scenario_names[j].append(scenario.name)
            for kind, s in scenario.severities(t).items():
                if s >= config.LABEL_SEVERITY_THRESHOLD:
                    label_types[j].append(kind)
                    label_severity[j] = max(label_severity[j], s)
                if kind in (MOTOR_WEAKNESS, TORQUE_DEGRADATION):
                    force_scale[j] *= 1.0 - s
                elif kind == SENSOR_DRIFT:
                    sev[SENSOR_DRIFT][j] = max(sev[SENSOR_DRIFT][j], s)
                    drift_elapsed[j] = max(drift_elapsed[j], t - scenario.start)
                else:
                    sev[kind][j] = max(sev[kind][j], s)

        self._update_labels(label_types, label_severity, health, scenario_names)
        return sev, force_scale, drift_elapsed

    def begin_step(self, t, true_positions, targets=None, target_velocities=None):
        """
        Apply actuator and plant faults for the step starting at time t.
        Returns (drive_targets, drive_velocities); both None when braking.
        """
        if not self.has_items:
            if self._plant_dirty():
                self._restore_plant()
            return targets, target_velocities

        sev, force_scale, drift_elapsed = self._collect(t)

        # Motor capacity.
        if np.any(np.abs(force_scale - self._force_scales) > 1e-6):
            self._force_scales = force_scale
            self.controller.set_force_scales(force_scale)

        # Friction: viscous coefficient relative to rated torque / rated speed.
        rated = self.controller.force_limits_nominal / np.maximum(self.controller.velocity_limits, 1e-6)
        damping = sev[FRICTION] * config.FAULT_FRICTION_MAX_FRACTION * rated
        self._apply_friction(damping)

        # Seizure: narrow the joint limits around where the joint was.
        stuck = sev[JOINT_STUCK]
        band = np.where(stuck > 0.0,
                        np.maximum((1.0 - stuck) * config.FAULT_STUCK_MAX_BAND_RAD,
                                   config.FAULT_STUCK_MIN_BAND_RAD),
                        np.nan)
        for j in range(self.n):
            if stuck[j] > 0.0 and np.isnan(self._stuck_center[j]):
                self._stuck_center[j] = float(true_positions[j])
            elif stuck[j] <= 0.0:
                self._stuck_center[j] = np.nan
        self._apply_stuck(band)

        # Unmodelled payload.
        self._load_kg = sev[OVERLOAD] * config.FAULT_OVERLOAD_MAX_KG
        for j in np.flatnonzero(self._load_kg > 0.0):
            link = self.robot.joint_indices[j]
            com = p.getLinkState(self.robot.body_id, link,
                                 physicsClientId=self.client_id)[0]
            p.applyExternalForce(self.robot.body_id, link,
                                 [0.0, 0.0, -abs(config.GRAVITY) * self._load_kg[j]],
                                 com, p.WORLD_FRAME, physicsClientId=self.client_id)

        # Sensors, applied in measure().
        self._noise_std = sev[SENSOR_NOISE] * config.FAULT_NOISE_MAX_STD_RAD
        self._bias = (sev[SENSOR_BIAS] * config.FAULT_BIAS_MAX_RAD
                      + sev[SENSOR_DRIFT] * config.FAULT_DRIFT_MAX_RAD_PER_S * drift_elapsed)

        if targets is None:
            return None, None

        # Backlash between motor command and joint.
        self._backlash_width = sev[BACKLASH] * config.FAULT_BACKLASH_MAX_RAD
        if not np.any(self._backlash_width > 0.0):
            self._backlash_output = None
            return targets, target_velocities
        x = np.asarray(targets, dtype=np.float64)
        if self._backlash_output is None:
            self._backlash_output = x.copy()
        half = 0.5 * self._backlash_width
        out = np.clip(self._backlash_output, x - half, x + half)
        engaged = np.abs(out - self._backlash_output) > 1e-12
        self._backlash_output = out
        velocities = np.array(target_velocities, dtype=np.float64)
        velocities[(self._backlash_width > 0.0) & ~engaged] = 0.0
        return out, velocities

    def _apply_friction(self, damping):
        torque_mode = self.controller.control_mode == "TORQUE_CONTROL"
        mode_changed = self._applied_mode != torque_mode
        self._applied_mode = torque_mode
        if torque_mode:
            # Damping is disabled in torque mode for stability; friction acts
            # as a torque loss on the motor command instead.
            self.controller.set_joint_friction(damping)
            self._applied_damping = damping
            return
        self.controller.set_joint_friction(np.zeros(self.n))
        base = self.controller.urdf_damping
        for j in range(self.n):
            if mode_changed or abs(damping[j] - self._applied_damping[j]) > 1e-6:
                p.changeDynamics(self.robot.body_id, self.robot.joint_indices[j],
                                 jointDamping=float(base[j] + damping[j]),
                                 physicsClientId=self.client_id)
        self._applied_damping = damping

    def _apply_stuck(self, band):
        for j in range(self.n):
            old = self._applied_band[j]
            new = band[j]
            same = (np.isnan(old) and np.isnan(new)) or (
                not np.isnan(old) and not np.isnan(new) and abs(old - new) < 1e-6)
            if same:
                continue
            link = self.robot.joint_indices[j]
            if np.isnan(new):
                p.changeDynamics(self.robot.body_id, link,
                                 jointLowerLimit=float(self._urdf_lower[j]),
                                 jointUpperLimit=float(self._urdf_upper[j]),
                                 physicsClientId=self.client_id)
            else:
                center = self._stuck_center[j]
                p.changeDynamics(self.robot.body_id, link,
                                 jointLowerLimit=float(center - new),
                                 jointUpperLimit=float(center + new),
                                 physicsClientId=self.client_id)
            self._applied_band[j] = new

    def _plant_dirty(self):
        return (np.any(self._force_scales != 1.0) or np.any(self._applied_damping != 0.0)
                or np.any(~np.isnan(self._applied_band)) or np.any(self.sensor_offset != 0.0)
                or self._label_key is not None)

    def _restore_plant(self):
        self._force_scales = np.ones(self.n)
        self.controller.set_force_scales(self._force_scales)
        self._apply_friction(np.zeros(self.n))
        self._stuck_center[:] = np.nan
        self._apply_stuck(np.full(self.n, np.nan))
        self._noise_std = np.zeros(self.n)
        self._bias = np.zeros(self.n)
        self._load_kg = np.zeros(self.n)
        self._backlash_width = np.zeros(self.n)
        self._backlash_output = None
        self.sensor_offset = np.zeros(self.n)
        self._labels = self._normal_labels()
        self._label_key = None

    # -- per step: sensors --------------------------------------------------

    def measure(self, positions, velocities, torques):
        """Return measured signals. Same objects when no sensor fault is active."""
        noisy = np.any(self._noise_std > 0.0)
        biased = np.any(self._bias != 0.0)
        if not noisy and not biased:
            if np.any(self.sensor_offset != 0.0):
                self.sensor_offset = np.zeros(self.n)
            return positions, velocities, torques
        q = np.array(positions, dtype=np.float64)
        qd = np.array(velocities, dtype=np.float64)
        tau = np.array(torques, dtype=np.float64)
        if biased:
            q += self._bias
        if noisy:
            std = self._noise_std
            q += self._rng.normal(0.0, 1.0, self.n) * std
            qd += self._rng.normal(0.0, 1.0, self.n) * std * config.FAULT_NOISE_VELOCITY_GAIN
            tau += (self._rng.normal(0.0, 1.0, self.n) * self.controller.force_limits_nominal
                    * config.FAULT_NOISE_TORQUE_FRACTION * (std / config.FAULT_NOISE_MAX_STD_RAD))
        self.sensor_offset = q - positions
        return q, qd, tau

    # -- labels -------------------------------------------------------------

    def _normal_labels(self):
        return tuple((LABEL_NORMAL, 0.0, 100.0, health_status(100.0), "")
                     for _ in range(self.n))

    def _update_labels(self, label_types, label_severity, health, scenario_names):
        labels = []
        for j in range(self.n):
            kinds = sorted(set(label_types[j]))
            name = "+".join(kinds) if kinds else LABEL_NORMAL
            h = round(float(health[j]), 1)
            labels.append((name, round(float(label_severity[j]), 3), h,
                           health_status(h), "+".join(scenario_names[j])))
        key = tuple(labels)
        if key != self._label_key:
            for j in range(self.n):
                old = self._labels[j][0] if j < len(self._labels) else LABEL_NORMAL
                if labels[j][0] != old:
                    self.events.append((j, old, labels[j][0]))
            self._label_key = key
            self._labels = key

    @property
    def labels(self):
        """Per joint (fault_type, severity, health, health_status, scenario). Immutable."""
        return self._labels

    def pop_events(self):
        events = self.events
        self.events = []
        return events
