"""
Joint level motor control.

All commands go out as a single batched call per physics step. Position,
velocity and force limits are enforced here, so no other module needs to know
about the URDF limits.

Every control mode tracks the same smooth reference (q_ref, qd_ref, qdd_ref)
from the trajectory generator. Only the way the reference is realised differs:

POSITION_CONTROL  PyBullet position servo, as in Phase 1.
VELOCITY_CONTROL  v = qd_ref + k (q_ref - q), sent to the velocity motor.
TORQUE_CONTROL    computed torque: tau = ID(q, qd, qdd_ref + Kp e + Kd de).
"""

import numpy as np
import pybullet as p

import config

CONTROL_POSITION = "POSITION_CONTROL"
CONTROL_VELOCITY = "VELOCITY_CONTROL"
CONTROL_TORQUE = "TORQUE_CONTROL"
CONTROL_MODES = (CONTROL_POSITION, CONTROL_VELOCITY, CONTROL_TORQUE)

# Low level motor state last sent to PyBullet.
_MOTOR_POSITION = "position"
_MOTOR_VELOCITY = "velocity"
_MOTOR_TORQUE = "torque"
_MOTOR_BRAKE = "brake"
_MOTOR_RELEASED = "released"


class JointController(object):

    def __init__(self, client_id, robot):
        self.client_id = client_id
        self.robot = robot
        self.n = robot.n

        self.position_gain = config.DEFAULT_POSITION_GAIN
        self.velocity_gain = config.DEFAULT_VELOCITY_GAIN
        self.control_mode = CONTROL_POSITION

        self._safe_lower = robot.lower_limits + config.JOINT_LIMIT_MARGIN
        self._safe_upper = robot.upper_limits - config.JOINT_LIMIT_MARGIN
        self._velocity_limit = robot.max_velocities * config.VELOCITY_SAFETY_FACTOR
        self._force_limit = robot.max_forces * config.FORCE_SAFETY_FACTOR
        self._force_nominal = self._force_limit.copy()
        self._joint_friction = np.zeros(self.n)

        # Python lists are what the PyBullet binding actually wants; building
        # them once avoids a conversion per step for the static arguments.
        self._forces = self._force_limit.tolist()
        self._brake_forces = [config.BRAKE_FORCE] * self.n
        self._zeros = [0.0] * self.n
        self._position_gains = [self.position_gain] * self.n
        self._velocity_gains = [self.velocity_gain] * self.n

        self._commanded = robot.home_pose.copy()
        self._commanded_velocity = np.zeros(self.n)
        self._applied_torque = np.zeros(self.n)

        self._motor_state = None
        self._inverse_dynamics_ok = bool(robot.supports_inverse_dynamics)
        self._warnings = []

        # URDF joint damping, and the largest damping that stays numerically
        # stable once the built in motor is off. See _limit_damping_for_torque.
        self._urdf_damping = np.zeros(self.n)
        self._stable_damping = np.zeros(self.n)
        self._damping_limited = False
        dt = config.FIXED_DT
        for i, joint_index in enumerate(robot.joint_indices):
            info = p.getJointInfo(robot.body_id, joint_index,
                                  physicsClientId=client_id)
            dynamics = p.getDynamicsInfo(robot.body_id, joint_index,
                                         physicsClientId=client_id)
            inertia = min(float(v) for v in dynamics[2]) if dynamics[2] else 0.0
            self._urdf_damping[i] = float(info[6])
            self._stable_damping[i] = config.TORQUE_DAMPING_STABILITY * inertia / dt

    # -- tuning -------------------------------------------------------------

    def set_gains(self, position_gain=None, velocity_gain=None):
        if position_gain is not None:
            self.position_gain = float(position_gain)
            self._position_gains = [self.position_gain] * self.n
        if velocity_gain is not None:
            self.velocity_gain = float(velocity_gain)
            self._velocity_gains = [self.velocity_gain] * self.n

    def set_force_scale(self, scale):
        scale = config.clamp(float(scale), 0.05, 1.0)
        self._force_limit = self.robot.max_forces * config.FORCE_SAFETY_FACTOR * scale
        self._forces = self._force_limit.tolist()

    def set_force_scales(self, scales):
        """Per joint available force fraction, used by fault injection."""
        scales = np.clip(np.asarray(scales, dtype=np.float64), 0.0, 1.0)
        self._force_limit = self._force_nominal * scales
        self._forces = self._force_limit.tolist()

    def set_joint_friction(self, coefficients):
        """Viscous friction subtracted from the motor torque in torque mode."""
        self._joint_friction = np.asarray(coefficients, dtype=np.float64)

    def clamp_positions(self, positions):
        return np.clip(positions, self._safe_lower, self._safe_upper)

    @property
    def force_limits_nominal(self):
        return self._force_nominal

    @property
    def urdf_damping(self):
        return self._urdf_damping

    def set_control_mode(self, mode):
        if mode not in CONTROL_MODES:
            return False
        if mode == self.control_mode:
            return True
        if mode == CONTROL_TORQUE and not self._inverse_dynamics_ok:
            # Plain PD torque without a dynamics model is unstable on light
            # links at 240 Hz, so torque mode is refused rather than degraded.
            self._warnings.append(
                "Torque mode needs inverse dynamics, which is unavailable for "
                "this body (free or constraint pinned base, or non revolute/"
                "prismatic joints). Staying in %s." % self.control_mode)
            return False
        self.control_mode = mode
        if mode == CONTROL_TORQUE:
            self._limit_damping_for_torque()
        else:
            self._restore_damping()
        # Disable the built in motors so a stale position or velocity servo can
        # never fight the new mode. TORQUE_CONTROL requires this.
        self.release_motors()
        return True

    def _limit_damping_for_torque(self):
        """
        PyBullet integrates joint damping explicitly. With the motor constraint
        active (position and velocity modes) that is harmless, but in pure
        torque mode it destabilises light links: the KUKA iiwa wrist has
        d = 0.5 on I = 0.001, so d * dt / I is about 2 at dt = 1/240. Damping
        is capped to TORQUE_DAMPING_STABILITY * I / dt (zero by default) while
        torque mode is active, and the URDF values are restored afterwards.
        """
        capped = np.minimum(self._urdf_damping, self._stable_damping)
        changed = np.flatnonzero(capped < self._urdf_damping - 1e-12)
        for i in changed:
            p.changeDynamics(self.robot.body_id, self.robot.joint_indices[i],
                             jointDamping=float(capped[i]),
                             physicsClientId=self.client_id)
        if changed.size:
            self._damping_limited = True
            self._warnings.append(
                "Torque mode: URDF joint damping reduced for numerical stability on "
                "joint(s) %s (URDF %s -> %s). Restored when leaving torque mode."
                % (", ".join(str(i + 1) for i in changed),
                   ", ".join("%.3f" % self._urdf_damping[i] for i in changed),
                   ", ".join("%.3f" % capped[i] for i in changed)))

    def _restore_damping(self):
        if not self._damping_limited:
            return
        for i, joint_index in enumerate(self.robot.joint_indices):
            p.changeDynamics(self.robot.body_id, joint_index,
                             jointDamping=float(self._urdf_damping[i]),
                             physicsClientId=self.client_id)
        self._damping_limited = False

    # -- commands -----------------------------------------------------------

    def apply(self, target_positions, target_velocities, target_accelerations=None,
              measured_positions=None, measured_velocities=None,
              sensor_offset=None, true_velocities=None):
        """
        Clamp the reference, then send one batched command for the mode.

        sensor_offset is measured minus true position. PyBullet's position servo
        reads the true joint state internally, so the offset is applied to its
        target: the arm then settles where the (faulty) encoder says the target
        is, exactly as a real servo on that encoder would.
        """
        positions = np.clip(target_positions, self._safe_lower, self._safe_upper)
        velocities = np.clip(target_velocities,
                             -self._velocity_limit, self._velocity_limit)
        self._commanded = positions
        self._commanded_velocity = velocities

        if self.control_mode == CONTROL_VELOCITY:
            self._apply_velocity(positions, velocities, measured_positions)
        elif self.control_mode == CONTROL_TORQUE:
            self._apply_torque(positions, velocities, target_accelerations,
                               measured_positions, measured_velocities, true_velocities)
        else:
            servo = positions
            if sensor_offset is not None:
                servo = positions - sensor_offset
            self._apply_position(servo, velocities)
        return positions

    def _apply_position(self, positions, velocities):
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.body_id,
            jointIndices=self.robot.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=positions.tolist(),
            targetVelocities=velocities.tolist(),
            forces=self._forces,
            positionGains=self._position_gains,
            velocityGains=self._velocity_gains,
            physicsClientId=self.client_id,
        )
        self._motor_state = _MOTOR_POSITION

    def _apply_velocity(self, positions, velocities, measured_positions):
        q = positions if measured_positions is None else np.asarray(measured_positions)
        command = velocities + config.VELOCITY_TRACKING_GAIN * (positions - q)
        command = np.clip(command, -self._velocity_limit, self._velocity_limit)
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.body_id,
            jointIndices=self.robot.joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=command.tolist(),
            forces=self._forces,
            physicsClientId=self.client_id,
        )
        self._motor_state = _MOTOR_VELOCITY

    def _apply_torque(self, positions, velocities, accelerations,
                      measured_positions, measured_velocities, true_velocities=None):
        if self._motor_state != _MOTOR_TORQUE:
            self.release_motors()

        q = positions if measured_positions is None else np.asarray(measured_positions)
        qd = velocities if measured_velocities is None else np.asarray(measured_velocities)
        qdd_ref = np.zeros(self.n) if accelerations is None else np.asarray(accelerations)
        error = positions - q
        error_rate = velocities - qd

        torque = None
        if self._inverse_dynamics_ok:
            desired = qdd_ref + config.TORQUE_KP * error + config.TORQUE_KD * error_rate
            try:
                torque = np.asarray(p.calculateInverseDynamics(
                    self.robot.body_id, q.tolist(), qd.tolist(), desired.tolist(),
                    physicsClientId=self.client_id), dtype=np.float64)
                if torque.shape[0] != self.n:
                    raise ValueError("inverse dynamics returned %d values for %d joints"
                                     % (torque.shape[0], self.n))
            except (p.error, ValueError, TypeError) as exc:
                self._inverse_dynamics_ok = False
                torque = None
                self._warnings.append(
                    "Inverse dynamics unavailable (%s). Torque mode falls back to "
                    "PD without gravity compensation." % exc)
        if torque is None:
            torque = (config.TORQUE_FALLBACK_KP * error
                      + config.TORQUE_FALLBACK_KD * error_rate)

        torque = np.clip(torque, -self._force_limit, self._force_limit)
        joint_torque = torque
        if np.any(self._joint_friction):
            speed = qd if true_velocities is None else np.asarray(true_velocities)
            joint_torque = torque - self._joint_friction * speed
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.body_id,
            jointIndices=self.robot.joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=joint_torque.tolist(),
            physicsClientId=self.client_id,
        )
        self._applied_torque = torque
        self._motor_state = _MOTOR_TORQUE

    def engage_brakes(self):
        """
        Emergency stop. Velocity control toward zero with a limited force is
        the safe behaviour: the arm decelerates and holds instead of dropping
        under gravity or snapping to a stored setpoint. Works from every mode.
        """
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.body_id,
            jointIndices=self.robot.joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=self._zeros,
            forces=self._brake_forces,
            physicsClientId=self.client_id,
        )
        self._motor_state = _MOTOR_BRAKE

    def release_motors(self):
        """Zero force velocity control: used while stopped and before torque mode."""
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.body_id,
            jointIndices=self.robot.joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=self._zeros,
            forces=self._zeros,
            physicsClientId=self.client_id,
        )
        self._motor_state = _MOTOR_RELEASED

    # -- introspection ------------------------------------------------------

    @property
    def commanded_positions(self):
        return self._commanded

    @property
    def commanded_velocities(self):
        return self._commanded_velocity

    @property
    def applied_torques(self):
        """
        PyBullet reports zero applied torque for TORQUE_CONTROL motors, so the
        commanded torque is the measurement in that mode. None otherwise.
        """
        if self._motor_state == _MOTOR_TORQUE:
            return self._applied_torque
        return None

    @property
    def velocity_limits(self):
        return self._velocity_limit.copy()

    def pop_warnings(self):
        warnings = self._warnings
        self._warnings = []
        return warnings

    def tracking_error(self, measured_positions):
        return self._commanded - measured_positions
