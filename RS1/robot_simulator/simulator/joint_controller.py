"""
Joint level motor control.

All commands go out as a single batched call per physics step. Position,
velocity and force limits are enforced here, so no other module needs to know
about the URDF limits.
"""

import numpy as np
import pybullet as p

import config


class JointController(object):

    def __init__(self, client_id, robot):
        self.client_id = client_id
        self.robot = robot
        self.n = robot.n

        self.position_gain = config.DEFAULT_POSITION_GAIN
        self.velocity_gain = config.DEFAULT_VELOCITY_GAIN

        self._safe_lower = robot.lower_limits + config.JOINT_LIMIT_MARGIN
        self._safe_upper = robot.upper_limits - config.JOINT_LIMIT_MARGIN
        self._velocity_limit = robot.max_velocities * config.VELOCITY_SAFETY_FACTOR
        self._force_limit = robot.max_forces * config.FORCE_SAFETY_FACTOR

        # Python lists are what the PyBullet binding actually wants; building
        # them once avoids a conversion per step for the static arguments.
        self._forces = self._force_limit.tolist()
        self._brake_forces = [config.BRAKE_FORCE] * self.n
        self._zeros = [0.0] * self.n
        self._position_gains = [self.position_gain] * self.n
        self._velocity_gains = [self.velocity_gain] * self.n

        self._commanded = robot.home_pose.copy()
        self._commanded_velocity = np.zeros(self.n)

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

    # -- commands -----------------------------------------------------------

    def apply(self, target_positions, target_velocities):
        """Clamp, then send one batched POSITION_CONTROL command."""
        positions = np.clip(target_positions, self._safe_lower, self._safe_upper)
        velocities = np.clip(target_velocities,
                             -self._velocity_limit, self._velocity_limit)

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

        self._commanded = positions
        self._commanded_velocity = velocities
        return positions

    def engage_brakes(self):
        """
        Emergency stop. Velocity control toward zero with a limited force is
        the safe behaviour: the arm decelerates and holds instead of dropping
        under gravity or snapping to a stored setpoint.
        """
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.body_id,
            jointIndices=self.robot.joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=self._zeros,
            forces=self._brake_forces,
            physicsClientId=self.client_id,
        )

    def release_motors(self):
        """Zero force velocity control, used while the simulation is stopped."""
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.body_id,
            jointIndices=self.robot.joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=self._zeros,
            forces=self._zeros,
            physicsClientId=self.client_id,
        )

    # -- introspection ------------------------------------------------------

    @property
    def commanded_positions(self):
        return self._commanded

    @property
    def commanded_velocities(self):
        return self._commanded_velocity

    def tracking_error(self, measured_positions):
        return self._commanded - measured_positions
