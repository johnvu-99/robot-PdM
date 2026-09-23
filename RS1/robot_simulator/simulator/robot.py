"""
Robot model wrapper.

Loads a URDF once, caches everything static about it (movable joint indices,
limits, names) and exposes batched state access. Nothing here touches Qt, and
every call must happen on the thread that owns the PyBullet client.
"""

import math

import numpy as np
import pybullet as p

import config


class Robot(object):
    """A single articulated robot inside one PyBullet client."""

    def __init__(self, client_id, urdf_path, base_position=(0.0, 0.0, 0.0),
                 fixed_base=True):
        self.client_id = client_id
        self.urdf_path = urdf_path
        self.base_position = list(base_position)
        self.fixed_base = bool(fixed_base)

        self.body_id = -1
        self.joint_indices = []
        self.joint_names = []
        self.n = 0

        self.lower_limits = np.zeros(0)
        self.upper_limits = np.zeros(0)
        self.max_forces = np.zeros(0)
        self.max_velocities = np.zeros(0)
        self.home_pose = np.zeros(0)

        # Preallocated state buffers, refilled in place every read.
        self._positions = np.zeros(0)
        self._velocities = np.zeros(0)
        self._torques = np.zeros(0)

    # -- lifecycle ----------------------------------------------------------

    def load(self):
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.body_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_position,
            useFixedBase=self.fixed_base,
            flags=flags,
            physicsClientId=self.client_id,
        )
        self._scan_joints()
        self._resolve_home_pose()
        self.reset_to(self.home_pose)
        return self.body_id

    def _scan_joints(self):
        indices = []
        names = []
        lower = []
        upper = []
        forces = []
        velocities = []

        joint_count = p.getNumJoints(self.body_id, physicsClientId=self.client_id)
        for index in range(joint_count):
            info = p.getJointInfo(self.body_id, index, physicsClientId=self.client_id)
            joint_type = info[2]
            if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue

            name = info[1]
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")

            low = float(info[8])
            high = float(info[9])
            if high <= low:
                # Continuous joint, or a URDF that declares no usable range.
                low, high = -math.pi, math.pi

            max_force = float(info[10])
            if max_force <= 0.0:
                max_force = config.FALLBACK_MAX_FORCE

            max_velocity = float(info[11])
            if max_velocity <= 0.0:
                max_velocity = config.FALLBACK_MAX_VELOCITY

            indices.append(index)
            names.append(name)
            lower.append(low)
            upper.append(high)
            forces.append(max_force)
            velocities.append(max_velocity)

        self.joint_indices = indices
        self.joint_names = names
        self.n = len(indices)
        self.lower_limits = np.array(lower, dtype=np.float64)
        self.upper_limits = np.array(upper, dtype=np.float64)
        self.max_forces = np.array(forces, dtype=np.float64)
        self.max_velocities = np.array(velocities, dtype=np.float64)

        self._positions = np.zeros(self.n)
        self._velocities = np.zeros(self.n)
        self._torques = np.zeros(self.n)

    def _resolve_home_pose(self):
        pose = np.zeros(self.n)
        configured = config.HOME_POSE
        if len(configured) == self.n:
            pose = np.array(configured, dtype=np.float64)
        pose = np.clip(pose,
                       self.lower_limits + config.JOINT_LIMIT_MARGIN,
                       self.upper_limits - config.JOINT_LIMIT_MARGIN)
        self.home_pose = pose

    # -- state --------------------------------------------------------------

    def get_states(self):
        """
        One batched call instead of n individual ones. Returns references to
        internal buffers, so callers must copy if they need to keep the data.
        """
        states = p.getJointStates(self.body_id, self.joint_indices,
                                  physicsClientId=self.client_id)
        for i, state in enumerate(states):
            self._positions[i] = state[0]
            self._velocities[i] = state[1]
            self._torques[i] = state[3]
        return self._positions, self._velocities, self._torques

    def reset_to(self, positions):
        """
        Teleport. Only valid for initialisation, reset and explicit fault
        injection; normal motion always goes through the motor controller.
        """
        values = np.asarray(positions, dtype=np.float64)
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.body_id, joint_index,
                              targetValue=float(values[i]),
                              targetVelocity=0.0,
                              physicsClientId=self.client_id)
        self._positions[:] = values
        self._velocities[:] = 0.0
        self._torques[:] = 0.0

    def describe(self):
        """Static description handed to the UI once, at startup."""
        return {
            "names": list(self.joint_names),
            "lower": self.lower_limits.tolist(),
            "upper": self.upper_limits.tolist(),
            "max_force": self.max_forces.tolist(),
            "max_velocity": self.max_velocities.tolist(),
            "home": self.home_pose.tolist(),
            "count": self.n,
            "urdf": self.urdf_path,
        }
