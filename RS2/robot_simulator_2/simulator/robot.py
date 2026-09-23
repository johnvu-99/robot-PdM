"""
Robot model wrapper.

Loads a URDF or SDF once, caches everything static about it (every joint with
its type and limits, movable joint indices) and exposes batched state access.
Nothing here touches Qt, and every call must happen on the thread that owns the
PyBullet client.
"""

import math
import os

import numpy as np
import pybullet as p

import config


def _joint_type_names():
    names = {}
    for attribute, caption in (
            ("JOINT_REVOLUTE", "REVOLUTE"),
            ("JOINT_PRISMATIC", "PRISMATIC"),
            ("JOINT_SPHERICAL", "SPHERICAL"),
            ("JOINT_PLANAR", "PLANAR"),
            ("JOINT_FIXED", "FIXED"),
            ("JOINT_POINT2POINT", "POINT2POINT"),
            ("JOINT_GEAR", "GEAR")):
        value = getattr(p, attribute, None)
        if value is not None:
            names[value] = caption
    return names


JOINT_TYPE_NAMES = _joint_type_names()


class RobotLoadError(Exception):
    """Raised when a model cannot be turned into a usable robot."""


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


class Robot(object):
    """A single articulated robot inside one PyBullet client."""

    def __init__(self, client_id, urdf_path, base_position=(0.0, 0.0, 0.0),
                 fixed_base=True):
        self.client_id = client_id
        self.urdf_path = urdf_path
        self.base_position = list(base_position)
        self.fixed_base = bool(fixed_base)
        self.file_type = os.path.splitext(urdf_path)[1].lower().lstrip(".") or "urdf"

        self.body_id = -1
        self._base_constraint = -1
        self.joint_indices = []
        self.joint_names = []
        self.joint_types = []
        self.n = 0

        # Every joint of the body, including fixed ones, for the joint table.
        self.all_joints = []

        self.lower_limits = np.zeros(0)
        self.upper_limits = np.zeros(0)
        self.max_forces = np.zeros(0)
        self.max_velocities = np.zeros(0)
        self.home_pose = np.zeros(0)

        # True when calculateInverseDynamics can be used: base fixed through
        # loadURDF and only revolute/prismatic degrees of freedom.
        self.supports_inverse_dynamics = False

        # Preallocated state buffers, refilled in place every read.
        self._positions = np.zeros(0)
        self._velocities = np.zeros(0)
        self._torques = np.zeros(0)

    # -- lifecycle ----------------------------------------------------------

    def load(self):
        try:
            if self.file_type == "sdf":
                self.body_id = self._load_sdf()
            else:
                self.body_id = p.loadURDF(
                    self.urdf_path,
                    basePosition=self.base_position,
                    useFixedBase=self.fixed_base,
                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                    physicsClientId=self.client_id,
                )
        except p.error as exc:
            raise RobotLoadError("PyBullet could not load %s (%s)"
                                 % (self.urdf_path, exc))
        if self.body_id < 0:
            raise RobotLoadError("PyBullet returned no body for %s" % self.urdf_path)

        self._scan_joints()
        self._resolve_home_pose()
        movable_types = (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
        dof_joints = [j for j in self.all_joints if j["type"] != "FIXED"]
        self.supports_inverse_dynamics = (
            self.n > 0 and self.fixed_base and self._base_constraint < 0
            and len(dof_joints) == self.n
            and all(t in movable_types for t in self.joint_types))
        if self.n > 0:
            self.reset_to(self.home_pose)
        return self.body_id

    def _load_sdf(self):
        """
        An SDF can hold several bodies. Keep the one with the most movable
        joints, remove the rest, and pin its base to the world if requested,
        because loadSDF has no useFixedBase argument.
        """
        bodies = p.loadSDF(self.urdf_path, physicsClientId=self.client_id)
        if not bodies:
            raise RobotLoadError("SDF contains no bodies: %s" % self.urdf_path)

        best = -1
        best_count = -1
        for body in bodies:
            count = 0
            for index in range(p.getNumJoints(body, physicsClientId=self.client_id)):
                info = p.getJointInfo(body, index, physicsClientId=self.client_id)
                if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    count += 1
            if count > best_count:
                best, best_count = body, count
        for body in bodies:
            if body != best:
                p.removeBody(body, physicsClientId=self.client_id)

        _, orientation = p.getBasePositionAndOrientation(
            best, physicsClientId=self.client_id)
        p.resetBasePositionAndOrientation(best, self.base_position, orientation,
                                          physicsClientId=self.client_id)
        if self.fixed_base:
            self._base_constraint = p.createConstraint(
                best, -1, -1, -1, p.JOINT_FIXED,
                jointAxis=[0.0, 0.0, 0.0],
                parentFramePosition=[0.0, 0.0, 0.0],
                childFramePosition=self.base_position,
                childFrameOrientation=orientation,
                physicsClientId=self.client_id)
        return best

    def unload(self):
        """Remove the body from the world. Safe to call on a partial load."""
        errors = []
        if self._base_constraint >= 0:
            try:
                p.removeConstraint(self._base_constraint, physicsClientId=self.client_id)
            except p.error as exc:
                errors.append("constraint: %s" % exc)
            self._base_constraint = -1
        if self.body_id >= 0:
            try:
                p.removeBody(self.body_id, physicsClientId=self.client_id)
            except p.error as exc:
                errors.append("body: %s" % exc)
            self.body_id = -1
        return errors

    def _scan_joints(self):
        indices = []
        names = []
        types = []
        lower = []
        upper = []
        forces = []
        velocities = []
        all_joints = []

        joint_count = p.getNumJoints(self.body_id, physicsClientId=self.client_id)
        for index in range(joint_count):
            info = p.getJointInfo(self.body_id, index, physicsClientId=self.client_id)
            joint_type = info[2]
            raw_low = float(info[8])
            raw_high = float(info[9])
            raw_force = float(info[10])
            raw_velocity = float(info[11])
            movable = joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)

            entry = {
                "id": index,
                "name": _decode(info[1]),
                "type": JOINT_TYPE_NAMES.get(joint_type, str(joint_type)),
                "lower": raw_low,
                "upper": raw_high,
                "max_force": raw_force,
                "max_velocity": raw_velocity,
                "movable": movable,
                "movable_index": len(indices) if movable else -1,
                "limits_fallback": False,
                "force_fallback": False,
                "velocity_fallback": False,
            }
            all_joints.append(entry)
            if not movable:
                continue

            low, high = raw_low, raw_high
            if high <= low:
                # Continuous joint, or a URDF that declares no usable range.
                low, high = -math.pi, math.pi
                entry["limits_fallback"] = True

            max_force = raw_force
            if max_force <= 0.0:
                max_force = config.FALLBACK_MAX_FORCE
                entry["force_fallback"] = True

            max_velocity = raw_velocity
            if max_velocity <= 0.0:
                max_velocity = config.FALLBACK_MAX_VELOCITY
                entry["velocity_fallback"] = True

            indices.append(index)
            names.append(entry["name"])
            types.append(joint_type)
            lower.append(low)
            upper.append(high)
            forces.append(max_force)
            velocities.append(max_velocity)

        self.all_joints = all_joints
        self.joint_indices = indices
        self.joint_names = names
        self.joint_types = types
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
        """Static description handed to the UI after every load."""
        return {
            "names": list(self.joint_names),
            "joint_ids": list(self.joint_indices),
            "lower": self.lower_limits.tolist(),
            "upper": self.upper_limits.tolist(),
            "max_force": self.max_forces.tolist(),
            "max_velocity": self.max_velocities.tolist(),
            "home": self.home_pose.tolist(),
            "count": self.n,
            "urdf": self.urdf_path,
            "file_type": self.file_type,
            "fixed_base": self.fixed_base,
            "inverse_dynamics": self.supports_inverse_dynamics,
            "all_joints": [dict(j) for j in self.all_joints],
        }
