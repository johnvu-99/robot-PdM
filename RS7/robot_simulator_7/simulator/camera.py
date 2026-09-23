"""
Orbit camera.

Mouse input never moves the camera directly: it moves a desired state, and the
actual state chases it with frame rate independent exponential smoothing. That
is what removes the stutter you get from applying raw mouse deltas.
"""

import math

import numpy as np
import pybullet as p

import config


class OrbitCamera(object):

    def __init__(self):
        self.yaw = config.CAMERA_DEFAULT_YAW
        self.pitch = config.CAMERA_DEFAULT_PITCH
        self.distance = config.CAMERA_DEFAULT_DISTANCE
        self.target = np.array(config.CAMERA_DEFAULT_TARGET, dtype=np.float64)

        self.desired_yaw = self.yaw
        self.desired_pitch = self.pitch
        self.desired_distance = self.distance
        self.desired_target = self.target.copy()

        # Camera basis in world space, refreshed whenever the view matrix is
        # rebuilt. Used so panning follows the screen, not the world axes.
        self._right = np.array([1.0, 0.0, 0.0])
        self._up = np.array([0.0, 0.0, 1.0])

        self._view_matrix = None
        self._projection_matrix = None
        self._projection_aspect = -1.0

        self._rebuild_view()

    # -- input --------------------------------------------------------------

    def orbit(self, dx, dy):
        self.desired_yaw += dx * config.ORBIT_SENSITIVITY
        self.desired_pitch -= dy * config.ORBIT_SENSITIVITY
        self.desired_pitch = config.clamp(self.desired_pitch,
                                          config.CAMERA_MIN_PITCH,
                                          config.CAMERA_MAX_PITCH)

    def pan(self, dx, dy):
        scale = config.PAN_SENSITIVITY * self.distance
        self.desired_target = (self.desired_target
                               - self._right * (dx * scale)
                               + self._up * (dy * scale))

    def zoom(self, notches):
        factor = math.pow(1.0 - config.ZOOM_SENSITIVITY, notches)
        self.desired_distance = config.clamp(self.desired_distance * factor,
                                             config.CAMERA_MIN_DISTANCE,
                                             config.CAMERA_MAX_DISTANCE)

    def apply_preset(self, name, immediate=False):
        preset = config.CAMERA_PRESETS.get(name)
        if preset is None:
            return False
        yaw, pitch, distance, target = preset

        # Take the shortest way around instead of unwinding several turns.
        turns = round((self.yaw - yaw) / 360.0)
        self.desired_yaw = yaw + turns * 360.0
        self.desired_pitch = pitch
        self.desired_distance = distance
        self.desired_target = np.array(target, dtype=np.float64)

        if immediate:
            self.yaw = self.desired_yaw
            self.pitch = self.desired_pitch
            self.distance = self.desired_distance
            self.target = self.desired_target.copy()
            self._rebuild_view()
        return True

    def reset(self):
        return self.apply_preset("Home")

    # -- update -------------------------------------------------------------

    def update(self, dt):
        alpha = config.frame_rate_independent_alpha(dt, config.CAMERA_SMOOTH_TAU)
        self.yaw += (self.desired_yaw - self.yaw) * alpha
        self.pitch += (self.desired_pitch - self.pitch) * alpha
        self.distance += (self.desired_distance - self.distance) * alpha
        self.target += (self.desired_target - self.target) * alpha
        self._rebuild_view()

    def _rebuild_view(self):
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target.tolist(),
            distance=float(self.distance),
            yaw=float(self.yaw),
            pitch=float(self.pitch),
            roll=0.0,
            upAxisIndex=2,
        )
        m = self._view_matrix
        # Column major OpenGL view matrix: the rows of its rotation block are
        # the camera axes expressed in world space.
        self._right = np.array([m[0], m[4], m[8]], dtype=np.float64)
        self._up = np.array([m[1], m[5], m[9]], dtype=np.float64)

    # -- matrices -----------------------------------------------------------

    @property
    def view_matrix(self):
        return self._view_matrix

    def projection_matrix(self, aspect):
        """Rebuilt only when the aspect ratio actually changes."""
        if self._projection_matrix is None or abs(aspect - self._projection_aspect) > 1e-6:
            self._projection_matrix = p.computeProjectionMatrixFOV(
                fov=config.CAMERA_FOV,
                aspect=float(aspect),
                nearVal=config.CAMERA_NEAR,
                farVal=config.CAMERA_FAR,
            )
            self._projection_aspect = aspect
        return self._projection_matrix

    def state(self):
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "distance": self.distance,
            "target": self.target.tolist(),
        }
