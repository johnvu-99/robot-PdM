"""
Offscreen rendering.

PyBullet runs in DIRECT mode, so frames are produced with getCameraImage and
handed to Qt as raw RGBA bytes. getCameraImage must be called on the thread
that owns the client, which is why rendering lives in the simulation thread and
the finished frame crosses the thread boundary as data, not as a widget call.

Frames are published through a single slot mailbox rather than a Qt signal per
frame. A mailbox cannot build a backlog: if the GUI is busy it simply misses an
intermediate frame instead of queueing megabytes of stale images.
"""

import threading
import time

import numpy as np
import pybullet as p

import config


class FrameMailbox(object):
    """One slot, last writer wins, safe across threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self._width = 0
        self._height = 0
        self._serial = 0

    def put(self, data, width, height):
        with self._lock:
            self._data = data
            self._width = width
            self._height = height
            self._serial += 1

    def take(self, last_serial):
        """Return (serial, data, width, height) or None when nothing is new."""
        with self._lock:
            if self._data is None or self._serial == last_serial:
                return None
            return self._serial, self._data, self._width, self._height

    def clear(self):
        with self._lock:
            self._data = None


class PyBulletRenderer(object):

    def __init__(self, client_id, quality=None):
        self.client_id = client_id
        self.quality = quality or config.DEFAULT_QUALITY
        if self.quality not in config.QUALITY_PRESETS:
            self.quality = config.DEFAULT_QUALITY

        self.renderer = p.ER_TINY_RENDERER
        self.renderer_name = "TinyRenderer (software)"
        self.hardware_available = False

        self._requested_width = config.QUALITY_PRESETS[self.quality]["max_width"]
        self._requested_height = config.QUALITY_PRESETS[self.quality]["max_height"]
        self._width = self._requested_width
        self._height = self._requested_height

        self._levels = []
        self._level = 0
        self._over_budget_time = 0.0
        self._under_budget_time = 0.0

        self._light_direction = list(config.LIGHT_DIRECTION)

        self._rebuild_levels()
        self._apply_level()

    # -- renderer selection -------------------------------------------------

    def select_renderer(self):
        """
        Pick between hardware OpenGL and the software rasteriser by measuring
        both, not by trusting the flag.

        In DIRECT mode Bullet has no OpenGL context, so ER_BULLET_HARDWARE_OPENGL
        either returns a blank buffer or is quietly serviced by the software
        path anyway. Timing both and taking the faster one means the viewport is
        never black and the reported renderer name is never a fiction.

        Returns (name, tiny_ms, hardware_ms); hardware_ms is None when the
        hardware path produced nothing usable.
        """
        tiny_ms = self._time_renderer(p.ER_TINY_RENDERER)
        self.renderer = p.ER_TINY_RENDERER
        self.renderer_name = "TinyRenderer (software)"
        self.hardware_available = False

        if not config.PREFER_HARDWARE_OPENGL:
            return self.renderer_name, tiny_ms, None

        hardware_ms = self._time_renderer(p.ER_BULLET_HARDWARE_OPENGL)
        if hardware_ms is None:
            return self.renderer_name, tiny_ms, None

        # Only claim hardware when it is meaningfully faster. A tie means the
        # software path is servicing the call under another name.
        if tiny_ms is None or hardware_ms < tiny_ms * 0.7:
            self.renderer = p.ER_BULLET_HARDWARE_OPENGL
            self.renderer_name = "Hardware OpenGL"
            self.hardware_available = True
        return self.renderer_name, tiny_ms, hardware_ms

    def _time_renderer(self, renderer, size=256, samples=3):
        """Median frame time in milliseconds, or None if unusable."""
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(config.CAMERA_DEFAULT_TARGET),
            distance=config.CAMERA_DEFAULT_DISTANCE,
            yaw=config.CAMERA_DEFAULT_YAW,
            pitch=config.CAMERA_DEFAULT_PITCH,
            roll=0.0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=config.CAMERA_FOV, aspect=1.0,
            nearVal=config.CAMERA_NEAR, farVal=config.CAMERA_FAR,
        )
        durations = []
        for _ in range(samples):
            start = time.perf_counter()
            try:
                result = p.getCameraImage(
                    size, size,
                    viewMatrix=view,
                    projectionMatrix=projection,
                    renderer=renderer,
                    flags=p.ER_NO_SEGMENTATION_MASK,
                    physicsClientId=self.client_id,
                )
            except Exception:
                return None
            durations.append(time.perf_counter() - start)
            pixels = self._as_array(result, size, size)
            if pixels is None:
                return None
            # A usable frame contains both the robot and the background.
            if int(pixels[:, :, :3].max()) - int(pixels[:, :, :3].min()) < 8:
                return None
        durations.sort()
        return 1000.0 * durations[len(durations) // 2]

    # -- sizing -------------------------------------------------------------

    def set_quality(self, quality):
        if quality not in config.QUALITY_PRESETS or quality == self.quality:
            return False
        self.quality = quality
        self._level = 0
        self._over_budget_time = 0.0
        self._under_budget_time = 0.0
        self._rebuild_levels()
        self._apply_level()
        return True

    def reset_adaptive(self):
        """Return to the top of the quality ladder, used on Reset."""
        self._level = 0
        self._over_budget_time = 0.0
        self._under_budget_time = 0.0
        self._apply_level()

    def set_viewport(self, width, height):
        """Called on resize. Debounced by the UI, not here."""
        self._requested_width = max(int(width), config.MIN_RENDER_WIDTH)
        self._requested_height = max(int(height), config.MIN_RENDER_HEIGHT)
        self._apply_level()

    def _rebuild_levels(self):
        preset = config.QUALITY_PRESETS[self.quality]
        top_fps = preset["render_fps"]
        shadows = preset["shadows"]
        fps_ladder = config.ADAPTIVE_FPS_LADDER
        scale_ladder = config.ADAPTIVE_SCALE_LADDER

        # Degrade in the order the brief asks for: frame rate first, then
        # resolution, then visual extras. Physics is never in this ladder.
        self._levels = [
            (min(top_fps, fps_ladder[0]), scale_ladder[0], shadows),
            (min(top_fps, fps_ladder[1]), scale_ladder[0], shadows),
            (min(top_fps, fps_ladder[2]), scale_ladder[0], shadows),
            (min(top_fps, fps_ladder[2]), scale_ladder[1], shadows),
            (min(top_fps, fps_ladder[2]), scale_ladder[2], False),
            (min(top_fps, fps_ladder[3]), scale_ladder[3], False),
        ]

    def _apply_level(self):
        preset = config.QUALITY_PRESETS[self.quality]
        fps, scale, shadows = self._levels[self._level]

        width = min(self._requested_width, preset["max_width"])
        height = min(self._requested_height, preset["max_height"])
        width = int(width * scale)
        height = int(height * scale)

        # Even dimensions keep the software rasteriser's row handling simple
        # and avoid odd one pixel seams after scaling in the widget.
        width = max(config.MIN_RENDER_WIDTH, width - (width % 2))
        height = max(config.MIN_RENDER_HEIGHT, height - (height % 2))

        self._width = width
        self._height = height
        self.target_fps = fps
        self.shadows = 1 if shadows else 0

    # -- adaptive control ---------------------------------------------------

    def note_render_duration(self, duration, elapsed):
        """
        Move one step down the ladder after a sustained overrun, and one step
        back up after a long stretch of headroom. Returns a message when the
        level changed, otherwise None.
        """
        if not config.ADAPTIVE_RENDERING:
            return None

        budget = config.RENDER_BUDGET_FRACTION / max(self.target_fps, 1.0)
        if duration > budget:
            self._over_budget_time += elapsed
            self._under_budget_time = 0.0
        elif duration < budget * 0.5:
            self._under_budget_time += elapsed
            self._over_budget_time = 0.0
        else:
            self._over_budget_time = 0.0
            self._under_budget_time = 0.0

        if (self._over_budget_time > config.ADAPTIVE_DEGRADE_DELAY
                and self._level < len(self._levels) - 1):
            self._level += 1
            self._over_budget_time = 0.0
            self._apply_level()
            return "Render load high, stepping down to %.0f FPS at %dx%d" % (
                self.target_fps, self._width, self._height)

        if (self._under_budget_time > config.ADAPTIVE_RECOVER_DELAY
                and self._level > 0):
            self._level -= 1
            self._under_budget_time = 0.0
            self._apply_level()
            return "Render headroom available, stepping up to %.0f FPS at %dx%d" % (
                self.target_fps, self._width, self._height)

        return None

    # -- rendering ----------------------------------------------------------

    def render(self, camera):
        """
        Produce one frame. Returns (rgba_bytes, width, height, duration).

        The only per frame allocation is the byte copy handed to Qt; matrices,
        light vectors and size values are all reused.
        """
        width = self._width
        height = self._height
        aspect = float(width) / float(height)

        start = time.perf_counter()
        result = p.getCameraImage(
            width,
            height,
            viewMatrix=camera.view_matrix,
            projectionMatrix=camera.projection_matrix(aspect),
            lightDirection=self._light_direction,
            shadow=self.shadows,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=self.renderer,
            physicsClientId=self.client_id,
        )
        pixels = self._as_array(result, width, height)
        duration = time.perf_counter() - start

        if pixels is None:
            return None, width, height, duration
        return pixels.tobytes(), width, height, duration

    @staticmethod
    def _as_array(result, width, height):
        """
        getCameraImage returns a numpy array when PyBullet was built with numpy
        support and a flat sequence otherwise. Handle both.
        """
        actual_width = int(result[0])
        actual_height = int(result[1])
        pixels = result[2]
        if not isinstance(pixels, np.ndarray):
            pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.dtype != np.uint8:
            pixels = pixels.astype(np.uint8)
        try:
            pixels = pixels.reshape(actual_height, actual_width, 4)
        except ValueError:
            try:
                pixels = pixels.reshape(height, width, 4)
            except ValueError:
                return None
        return np.ascontiguousarray(pixels)

    # -- introspection ------------------------------------------------------

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def describe(self):
        return {
            "renderer": self.renderer_name,
            "quality": self.quality,
            "width": self._width,
            "height": self._height,
            "target_fps": self.target_fps,
            "shadows": bool(self.shadows),
            "level": self._level,
        }
