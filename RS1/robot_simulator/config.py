"""
Central configuration for the industrial robot simulator.

Every tunable number lives here, so timing, control and rendering behaviour can
be changed without touching simulation code. Values are plain module level
constants on purpose: they are read once at construction time and never mutated
from more than one thread.
"""

import math

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

# Physics runs at a fixed, deterministic rate. Nothing in the rendering or UI
# path is allowed to change this value at runtime.
PHYSICS_HZ = 240.0
FIXED_DT = 1.0 / PHYSICS_HZ

# Largest wall clock delta a single loop iteration may push into the
# accumulator. Protects against the "spiral of death" after the window was
# dragged, minimised, or the process was blocked by the OS.
MAX_FRAME_DELTA = 0.10

# Hard ceiling on physics steps executed in one loop iteration. It is set to
# exactly what MAX_FRAME_DELTA can produce, so an ordinary slow render frame is
# always caught up fully, and only a stall longer than the clamp can reach the
# ceiling. Anything beyond is dropped instead of replayed, so a hitch never
# turns into a burst of hundreds of catch-up steps.
MAX_STEPS_PER_ITERATION = int(MAX_FRAME_DELTA * PHYSICS_HZ)

# Rendering and UI rates. These are targets, not guarantees; physics never
# waits for them.
RENDER_FPS = 60.0
TELEMETRY_UI_HZ = 10.0
STATS_UI_HZ = 4.0

# Rate at which telemetry records enter the ring buffer inside the worker.
# Can be raised to PHYSICS_HZ or lowered without affecting the UI rates.
TELEMETRY_SAMPLE_HZ = 240.0
TELEMETRY_BUFFER_SECONDS = 10.0

# The GUI polls the frame mailbox faster than the render rate so a finished
# frame is picked up with minimal latency. The poll itself is a mutex lock and
# an integer compare, so it is close to free.
UI_POLL_HZ = 120.0

# Windows timer resolution request (milliseconds). 1 ms makes time.sleep()
# accurate enough that the accumulator rarely has to correct.
WINDOWS_TIMER_RESOLUTION_MS = 1

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

URDF_PATH = "kuka_iiwa/model.urdf"
ROBOT_BASE_POSITION = (0.0, 0.0, 0.0)
USE_FIXED_BASE = True
LOAD_GROUND_PLANE = True

# Home pose for the 7 axis KUKA iiwa. If a different URDF with a different
# joint count is loaded, the robot module falls back to an all zero pose.
HOME_POSE = (0.0, 0.35, 0.0, -1.40, 0.0, 1.20, 0.0)

# Solver settings. 50 iterations is generous for a 7 DOF arm and keeps the
# position controller from visibly sagging under gravity.
SOLVER_ITERATIONS = 50
GRAVITY = -9.81

# ---------------------------------------------------------------------------
# Joint control
# ---------------------------------------------------------------------------

# PyBullet POSITION_CONTROL gains. Deliberately soft: aggressive gains are the
# usual cause of buzzing and limit cycle oscillation in Bullet.
DEFAULT_POSITION_GAIN = 0.25
DEFAULT_VELOCITY_GAIN = 1.0

# Used only when the URDF does not declare a usable limit.
FALLBACK_MAX_FORCE = 300.0
FALLBACK_MAX_VELOCITY = 2.5

# Fractions of the URDF declared limits actually used.
FORCE_SAFETY_FACTOR = 0.90
VELOCITY_SAFETY_FACTOR = 0.60

# Commanded positions stay this far inside the URDF limits (radians).
JOINT_LIMIT_MARGIN = 0.02

# Emergency stop brakes with velocity control toward zero at this force.
BRAKE_FORCE = 150.0

# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

# Duration of the quintic blend used whenever the motion mode changes.
TRANSITION_TIME = 2.0

# Sine mode. Base frequency is scaled per joint and by the speed slider.
SINE_BASE_FREQUENCY = 0.18
SINE_JOINT_AMPLITUDE = (0.90, 0.45, 0.90, 0.50, 1.10, 0.55, 1.30)
SINE_JOINT_FREQ_SCALE = (0.60, 0.40, 0.75, 0.50, 0.90, 0.65, 1.05)

# Amplitude envelope ramp applied when the sine starts, so position, velocity
# and acceleration all leave zero smoothly.
SINE_RAMP_TIME = 2.5

# Speed and amplitude sliders are filtered before they reach the trajectory,
# so dragging a slider can never produce a step in the commanded motion.
SPEED_FILTER_TAU = 0.60
AMPLITUDE_FILTER_TAU = 0.60

# Manual mode uses a third order critically damped filter. Higher bandwidth
# tracks the slider more tightly but with sharper acceleration.
MANUAL_BANDWIDTH = 5.0

# Point to point mode.
P2P_SEGMENT_TIME = 3.0
P2P_DWELL_TIME = 0.40
P2P_WAYPOINTS = (
    (0.00, 0.35, 0.00, -1.40, 0.00, 1.20, 0.00),
    (1.20, 0.70, -0.40, -1.10, 0.60, 0.90, -0.80),
    (0.20, 1.00, 0.30, -0.60, -0.50, 1.30, 0.90),
    (-1.10, 0.55, 0.50, -1.50, 0.40, 1.00, 0.30),
    (-0.30, 0.10, -0.60, -1.90, -0.30, 1.40, -0.60),
)

SPEED_RANGE = (0.10, 2.00)
AMPLITUDE_RANGE = (0.05, 1.20)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

CAMERA_DEFAULT_YAW = 48.0
CAMERA_DEFAULT_PITCH = -28.0
CAMERA_DEFAULT_DISTANCE = 2.30
CAMERA_DEFAULT_TARGET = (0.0, 0.0, 0.55)

# Time constant of the exponential camera follow, in seconds. Smaller is
# snappier, larger is floatier. Smoothing is frame rate independent.
CAMERA_SMOOTH_TAU = 0.085

CAMERA_MIN_DISTANCE = 0.45
CAMERA_MAX_DISTANCE = 8.00
CAMERA_MIN_PITCH = -89.0
CAMERA_MAX_PITCH = 89.0

CAMERA_FOV = 55.0
CAMERA_NEAR = 0.05
CAMERA_FAR = 30.0

ORBIT_SENSITIVITY = 0.32      # degrees per pixel
PAN_SENSITIVITY = 0.0022      # metres per pixel per metre of distance
ZOOM_SENSITIVITY = 0.12       # fraction of distance per wheel notch

CAMERA_PRESETS = {
    "Home": (48.0, -28.0, 2.30, (0.0, 0.0, 0.55)),
    "Front": (90.0, -8.0, 2.30, (0.0, 0.0, 0.60)),
    "Side": (0.0, -8.0, 2.30, (0.0, 0.0, 0.60)),
    "Top": (90.0, -88.0, 2.60, (0.0, 0.0, 0.40)),
    "Isometric": (45.0, -35.0, 2.60, (0.0, 0.0, 0.55)),
}

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

QUALITY_PRESETS = {
    "LOW": {
        "max_width": 640,
        "max_height": 360,
        "shadows": False,
        "render_fps": 30.0,
    },
    "MEDIUM": {
        "max_width": 960,
        "max_height": 540,
        "shadows": False,
        "render_fps": 60.0,
    },
    "HIGH": {
        "max_width": 1280,
        "max_height": 720,
        "shadows": True,
        "render_fps": 60.0,
    },
}
QUALITY_ORDER = ("LOW", "MEDIUM", "HIGH")
DEFAULT_QUALITY = "MEDIUM"

# Try hardware OpenGL first and fall back to the software rasteriser if the
# probe frame comes back unusable. See rendering/pybullet_renderer.py.
PREFER_HARDWARE_OPENGL = True

# Smallest render target, used while the window is very small.
MIN_RENDER_WIDTH = 320
MIN_RENDER_HEIGHT = 180

# Resize events are debounced in the UI so dragging the window edge does not
# reallocate render buffers on every mouse move.
RESIZE_DEBOUNCE_MS = 200

# Adaptive rendering. Physics correctness is never traded away: the ladder only
# reduces render rate, then resolution scale, then shadows.
ADAPTIVE_RENDERING = True
ADAPTIVE_FPS_LADDER = (60.0, 45.0, 30.0, 20.0)
ADAPTIVE_SCALE_LADDER = (1.00, 0.80, 0.65, 0.50)

# A render is considered too expensive when it eats more than this fraction of
# its own frame period.
RENDER_BUDGET_FRACTION = 0.60

# Seconds of sustained overrun/headroom before the ladder moves.
ADAPTIVE_DEGRADE_DELAY = 1.5
ADAPTIVE_RECOVER_DELAY = 5.0

LIGHT_DIRECTION = (1.0, 1.0, 1.6)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

WINDOW_TITLE = "Industrial Robot Simulator - Phase 1"
WINDOW_MIN_SIZE = (1280, 800)
CONTROL_PANEL_WIDTH = 330
EVENT_LOG_MAX_LINES = 300

# Slider resolution for joint sliders (integer ticks per radian range).
SLIDER_TICKS = 1000

TELEMETRY_COLUMNS = (
    "Joint",
    "Position",
    "Target",
    "Error",
    "Velocity",
    "Torque",
    "Power",
)


def clamp(value, low, high):
    """Small helper used across modules."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def frame_rate_independent_alpha(dt, tau):
    """
    Exponential smoothing factor that behaves identically regardless of how
    often it is called. alpha = 1 - exp(-dt / tau).
    """
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)
