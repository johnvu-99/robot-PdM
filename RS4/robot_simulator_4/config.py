"""
Central configuration for the industrial robot simulator.

Every tunable number lives here, so timing, control and rendering behaviour can
be changed without touching simulation code. Values are plain module level
constants on purpose: they are read once at construction time and never mutated
from more than one thread.
"""

import math
import os

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
TELEMETRY_BUFFER_SECONDS = 60.0

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
# Robot discovery (Phase 2)
# ---------------------------------------------------------------------------

# pybullet_data.getDataPath() is always scanned first. These extra folders are
# optional: a folder that does not exist is skipped silently.
EXTRA_MODEL_SEARCH_PATHS = (
    r"D:\bullet3-master\bullet3-master\data",
)

# Additional folders can be supplied without editing code, separated by
# os.pathsep (";" on Windows).
MODEL_SEARCH_ENV_VAR = "ROBOT_SIM_MODEL_PATHS"

MODEL_EXTENSIONS = (".urdf", ".sdf")
MODEL_SCAN_MAX_DEPTH = 5
MODEL_SCAN_MAX_FILES = 6000

# Bytes read from each file to estimate whether it is an articulated robot.
MODEL_SNIFF_BYTES = 262144

# Always listed first. Missing ones are shown greyed out instead of failing.
REQUIRED_MODELS = (
    "kuka_iiwa/model.urdf",
    "kuka_iiwa/model_free_base.urdf",
    "kuka_iiwa/model_vr_limits.urdf",
    "kuka_iiwa/model_for_sdf.urdf",
    "kuka_lwr/kuka.urdf",
)
PRIORITY_KEYWORDS = ("kuka", "iiwa", "lwr")

# ---------------------------------------------------------------------------
# Joint control modes (Phase 2)
# ---------------------------------------------------------------------------

DEFAULT_CONTROL_MODE = "POSITION_CONTROL"

# VELOCITY_CONTROL: v = v_ref + gain * (q_ref - q). Units 1/s.
VELOCITY_TRACKING_GAIN = 8.0

# TORQUE_CONTROL: computed torque, tau = ID(q, qd, qdd_ref + Kp e + Kd de).
# Kp = wn^2 and Kd = 2 wn gives a critically damped error at wn = 10 rad/s.
TORQUE_KP = 100.0
TORQUE_KD = 20.0

# Joint damping cap in torque mode, as a fraction of link inertia / dt.
# PyBullet applies URDF joint damping explicitly, and with the built in motor
# off it destabilises light links (the iiwa wrist: d = 0.5, I = 0.001). Tests
# showed any remaining damping degraded tracking through joint coupling, so the
# default removes it while torque mode is active; the computed torque Kd term
# provides the damping instead. URDF values are restored when leaving the mode.
TORQUE_DAMPING_STABILITY = 0.0

# Plain PD torque, used only if inverse dynamics is unavailable for the body.
TORQUE_FALLBACK_KP = 400.0
TORQUE_FALLBACK_KD = 40.0

# Blend time when a disabled joint is enabled again.
REENABLE_TRANSITION_TIME = 1.5

# Random pose: uniform within this fraction of each joint's half range around
# the home pose, clipped to the safe limits.
RANDOM_POSE_RANGE_FRACTION = 0.55

# Stress motion: large, fast, multi axis sweep that still respects URDF limits.
STRESS_JOINT_AMPLITUDE = (1.60, 0.70, 1.60, 0.60, 2.00, 0.90, 2.50)
STRESS_RANGE_FRACTION = 0.80
STRESS_VELOCITY_FRACTION = 0.85     # of the controller velocity limit
STRESS_BASE_FREQUENCY = 0.45        # Hz before per joint scaling and speed
STRESS_JOINT_FREQ_SCALE = (1.00, 0.83, 1.17, 0.91, 1.29, 1.07, 1.41)

# ---------------------------------------------------------------------------
# Telemetry, charts and logging (Phase 3)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")

# Charts receive decimated chunks at CHART_UI_HZ, holding samples taken at
# CHART_SAMPLE_HZ. 60 samples per second is far above what a 60 s window can
# show on screen, and keeps the GUI thread (and the GIL) lightly loaded.
CHART_SAMPLE_HZ = 60.0
CHART_UI_HZ = 25.0
CHART_WINDOWS_S = (10, 30, 60)
CHART_DEFAULT_WINDOW_S = 10

CSV_RATES_HZ = (50, 100, 240)
CSV_DEFAULT_RATE_HZ = 100
CSV_FLUSH_INTERVAL_S = 1.0

# Used only for rows without injector labels.
CSV_DEFAULT_FAULT_TYPE = "NORMAL"
CSV_DEFAULT_HEALTH_STATUS = "UNKNOWN"

# ---------------------------------------------------------------------------
# Fault injection and degradation (Phase 4)
# ---------------------------------------------------------------------------

# Severity is always 0..1. These constants give it physical meaning.
FAULT_FRICTION_MAX_FRACTION = 0.60     # friction torque at rated speed / rated torque
FAULT_STUCK_MIN_BAND_RAD = 0.002       # "fully seized" half band
FAULT_STUCK_MAX_BAND_RAD = 0.30        # half band at severity 0 (just stiff)
FAULT_BACKLASH_MAX_RAD = 0.08          # full dead band width at severity 1
FAULT_NOISE_MAX_STD_RAD = 0.010        # encoder noise sigma at severity 1
FAULT_NOISE_VELOCITY_GAIN = 25.0       # velocity noise sigma = position sigma * gain
FAULT_NOISE_TORQUE_FRACTION = 0.02     # torque sensor sigma / rated torque at severity 1
FAULT_BIAS_MAX_RAD = 0.10
FAULT_DRIFT_MAX_RAD_PER_S = 0.010
FAULT_OVERLOAD_MAX_KG = 12.0
FAULT_DEGRADATION_RAMP_S = 60.0        # torque degradation ramp for permanent faults
FAULT_BURST_ON_S = (0.15, 0.80)        # intermittent burst length range
FAULT_BURST_OFF_S = (0.50, 3.00)       # gap between bursts

# Motor weakness presets from the specification (available torque).
MOTOR_WEAKNESS_LEVELS = (100, 80, 60, 40, 20)

# A degradation mechanism appears in the label once its severity reaches this.
LABEL_SEVERITY_THRESHOLD = 0.05

# Ground truth health bands (health index in percent).
HEALTH_STATUS_BANDS = (
    (90.0, "HEALTHY"),
    (70.0, "MINOR_DEGRADATION"),
    (40.0, "WARNING"),
    (0.0, "CRITICAL"),
)

# Degradation scenarios: mechanism -> severity reached at the end of the run.
DEGRADATION_SCENARIOS = {
    "GEARBOX_WEAR": {"mechanisms": {"FRICTION": 0.9, "BACKLASH": 0.7}, "shape": 1.6},
    "MOTOR_AGEING": {"mechanisms": {"TORQUE_DEGRADATION": 0.6, "SENSOR_NOISE": 0.3}, "shape": 1.4},
    "ENCODER_DEGRADATION": {"mechanisms": {"SENSOR_DRIFT": 0.8, "SENSOR_NOISE": 0.6}, "shape": 1.2},
    "BEARING_WEAR": {"mechanisms": {"FRICTION": 0.7, "SENSOR_NOISE": 0.4}, "shape": 2.0},
    "PAYLOAD_CREEP": {"mechanisms": {"OVERLOAD": 0.9, "FRICTION": 0.3}, "shape": 1.0},
}
DEGRADATION_DEFAULT_DURATION_S = 120.0
DEGRADATION_DEFAULT_FINAL_HEALTH = 30.0

# ---------------------------------------------------------------------------
# Dataset generator (Phase 4)
# ---------------------------------------------------------------------------

SIMULATION_DATA_DIR = os.path.join(DATA_DIR, "simulation")
DATASET_CSV_RATE_HZ = 100
DATASET_SEED = 20240601

DATASET_PRESETS = {
    "QUICK": {
        "run_seconds": 12.0,
        "fault_onset_s": 4.0,
        "normal_patterns": ("SINE", "POINT_TO_POINT"),
        "normal_speeds": (1.0,),
        "fault_severities": (0.6,),
        "fault_joints": (1,),
        "degradation_scenarios": ("GEARBOX_WEAR",),
        "degradation_seconds": 40.0,
    },
    "STANDARD": {
        "run_seconds": 20.0,
        "fault_onset_s": 6.0,
        "normal_patterns": ("SINE", "POINT_TO_POINT", "STRESS"),
        "normal_speeds": (0.6, 1.0, 1.4),
        "fault_severities": (0.3, 0.6, 0.9),
        "fault_joints": (1, 3),
        "degradation_scenarios": ("GEARBOX_WEAR", "MOTOR_AGEING", "ENCODER_DEGRADATION",
                                  "BEARING_WEAR", "PAYLOAD_CREEP"),
        "degradation_seconds": 120.0,
    },
}
DATASET_FAULT_PATTERNS = ("SINE", "POINT_TO_POINT", "STRESS")

# The KUKA URDF declares 300 N m on every joint, several times what the arm
# needs (joint 2 peaks near 80 N m in Stress motion, the wrist below 5 N m).
# Capacity faults only become physically visible once available torque falls
# below demand, so generated datasets use higher severities for them. Without
# this, runs would carry fault labels for faults that leave no trace.
DATASET_CAPACITY_FAULT_SEVERITIES = (0.80, 0.90, 0.95)
DATASET_CAPACITY_FAULTS = ("MOTOR_WEAKNESS", "TORQUE_DEGRADATION", "INTERMITTENT_FAULT")

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

WINDOW_TITLE = "Industrial Robot Simulator - Phase 4"
WINDOW_MIN_SIZE = (1280, 800)
CONTROL_PANEL_WIDTH = 372
EVENT_LOG_MAX_LINES = 300

# Slider resolution for joint sliders (integer ticks per radian range).
SLIDER_TICKS = 1000

TELEMETRY_COLUMNS = (
    "Joint",
    "Position",
    "Target",
    "Error",
    "Velocity",
    "Accel",
    "Torque",
    "Power",
    "|F| N",
    "|M| Nm",
    "Fault (truth)",
    "Health (truth)",
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
