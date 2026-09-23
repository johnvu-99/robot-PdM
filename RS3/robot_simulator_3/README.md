# Industrial Robot Simulator - Phase 3

A KUKA iiwa simulator built for smooth, deterministic motion first and features
second. Fixed 240 Hz physics on a dedicated thread, offscreen PyBullet
rendering, PyQt5 front end.

## Project layout

```
robot_simulator/
├── main.py                        entry point, high DPI setup
├── config.py                      every tunable constant
├── requirements.txt
│
├── simulator/
│   ├── physics_engine.py          worker thread, accumulator loop, state machine
│   ├── robot.py                   URDF/SDF load, joint inspection, batched state reads
│   ├── robot_loader.py            model discovery (pybullet_data, bullet3, env var)
│   ├── joint_controller.py        position / velocity / torque control, limits, brakes
│   ├── trajectory.py              quintic segments, sine, point to point, manual
│   └── camera.py                  orbit camera with smoothed follow
│
├── monitoring/
│   └── telemetry.py               240 Hz deque ring buffer, chart/CSV decimation
│
├── storage/
│   └── csv_logger.py              buffered CSV writer on its own thread
│
├── rendering/
│   └── pybullet_renderer.py       offscreen frames, adaptive quality, frame mailbox
│
└── ui/
    ├── main_window.py             layout, thread ownership, signal wiring
    ├── simulator_view.py          viewport widget, mouse input
    ├── controls_panel.py          motion patterns, gains, view, telemetry tabs, log
    ├── charts_panel.py            PyQtGraph realtime charts
    ├── robot_panel.py             model dropdown, discovery thread, control mode
    └── joint_panel.py             per joint slider/enable, full joint table
```

Dependencies point one way only: `ui` knows about `simulator`, `simulator` never
imports anything from `ui`.

## Install (Windows 11, Python 3.7.9)

From a normal Command Prompt or PowerShell, in the project folder:

```
py -3.7 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade "pip==24.0" setuptools wheel
pip install -r requirements.txt
```

`pip 24.0` is the last release that runs on Python 3.7, which is why the upgrade
is pinned.

Install order matters. `numpy` is listed first in `requirements.txt` on purpose:
when pybullet is built or installed with numpy already present, `getCameraImage`
returns a numpy array instead of a flat Python list, which is the difference
between a fast frame copy and a very slow one.

If pip decides to build `pybullet` from source instead of using a wheel, install
**Visual Studio Build Tools 2019 or newer** with the "Desktop development with
C++" workload first, then run the install again. The build takes several
minutes.

Verify the install:

```
python -c "import pybullet, PyQt5, numpy; print(pybullet.getAPIVersion())"
```

## Run

```
.venv\Scripts\activate
python main.py
```

## Physics and render timing

The two clocks are fully separate.

```
Qt main thread                       Simulation thread
├── controls                         ├── command queue (deque)
├── viewport widget                  ├── accumulator loop
├── telemetry table                  ├── PyBullet DIRECT client
└── 120 Hz mailbox poll  <-- frames --┤── 240 Hz physics
        ^                             ├── trajectory + controller
        └── signals at 4-10 Hz -------┘── offscreen render at 30-60 Hz
```

**Fixed timestep accumulator.** Each loop iteration measures real elapsed time
with `time.perf_counter()`, adds it to an accumulator, and consumes it in exact
`1/240` chunks:

```
delta = now - last                  clamped to MAX_FRAME_DELTA (0.1 s)
accumulator += delta
while accumulator >= dt and steps < 8:
    step physics with dt
    accumulator -= dt
if accumulator >= dt:               backlog beyond 8 steps is discarded
    accumulator = 0
```

Two guards prevent the spiral of death. The clamp stops a long stall (window
drag, minimise, OS scheduling hitch) from injecting a huge delta, and the
8 step ceiling stops any single iteration from replaying a burst of catch-up
steps. Whatever is left after those is thrown away rather than executed later,
so the robot never fast-forwards after a hitch.

**No busy waiting.** The loop sleeps on a `threading.Event` until the next
step deadline. A posted command sets the event and wakes the loop immediately.
On Windows the worker asks for 1 ms timer resolution (`timeBeginPeriod`) so the
sleep lands near the deadline instead of drifting by a scheduler tick; the
accumulator absorbs whatever error remains, so the long run average stays at
exactly 240 Hz.

**Thread boundary.** PyBullet is touched from one thread only. Frames cross to
the GUI through a single slot mailbox rather than a signal per frame: if the UI
stalls, an intermediate frame is skipped instead of two megabytes of stale
images piling up in the event queue. Telemetry arrives as decimated snapshots at
10 Hz, performance figures at 4 Hz. Nothing is emitted at 240 Hz.

**Pause and resume.** Pause holds the last commanded pose, stops stepping and
zeroes the accumulator, so resuming cannot produce a time spike. The camera keeps
smoothing against wall clock time while paused, so the view stays responsive.

**Why there is no render side interpolation.** `getCameraImage` renders the
world as it currently is; PyBullet has no way to rasterise a pose that has not
been committed to the simulation. Faking it would mean calling
`resetJointState` to an interpolated pose before the frame and restoring after,
which mutates the true state, destroys the solver's velocity history and is
exactly the jitter source the brief rules out. The practical alternative used
here is to render immediately after the physics steps of that iteration, so the
displayed state is at most one 4.17 ms step old. At 60 FPS that residual is far
below the perception threshold, and because trajectories are quintic, the motion
between sampled states is already smooth.

## Motion smoothness

- Normal motion only ever goes through `setJointMotorControlArray` in
  `POSITION_CONTROL`. `resetJointState` is used only for load, reset and stop.
- Every mode change is blended with a quintic segment
  `h(s) = 10s³ - 15s⁴ + 6s⁵`, whose first and second derivatives vanish at both
  ends, so position, velocity and acceleration are all continuous.
- The sine mode integrates phase instead of evaluating `sin(ωt)`, so changing
  the speed slider bends the motion instead of stepping it, and its amplitude
  is ramped by the same quintic envelope from zero.
- Speed and amplitude sliders pass through a time constant filter before they
  reach the generator.
- Manual joint sliders drive a third order critically damped filter, so even a
  dragged slider produces continuous acceleration.
- Targets are clamped inside the URDF limits with a margin, velocities to 60%
  of the URDF maximum, forces to 90%.

## Expected performance

Physics is cheap; rendering is what costs.

Measured on a deliberately weak two core virtual machine with no GPU, running
the full application for 13 seconds in sine mode:

| Metric | Measured |
| --- | --- |
| Physics rate | mean 240.1 Hz, window range 236.8 - 243.3 Hz |
| Real time factor | mean 1.001, range 0.986 - 1.014 |
| Physics step duration | 0.07 - 0.14 ms for the 7 axis KUKA plus ground plane |
| Dropped physics backlog | 0 |
| Joint tracking error | 0.0007 rad holding, 0.004 rad at normal sweep speed, 0.015 rad at maximum speed and amplitude |
| Render, 640x360 software | 27 ms per frame, 22 FPS on that machine |

The important result is the second row: rendering on that machine was slow
enough to miss its frame target continuously, and the physics rate did not move.
That is the separation working.

On a normal Windows 11 desktop expect the physics figures to be the same or
slightly better, and the render figures to be roughly two to four times faster,
since the rasteriser is single threaded and scales with core speed and pixel
count:

| Preset | Frame | Typical software render |
| --- | --- | --- |
| LOW | 640x360 | 8 - 15 ms, holds 30 FPS |
| MEDIUM | 960x540 | 18 - 40 ms, 30 - 55 FPS |
| HIGH | 1280x720 with shadows | 40 - 90 ms, 15 - 25 FPS |

Physics costs roughly 240 x 0.1 ms = 25 ms of CPU per second, about 2 to 5% of
one core. Everything else in the process is the rasteriser, so total CPU is
close to one saturated core at MEDIUM and well under that at LOW.

**About hardware rendering.** `p.connect(p.DIRECT)` creates no OpenGL context,
so on Windows `ER_BULLET_HARDWARE_OPENGL` has nothing real to draw into. Rather
than trusting the flag, the renderer times a 256 pixel probe frame through both
paths at startup, checks that each produced an actual image, and keeps hardware
only when it is meaningfully faster. Both probe times are written to the event
log, so you can see exactly what was chosen and why. On most Windows installs
the answer will be TinyRenderer, and the software path returning a valid image
under the hardware flag is precisely the case that would otherwise be
mislabelled.

**One thing to know about the frame clamp.** `MAX_FRAME_DELTA` is 0.1 s, as
specified. It exists for stalls: window drags, minimise, OS hitches. If a single
render frame ever takes longer than that, the excess is treated as a stall and
discarded, and the real time factor dips below 1.00 until adaptive rendering
brings the frame time back down. If you want a very heavy render setting to stay
at 1.00x instead, raise `MAX_FRAME_DELTA` in `config.py` to 0.25;
`MAX_STEPS_PER_ITERATION` follows it automatically and 60 catch-up steps still
cost under 10 ms.

## Controls

| Action | Input |
| --- | --- |
| Orbit | Left drag |
| Pan | Middle or right drag |
| Zoom | Wheel |
| Camera presets | Keys 1-5, or the View buttons |
| Reset view | Double click the viewport |

Emergency stop latches: it brakes the joints with velocity control toward zero
and blocks motion until Reset is pressed.

## Phase 2: multi robot and joint control

The Phase 1 timing loop, threading model, render mailbox and accumulator are
unchanged. Everything new enters the worker through the existing command queue.

**Robot discovery.** On startup a short lived QThread (never the physics
thread) scans `pybullet_data`, `D:\bullet3-master\bullet3-master\data` if it
exists, and any folders in the `ROBOT_SIM_MODEL_PATHS` environment variable for
`*.urdf` and `*.sdf`. Required KUKA models are listed first; missing ones appear
greyed out as "(not found)". "Robots only" hides files with no movable joints.
Paths live in `config.py`.

Note: the pip `pybullet_data` does not ship `kuka_lwr/kuka.urdf`, and its
`kuka_iiwa/model_for_sdf.urdf` references meshes that are not included. Both
load from the full bullet3 `data` folder. A model that fails to load is
reported in the event log and the current robot stays loaded.

**Loading.** Select a model and press Load. The new robot is fully built before
the old body is removed, then the simulation returns to Stopped at home. SDF
files keep the body with the most movable joints and pin its base with a fixed
constraint. The Joints tab lists every joint (ID, name, type, position, limits,
max force, max velocity); values replaced by defaults are marked.

**Control modes.** All three modes follow the same smooth quintic/sine
reference, so switching mode never changes the motion itself.

| Mode | Realisation |
| --- | --- |
| POSITION_CONTROL | PyBullet position servo (Phase 1 behaviour) |
| VELOCITY_CONTROL | `v = qd_ref + 8 (q_ref - q)`, clamped to the velocity limit |
| TORQUE_CONTROL | computed torque `tau = ID(q, qd, qdd_ref + Kp e + Kd de)`, clamped to max force |

Torque mode needs `calculateInverseDynamics`, so it is refused (with a log
message) for bodies without it, such as constraint pinned SDF models. While
torque mode is active, URDF joint damping is set to zero and restored on exit:
PyBullet applies joint damping explicitly, and on the iiwa wrist
(damping 0.5, inertia 0.001) that is unstable at 240 Hz once the motor
constraint is off. The controller's Kd term provides the damping instead.

**Motion patterns.** Zero All, Home, Random Pose (new pose every click), Sine
Motion, Trajectory Motion (point to point cycle), Stress Motion (large, fast
sweep with each joint's frequency capped so peak velocity stays at 85% of the
velocity limit), and Manual Sliders.

**Per joint.** Each joint has an enable checkbox, slider, target, actual
position, velocity and torque. A disabled joint holds where its reference was
when disabled; enabling it re-enters the current pattern through a 1.5 s blend.
In torque mode the displayed torque is the commanded torque, because PyBullet
reports zero for TORQUE_CONTROL motors.

**Measured on the dev machine** (all four loadable iiwa URDFs, 4 s per pattern,
max tracking error after settling):

| Pattern | Position | Velocity | Torque |
| --- | --- | --- | --- |
| Hold poses | 0.000 rad | 0.000 rad | 0.001 rad |
| Sine / Trajectory | 0.003 rad | 0.003 rad | 0.003 rad |
| Stress | 0.020 rad | 0.018 rad | 0.022 rad |

Physics stayed at 237-243 Hz in the full application while switching patterns,
control modes, joint enables and robot models.

## Phase 3: telemetry, charts and data logging

Install the one new dependency: `pip install -r requirements.txt` (adds
`pyqtgraph==0.12.4`, the last release for Python 3.7).

**Data pipeline.**

```
physics step (240 Hz, simulation thread)
  -> TelemetryRecorder.record()     one (n_joints x 14) row per step, deque ring buffer (60 s)
       |-- latest row      -> telemetry table   10 Hz  (Qt signal)
       |-- decimated 60 Hz -> charts            25 Hz  (Qt signal, chunks)
       '-- decimated CSV   -> CsvTelemetryWriter queue -> writer thread -> file
```

Per movable joint and step: timestamp, robot model, robot body ID, joint ID and
name, position, target, tracking error (target - actual), velocity,
acceleration ((v - v_prev) / dt), motor torque, reaction force and torque XYZ
from `enableJointForceTorqueSensor`, mechanical power (torque x velocity) and
reaction force magnitude. Nothing is emitted to Qt at 240 Hz.

**Charts tab.** Two stacked, time linked plots, each showing any of position,
velocity, acceleration, torque, tracking error, power or reaction force
magnitude. The window is 10, 30 or 60 s, each joint can be toggled, and the
plots can be frozen or cleared. Plots only redraw while the tab is visible.

**Logging.** Choose 50, 100 or 240 Hz and press Start Logging. Files go to
`data/logs/telemetry_YYYYmmdd_HHMMSS.csv` (the folder is set in `config.py`).
The file is opened once; rows are formatted and written by a background thread
and flushed every second, so disk speed cannot stall physics. Stop, Reset and
loading a robot end the current file, because simulated time restarts at zero.
Export CSV saves the last 60 s of full rate (240 Hz) telemetry to a file you
choose.

CSV columns (long format, one row per joint per sample):
`timestamp, robot_model, joint_id, joint_name, position, target_position,
tracking_error, velocity, acceleration, motor_torque, reaction_force_x,
reaction_force_y, reaction_force_z, reaction_torque_x, reaction_torque_y,
reaction_torque_z, power, fault_type, health_status`.
`timestamp` is simulated time in seconds. Until fault injection and the health
model exist, `fault_type` is `NORMAL` and `health_status` is `UNKNOWN`.

**Measured on the dev machine** (full application, Stress motion, charts tab
visible, 240 Hz logging, then torque control):

| Check | Result |
| --- | --- |
| Physics rate | mean 240.02 Hz, range 236.7 - 243.6 |
| CSV sample rate | 240.0 Hz and 50.0 Hz, largest gap one step |
| tracking_error = target - position | within 1e-5 (text rounding) |
| power = torque x velocity | within 6e-4 (text rounding) |
| Joint 1 vertical reaction force | about 200 N, consistent with the arm weight |

## Not in Phase 3

Fault injection, anomaly detection, datasets, RUL and maintenance are still
deliberately absent.
