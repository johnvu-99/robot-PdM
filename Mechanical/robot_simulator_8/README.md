# Industrial Robot Simulator - Phase 7 + Partner Data

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
│   ├── fault_injector.py          10 fault types, degradation scenarios, labels
│   ├── step_pipeline.py           one physics step, shared by UI worker and generator
│   ├── dataset_generator.py       labelled dataset runs in a separate process
│   ├── joint_controller.py        position / velocity / torque control, limits, brakes
│   ├── trajectory.py              quintic segments, sine, point to point, manual
│   └── camera.py                  orbit camera with smoothed follow
│
├── monitoring/
│   ├── telemetry.py               240 Hz deque ring buffer, chart/CSV/monitor feeds
│   ├── feature_extractor.py       1/5/10 s rolling statistics and trends (numpy)
│   ├── baseline.py                per mode normal baseline (mean, std, percentiles)
│   ├── anomaly_detector.py        rule based detector, Isolation Forest per joint
│   ├── monitoring_service.py      feature/detection thread, training, model I/O
│   └── dataset_features.py        the same features from generated CSV runs
│
├── datasets/                      real machine data adapters (no Qt, no PyBullet)
│   ├── base_adapter.py            Recording, caching, discovery interface
│   ├── signal_features.py         vibration and torque features, fast CSV parsing
│   ├── mechanical_dataset.py      MechanicalDatasetAdapter (SEU gearbox, CWRU)
│   ├── run_to_failure.py          shared life_fraction / RUL_fraction logic
│   ├── xjtu_sy.py                 XJTUSYDatasetAdapter
│   ├── phm2012.py                 PHM2012DatasetAdapter
│   └── feature_schema.py          ConditionFeatures and the two feature domains
│
├── predictive/
│   ├── health_model.py            joint health score 0-100, configurable weights
│   ├── maintenance_engine.py      simulation-based maintenance recommendations
│   ├── event_log.py               edge triggered maintenance events, CSV export
│   ├── rul_model.py               causal features, Linear/RF/GB regressors, metrics
│   ├── fault_validation.py        fault classification and anomaly validation
│   └── dataset_jobs.py            load / preprocess / extract / train / evaluate jobs
│
├── storage/
│   ├── csv_logger.py              buffered CSV writer on its own thread
│   └── model_store.py             joblib model bundles in models/
│
├── rendering/
│   └── pybullet_renderer.py       offscreen frames, adaptive quality, frame mailbox
│
└── ui/
    ├── main_window.py             layout, thread ownership, signal wiring
    ├── simulator_view.py          viewport widget, mouse input
    ├── overview_panel.py          dashboard: robot health, joint bars, joint table
    ├── maintenance_panel.py       recommendations, score breakdown, weights editor
    ├── event_log_panel.py         Log tab
    ├── rul_panel.py               RUL tab (jobs run in a separate process)
    ├── anomaly_panel.py           Anomaly tab: training controls, per joint results
    ├── fault_panel.py             fault injection, degradation, dataset generator
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

## Phase 4: fault injection and degradation

**Every fault changes the simulated system.** Telemetry is never edited after
the fact. Severity is 0 to 1; the constants that give it physical meaning live
in `config.py` (`FAULT_*`).

| Fault | Mechanism |
| --- | --- |
| Motor weakness | motor force limit x (1 - severity); presets 100/80/60/40/20% available |
| Increased friction | extra viscous joint damping (`changeDynamics`); in torque mode subtracted from the motor torque |
| Joint stuck | joint limits narrowed around the seizure position (fully seized at 100%) |
| Backlash | hysteresis dead band between motor command and joint |
| Sensor noise | Gaussian noise on position, velocity and torque sensors |
| Sensor bias | constant encoder offset |
| Sensor drift | encoder offset growing linearly with time |
| Overload | unmodelled payload: downward external force on the link, every step |
| Torque degradation | force capacity falls linearly over the fault window (60 s if permanent) |
| Intermittent fault | random bursts of motor power loss |

Sensor faults feed back into control: all three control modes servo on the
measured signal, so a biased encoder really puts the arm in the wrong place.
In position mode the offset is applied to the servo target, because PyBullet's
position motor reads the true joint state internally.

**Ground truth labels.** Each joint on each sample has a `fault_type`
(`NORMAL`, one of the ten labels, or a `+` joined combination), a
`fault_severity`, a `ground_truth_health` (0 to 100) and a `health_status`
(HEALTHY >= 90, MINOR_DEGRADATION >= 70, WARNING >= 40, CRITICAL). The CSV keeps
the Phase 3 columns and appends `fault_severity`, `ground_truth_health` and
`scenario`. Health here is the scenario's ground truth, not an estimate; for a
single fault it is 100 x (1 - severity).

**Degradation scenarios.** Health falls from 100% to the chosen final health
along (t/T)^shape, and each mechanism's severity follows it:

| Scenario | Mechanisms (severity at end) |
| --- | --- |
| Gearbox wear | friction 0.9, backlash 0.7 |
| Motor ageing | torque degradation 0.6, sensor noise 0.3 |
| Encoder degradation | sensor drift 0.8, sensor noise 0.6 |
| Bearing wear | friction 0.7, sensor noise 0.4 |
| Payload creep | overload 0.9, friction 0.3 |

A mechanism appears in the label once its severity reaches 0.05.

**Dataset generator.** Choose Quick (13 runs) or Standard (74 runs) and press
Generate. It runs in a separate process at below normal priority, steps the
same `StepPipeline` as the live simulator as fast as possible, and writes
100 Hz CSVs to `data/simulation/normal/` and `data/simulation/faulty/`, plus a
`manifest_<time>.json` with every run's settings, seed and label counts.
Faulty runs start healthy, so each file holds NORMAL samples before the onset.
Quick finished in about 4 s on the dev machine (roughly 40x real time), and the
live physics rate did not change while it ran.

Faults and scenarios are scheduled on simulated time, so Stop, Reset and loading
a robot clear them.

**Measured effect** (joint 2, Stress motion, severity 0.7, identical normal and
faulty runs compared over the same window):

| Fault | Position control | Torque control |
| --- | --- | --- |
| Joint stuck | tracking error x172, torque +390% | error x175, torque +288% |
| Backlash | error x4.8, torque +11% | error x5.4 |
| Friction | torque +13% | error x33 |
| Overload | torque +7% | error x4.4, torque +9% |
| Sensor noise | error x1.9, measured vs true up to 0.022 rad | error x1.9 |
| Sensor bias | measured vs true 0.070 rad | error x4.1 |
| Sensor drift | measured vs true 0.056 rad after 7 s | error x1.8 |

**Important finding about capacity faults.** The KUKA URDF gives every joint
300 N m, but Stress motion needs at most 82 N m on joint 2 and under 5 N m on
the wrist. Motor weakness, torque degradation and intermittent faults therefore
have no physical effect until available torque drops below demand: motor
weakness at 0.7 changed nothing, at 0.9 tracking error rose 300x. That is
physically correct (a motor with margin hides its weakness), but it means a
label can describe a fault that leaves no trace in the signals. The dataset
generator uses severities 0.80/0.90/0.95 and loaded joints 2 and 4 for these
faults for that reason (`DATASET_CAPACITY_FAULT_SEVERITIES` in `config.py`).

## Phase 5: feature extraction and anomaly detection

New dependencies: `scikit-learn==1.0.2`, `scipy==1.7.3`, `joblib==1.3.2` (the
last releases for Python 3.7). Run `pip install -r requirements.txt` again.

**Threads.** Physics hands every 240 Hz sample to a queue (no ML code in the
physics module). A dedicated monitoring thread keeps a 11 s ring buffer,
computes features every 0.5 s of simulated time, runs both detectors, trains
models and saves/loads them. The Anomaly tab polls its result at 5 Hz.

**Features.** For torque, velocity, acceleration, tracking error, power and
reaction force magnitude, over 1, 5 and 10 s windows: mean, std, variance, min,
max, median, RMS, peak to peak, 95th percentile, plus least squares trends of
torque, tracking error and power. Engineered features use the 5 s window:
`torque_RMS`, `tracking_error_RMS`, `acceleration_RMS`, `power_RMS`,
`force_RMS`, `torque_trend`, `tracking_error_trend`, `power_trend`.

**Baseline.** Press Train Baseline while the robot runs normal motion. Only
clean windows are used: no injected fault anywhere in the 10 s window (checked
against ground truth labels), robot running, and at least 3 s since the last
motion mode change. Mean, std and 5/50/95th percentiles are stored per motion
mode (a Stress sweep and a held pose are very different) with a pooled ALL
fallback. Run every pattern you want monitored during collection.
Baseline from Dataset does the same from `data/simulation/normal` (generate the
STANDARD preset; 12 s QUICK runs are too short for 10 s windows).

**Rule detector.** WARNING when a feature exceeds baseline mean + 2 std for 5
consecutive windows, CRITICAL above mean + 3 std for 10 consecutive windows, on
the five engineered RMS features. Std has floors (5% of the mean, and a small
absolute value per signal) so a near constant baseline cannot turn numerical
noise into alarms.

**Isolation Forest.** Train Isolation Forest uses the baseline's normal
windows: one forest per joint on 20 features, StandardScaler fitted on that
normal data only. The anomaly score is 0.5 at the learned threshold and rises
toward 1. WARNING/CRITICAL use the same 5/10 window persistence. Save Model and
Load Model write joblib bundles to `models/`; a bundle for a different robot is
refused.

Per joint the tab shows anomaly score, both detector states, the combined
status, the most deviating feature with its z score, and the injected ground
truth, plus running TP/FP/FN/TN counts against that ground truth.

**Measured** (baseline 90 s Sine, joint 2 faults, position control):

| Check | Result |
| --- | --- |
| Physics with monitoring running | 240.00 Hz mean (236 - 245) |
| Clean motion after training | 0 alarms in 840 joint evaluations |
| Joint stuck 0.6 | WARNING 5.5 s, CRITICAL 8.5 s |
| Backlash 0.6 | WARNING 2.5 s, CRITICAL 5.5 s |
| Sensor noise 0.6 | WARNING 2.5 s, CRITICAL 5.5 s |
| Overload 0.6 | WARNING 3.5 s, CRITICAL 6.0 s |
| Intermittent 0.9 | WARNING 4.0 s, CRITICAL 6.5 s |
| Motor weakness 0.9 | WARNING 9.0 s, CRITICAL 12.5 s |
| Friction 0.6, sensor bias 0.6 | brief alarm at onset, then NORMAL |
| Torque degradation 0.9 (60 s ramp) | not detected within 25 s |

2.5 s is the fastest possible WARNING: 5 windows x 0.5 s.

**Limits worth knowing.**
- A biased encoder is invisible in position control: the servo closes the loop
  on the biased reading, so measured tracking error stays normal while the arm
  sits in the wrong place. Detecting it needs an independent reference.
- Friction at moderate severity is absorbed by the position servo (about +13%
  torque, under 2 std of a Sine baseline).
- A fault on one joint disturbs the others through the arm's dynamics, so
  healthy joints can alarm too (overload on joint 4 raised joint 2). The rule
  detector points at the right joint by z score, but per joint isolation is not
  guaranteed; the ground truth columns make this visible.
- The baseline is specific to motion mode and roughly to speed. New motion
  outside what was collected will look anomalous.

## Phase 6: real mechanical datasets and RUL

**These datasets are not robot telemetry.** They come from gearbox and rolling
bearing test rigs. They are used to develop and validate condition monitoring
features, anomaly detection and the RUL method. Simulation telemetry and real
machine data stay separate raw domains; a production KUKA model would need real
KUKA sensor data.

### Getting the data

```
cd data\external
git clone https://github.com/cathysiyu/Mechanical-datasets
git clone https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset
```

XJTU-SY is not stored in its GitHub repository; the README there links to the
download (Google Drive, Dropbox, MEGA and others). Extract it so the bearing
folders are below `data\external\XJTU-SY_Bearing_Datasets`, for example
`...\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_1\1.csv`. Any extra wrapping
folder is fine. All three paths can be changed in `config.py` or in the RUL tab.

| Dataset | Role | What the adapter reads |
| --- | --- | --- |
| Mechanical-datasets | Fault detection and anomaly validation | `gearbox/gearset` and `gearbox/bearingset` CSVs (SEU drivetrain simulator: motor vibration, planetary gearbox x/y/z, motor torque, parallel gearbox x/y/z; tab or comma separated, 16 header lines, 5120 Hz from the header's 2000 Hz frequency limit), plus the CWRU `.mat` bearing files (fan end) |
| XJTU-SY | Degradation learning and RUL training | 15 run-to-failure bearings, one 32768 sample horizontal/vertical snapshot per minute |
| PHM 2012 / PRONOSTIA | Independent RUL validation | `Learning_set` (6) and `Full_Test_Set` (11) complete runs, 2560 samples every 10 s; comma or semicolon separated; `temp_*` files ignored. `Test_set` is truncated and not used, because its true RUL is not in the repository |

### Pipeline

**Features.** Vibration: RMS, std, peak to peak, crest factor, kurtosis,
skewness, energy, dominant frequency, spectral centroid, spectral energy, four
band energies. Torque (SEU): mean, RMS, std, peak, trend. Results are cached in
`data/processed/` keyed on file size and time, so a second run is instant.

**Unified format.** `datasets/feature_schema.py` keeps SIMULATION features
(torque_RMS, tracking_error_RMS, ...) and REAL_MACHINE features
(vibration_RMS, kurtosis, ...) under their own names, and `ConditionFeatures`
maps them into LOAD, MOTION_ERROR, VIBRATION, ENERGY, DEGRADATION and ANOMALY.
Missing groups stay missing (NaN plus a mask); nothing is imputed.

**RUL labels.** `life_fraction = t / t_failure`, `RUL_fraction = 1 - life_fraction`,
with t from the file number and snapshot interval.

**RUL features.** Dimensionless and causal: each run is normalised by its own
first 10 snapshots (log RMS ratio, kurtosis ratio, band energy change, ...),
then fast and slow EMAs and a running maximum. The health indicator shown in the
plot is `exp(-max(0, running max of the slow EMA of log RMS ratio)))`.

**Models.** Linear regression, random forest and gradient boosting, each in an
sklearn Pipeline with a StandardScaler, saved with joblib to `models/`.

**Leakage controls.**
- Splits are by complete bearing run, never by snapshot.
- Leave one run out cross validation on the training dataset.
- Train and validation run IDs are asserted disjoint; evaluating a model on a
  run it was trained on raises an error.
- The scaler is inside the pipeline, so it is fitted on training runs only.
- Mechanical-datasets segments are split into a contiguous training block and
  a later test block per recording, with a 5 segment gap.

**Metrics.** MAE, RMSE and R2 on RUL fraction (MAE on a fraction is already
error as a percent of total life). A predicted fraction f is converted to time
with `elapsed x f / (1 - f)`, which is unstable early in life, so time errors
are only reported from 50% of life onwards.

**Protocols.** Default: train on XJTU-SY, validate on PHM 2012. If XJTU-SY is not
downloaded, "PHM Learning_set -> Full_Test_Set" runs the same pipeline on PHM
alone (it is labelled as such everywhere).

### RUL tab

Load Dataset, Preprocess Dataset (checks every recording is readable and writes
`data/processed/<dataset>/index.json`), Extract Features, Fault Validation,
Train RUL Model, Evaluate RUL Model, Load Saved Model, Cancel. The per run table
shows life, MAE, RMSE, R2 and late life time error; selecting a run plots true
RUL, predicted RUL and the health indicator, and the slider reads out True RUL,
Predicted RUL, Error and Health Indicator at any point in life.

All jobs run in a separate process. Physics stayed at 239.8 Hz while they ran.

### Measured on the dev machine

Tested on a partial download: the first 6 MB of 12 SEU files, and every third
snapshot of 7 PHM bearings (3 Learning_set, 4 Full_Test_Set).

Fault validation, SEU drivetrain (later time blocks, unseen in training):

| Check | Result |
| --- | --- |
| 9 class fault classification | accuracy 0.93, macro F1 0.91 |
| Anomaly detection, per rig and condition | every fault label detected in 100% of test segments, ROC AUC 0.99 - 1.00 |
| False alarms on healthy test segments | 0 - 9% |

Anomaly detection must be trained per rig and operating condition: a single
healthy model pooled over the gearbox at 20-0, the gearbox at 30-2 and the
bearing rig detected nothing, because "healthy" became broader than the faults.

RUL, PHM Learning_set (3 runs) -> Full_Test_Set (4 runs):

| Model | Validation MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Linear regression | 0.136 | 0.162 | 0.66 |
| Random forest | 0.210 | 0.260 | 0.15 |
| Gradient boosting | 0.230 | 0.278 | -0.01 |

Per run, the linear model ranged from R2 0.88 (Bearing2_6) to 0.28
(Bearing2_7, a 38 minute life). With three training bearings the flexible
models overfit, which is why the interpretable baseline wins here. These are
small sample numbers, not a benchmark claim.

**Not verified on real XJTU-SY files.** The data is hosted outside GitHub, so
the adapter was written to the documented layout and tested on a synthetic
folder with that structure (header row, 32768 x 2 samples, 3 conditions). It
parsed correctly (about 10 ms per snapshot) and the XJTU -> PHM protocol ran end
to end, but no XJTU-SY performance figures exist yet. Please check the first
Load Dataset report when you have the real files.

## Phase 7: health score, predictive maintenance and dashboard

Tabs: **Overview | Telemetry | Charts | Anomaly | RUL | Maintenance | Log | Joints**,
with a Robot health chip in the top bar. Health, recommendations and events are
computed in the monitoring thread; the physics module is unchanged.

### Joint health score

Evidence per joint, each turned into a penalty from 0 to 1:

| Evidence | Source | Penalty ramp |
| --- | --- | --- |
| tracking error, torque, force, power, acceleration | baseline z score of the 5 s RMS feature | z 2 -> 10 |
| isolation forest | anomaly score, counted only after its 5 window persistence | 0.5 -> 0.95 |
| degradation trend | largest z of torque, tracking error and power trends | z 3 -> 12 |
| RUL | health trend extrapolation (below) | RUL 600 s -> 0 |

```
health = 100 x product(1 - weight_i x penalty_i)   over available evidence
```

Each weight is the largest share of health that item can remove on its own, and
all are editable on the Maintenance tab (Apply and save writes
`models/health_weights.json`). A product is used instead of a weighted average
so a single severe fault cannot be diluted by healthy signals. Evidence that is
unavailable (no baseline, no forest) is left out rather than counted as healthy,
and a joint with no evidence shows "no data", never 100%. The score is smoothed
with a 2 s time constant, and the Maintenance tab lists how much each item
removed. Bands: 90-100 HEALTHY, 70-90 MINOR_DEGRADATION, 40-70 WARNING,
0-40 CRITICAL. Robot health is the mean of joint scores
(`ROBOT_HEALTH_AGGREGATION` also offers MIN and MEAN_MIN).

**RUL for simulated joints.** The Phase 6 model predicts bearing life from
vibration, and PyBullet joints have no vibration sensor, so it is not applied
to them. A joint's Estimated RUL is a linear extrapolation of its last 60 s of
health to the CRITICAL threshold, labelled as such. It only affects the score
while the joint is still above CRITICAL, so the same drop is not counted twice.

### Maintenance engine

SIMULATION-BASED MAINTENANCE RECOMMENDATIONS, shown on the Overview and
Maintenance tabs, with the reasoning for each joint:

| Situation | Recommendation |
| --- | --- |
| HEALTHY | No action required |
| MINOR_DEGRADATION | Monitor joint |
| WARNING, acceleration noise dominant | Inspect encoder |
| WARNING, tracking fails while torque is below normal | Inspect motor load |
| WARNING, force or power up, tracking normal | Inspect motor load |
| WARNING, tracking error and torque up | Inspect gearbox |
| WARNING, torque rising over time, tracking normal | Check lubrication |
| WARNING, tracking error without extra torque | Inspect gearbox |
| WARNING, no dominant signal | Inspect joint |
| CRITICAL | Schedule maintenance (+ probable cause) |
| health < 15% and tracking error with torque or force far above baseline | Stop machine immediately |

These are rules written for this simulator's fault models, not certified
industrial diagnoses.

### Event log

Events are edge triggered: anomaly detected, a feature deviation of 25% or more
above baseline (it must also be statistically significant and hold for 4
evaluations), degradation trend detected, severity changes and new
recommendations. Columns: timestamp, simulated time, robot, joint, event,
severity, details. Export CSV writes the log to `data/logs`.

Example from a joint stuck fault on joint 2 (dev machine):

```
15:05:35  t=93.7  Joint 2 WARNING                                   health 47%
15:05:35  t=93.7  Maintenance recommendation: inspect gearbox       tracking error and torque up together
15:05:36  t=94.2  Joint 2 CRITICAL                                  health 31%
15:05:36  t=94.8  Joint 2 torque RMS +119%
15:05:37  t=95.3  Joint 2 tracking error RMS +1315%
15:05:40  t=99.0  Joint 2 degradation trend detected
15:05:41  t=100.1 Maintenance recommendation: stop machine immediately
```

### Measured on the dev machine

Baseline 90 s of Sine motion, faults on joint 2, position control:

| Fault | Joint 2 health | Recommendation |
| --- | --- | --- |
| none | 100% on all joints | No action required |
| Joint stuck 0.6 | 14% CRITICAL | Stop machine immediately + inspect gearbox |
| Backlash 0.6 | 29% CRITICAL | Schedule maintenance + inspect gearbox |
| Sensor noise 0.6 | 3% CRITICAL | Stop machine immediately + inspect encoder |
| Overload 0.9 | 65% WARNING | Inspect motor load |
| Motor weakness 0.9 | 29% CRITICAL | Schedule maintenance + inspect gearbox |
| Friction 0.9, sensor drift 0.8 | 100% | No action required |

Physics stayed at 240.00 Hz with all dashboards running. With a 60 s baseline,
clean motion produced no events at all before the fault.

### Limitations

- Motor weakness is recommended as a gearbox inspection: joint 2's torque RMS
  stays close to normal when capped, so these features cannot tell it apart
  from backlash.
- Friction and encoder drift are invisible in position control (see Phase 5),
  so health stays 100% even though a fault exists. The Telemetry tab's ground
  truth column shows this honestly.
- A severe fault on one joint disturbs the others (sensor noise on joint 2
  pulled other joints down to 15%), because the arm is dynamically coupled.
- A short baseline makes the Isolation Forest unreliable. Collect at least 60 s
  of each motion pattern you will run.
- All thresholds and weights are assumptions for this simulator, exposed in
  `config.py`, not calibrated values.

## Extension: Mechanical-datasets anomaly detection and partner data

Command line tools, run from the project root. Nothing in the simulator, the
physics loop or Phases 1 to 7 changed.

```
python -m tools.mechanical_anomaly train    --root data/external/Mechanical-datasets
python -m tools.mechanical_anomaly score    path/to/health_20_0.csv path/to/IR007_0.mat

python -m tools.partner_data template --out data/external/partner_example --mode fault_labels
python -m tools.partner_data validate --root data/external/partner
python -m tools.partner_data train    --root data/external/partner
python -m tools.partner_data score    --root data/external/new_batch
```

`PARTNER_DATA_SPEC.md` describes how the partner should record and deliver data.

Architecture:

```
Mechanical-datasets (SEU CSV, CWRU .mat)        partner folder (dataset.json + recordings.csv + CSV/xlsx/mat)
        |                                                   |
datasets/mechanical_dataset.py                  datasets/partner_schema.py   read and check the description
  load, CWRU 48 kHz -> 12 kHz, segment          datasets/partner_dataset.py  load, split joints, validate, window
datasets/signal_features.py                     datasets/partner_features.py features chosen by channel type
        |                                                   |
        +---------------------+-----------------------------+
                              |
                predictive/anomaly_engine.py   one model per setup (machine/condition/joint):
                                               RobustScaler + IsolationForest, threshold from held
                                               out healthy data, robust z explanations
                              |
        tools/mechanical_anomaly.py            predictive/partner_models.py (+ classification, RUL)
                                               tools/partner_data.py
```

Findings from testing:
- CWRU healthy files (normal_*.mat) are 48 kHz while fault files are 12 kHz
  (checked against the shaft frequency). Reading both as 12 kHz inflated the
  separation and gave 25% false alarms. Healthy files are now decimated to
  12 kHz; false alarms dropped to 0% and the smallest ball fault became the
  hard case it is known to be (58% of windows detected, AUC 0.98).
- One healthy model per setup is required. Pooling different rigs, loads or
  robot joints makes "healthy" broader than the faults.
- The threshold must be calibrated on held out healthy data from every
  recording, not only the last one.
- Windows must span a full motion cycle for robot telemetry.

## Temperature (motor thermal model and temperature channels)

**Simulator.** `simulator/thermal_model.py` adds a first order thermal model per
joint, stepped with the physics but outside the solver:

```
P_loss = copper * torque^2 + friction * |speed| + idle      [W]
dT/dt  = (P_loss * R_thermal - (T - ambient)) / tau_thermal  [C/s]
```

Temperature therefore follows the real losses: a joint that works harder settles
hotter, friction and overload faults heat it slowly afterwards, and the new
**COOLING_FAILURE** fault (and the COOLING_DEGRADATION scenario) raises the
steady state rise up to 3.5x for the same load. Temperature appears as a
telemetry channel, a `temperature` column in the CSV, a chart metric, a column
in the telemetry table, and as evidence in the health score.

Measured (stress motion, 7 minute time constant): joints settle near 26 C
holding a pose and 34 C sweeping; a cooling failure on joint 2 took it to 42 C,
a 17 C rise against 5.6 C healthy, while the other joints did not move.

**How temperature is judged.** Not by a baseline z score: a motor warms up for
minutes after starting, and that warm-up is normal. Instead:

* absolute rise over ambient (warning at 25 C, critical at 45 C), and
* rise relative to that joint's own normal rise (warning at 1.6x, critical 3x),

whichever is worse. A small wrist motor never reaches 25 C rise even with a
failed fan, which is why the ratio test matters. If the baseline was collected
while the motors were still warming, the Anomaly tab says so and asks for a new
baseline. Maintenance rules: hot with raised torque -> check lubrication; hot
with extra load -> inspect motor load; hot alone -> inspect joint, check the
cooling path.

**Partner data.** Channel types `temperature` and `ambient_temperature` are
supported. Their features (level, maximum, C per minute, rise since start, rise
over ambient, rise per watt) use a longer window set by
`temperature_window_seconds`, and `warmup_skip_seconds` drops the warm-up at the
start of every recording. The validator warns when temperature is still climbing
after that, or when no ambient channel exists.

A cooling fault moves a few slow features moderately, which an IsolationForest
over a hundred vibration features averages away: on the synthetic example it
detected only 6.6% of cooling windows. The engine therefore also compares
thermal features directly against their healthy median and flags
THERMAL_RATIO_WARN, which raised cooling detection to 75% with no change in
false alarms.

## RUL from the command line

`tools/rul.py` runs the same RUL code as the app's RUL tab, without opening the
simulator:

```
python -m tools.rul info                                   # what the dataset folders contain
python -m tools.rul train    --protocol XJTU_TO_PHM --model RANDOM_FOREST
python -m tools.rul evaluate --model-file models/rul_....joblib --curves curves.csv
python -m tools.rul compare  --protocol XJTU_TO_PHM         # all three models, one table
python -m tools.rul predict  --model-file models/rul_....joblib data/external/.../Bearing1_4
```

Without XJTU-SY, use `--protocol PHM_LEARNING_TO_FULL_TEST`.

### Measured on the real XJTU-SY download

All 15 bearings (9216 snapshot files, 11 GB), features extracted in 2 min 19 s
and cached in `data/processed/` (1.7 MB). Trained on XJTU-SY with leave one
bearing out cross validation, validated on PHM 2012 bearings the model never
saw (a different lab, machine and load).

| Model | XJTU cross validation MAE / R2 | PHM validation MAE / R2 / rank |
| --- | --- | --- |
| Linear regression | 0.212 / 0.12 | 0.266 / -0.30 / - |
| **Random forest** | 0.209 / 0.19 | **0.173 / 0.47 / 0.77** |
| Gradient boosting | 0.193 / 0.32 | 0.218 / 0.10 / 0.74 |

Random forest is the default (`RUL_DEFAULT_MODEL`). Per bearing it reaches R2
0.20 to 0.72 on PHM 2012, with rank correlation 0.62 to 0.95.

Two things were changed after measuring on the real data:

* **Time based smoothing** (`RUL_TIME_AWARE_SMOOTHING`). XJTU-SY records one
  snapshot per minute and PHM 2012 one every 10 s, so smoothing counted in
  snapshots covered 6x more real time in training than in validation. The EMAs
  now use time constants in seconds and the healthy baseline is the first
  `RUL_BASELINE_SECONDS`. Linear regression validation improved from MAE 0.314,
  R2 -0.69 to MAE 0.266, R2 -0.30.
* **Rank correlation reported** next to MAE and R2. Across datasets the
  predicted fraction is biased (XJTU bearings live up to 42 h, PHM bearings
  under 2.5 h, so PHM bearings look closer to failure than they are) while the
  ordering stays right, and an alarm threshold needs the ordering.
