# Industrial Robot Simulator - Phase 1

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
│   ├── robot.py                   URDF load, joint discovery, batched state reads
│   ├── joint_controller.py        limit clamping, batched motor commands, brakes
│   ├── trajectory.py              quintic segments, sine, point to point, manual
│   └── camera.py                  orbit camera with smoothed follow
│
├── rendering/
│   └── pybullet_renderer.py       offscreen frames, adaptive quality, frame mailbox
│
└── ui/
    ├── main_window.py             layout, thread ownership, signal wiring
    ├── simulator_view.py          viewport widget, mouse input
    └── controls_panel.py          controls, telemetry table, event log
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

## Not in Phase 1

Fault injection, CSV logging, anomaly detection, predictive maintenance,
charting and multi-robot support are deliberately absent. The extension points
are in place: the trajectory generator, controller and renderer are separate
objects, and the telemetry ring buffer already collects at 240 Hz with only a
decimated view reaching the UI.
