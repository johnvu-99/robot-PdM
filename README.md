# robot-PdM

Predictive maintenance AI for factories' one-arm robots.

A KUKA iiwa simulator built for smooth, deterministic motion first and features
second: fixed 240 Hz physics on a dedicated thread, offscreen PyBullet
rendering, PyQt5 front end — grown phase by phase into a condition-monitoring
and remaining-useful-life (RUL) research platform.

[![tests](https://github.com/johnvu-99/robot-PdM/actions/workflows/tests.yml/badge.svg)](https://github.com/johnvu-99/robot-PdM/actions/workflows/tests.yml)

## What it does

| Part | What it is | Result |
| --- | --- | --- |
| Simulator | KUKA iiwa with fault injection, a motor thermal model, live telemetry, per-joint health scores and maintenance advice | physics holds 240 Hz with all dashboards running |
| Fault detection | anomaly detection trained on healthy data only, plus fault classification, on the SEU gearbox and CWRU bearing test rigs | 99.6% accuracy (9 classes, SEU), 98.0% (4 classes, CWRU) |
| Remaining useful life (RUL) | predicts the share of life left from vibration, on 32 run-to-failure bearings from two labs (XJTU-SY, PHM 2012) | trained on one lab, tested on the other: error 17.5% of life (guessing: 25%), order of degradation right (rank 0.84), and every prediction comes with a 90% range that held on 94% of unseen snapshots |

These are test rig datasets, not robot telemetry. The limits are documented as
carefully as the results: which bearings the model still gets wrong and why,
which ideas were tried and rejected, and two weaknesses that the tests exposed
and that were then fixed.

## Repository layout

The project is [`Mechanical/robot_simulator_8/`](Mechanical/robot_simulator_8).
It was built phase by phase (core simulator → fault injection → telemetry →
baselines → anomaly detection → real datasets and RUL → health and maintenance
→ partner data); all phases are contained in it. The phase 7 snapshot is kept
as the git tag [`phase-7`](https://github.com/johnvu-99/robot-PdM/tree/phase-7).

| Where | What |
| --- | --- |
| [`README.md`](Mechanical/robot_simulator_8/README.md) | architecture, install, how to run, all results |
| [`CODE_GUIDE_MECHANICAL.md`](Mechanical/robot_simulator_8/CODE_GUIDE_MECHANICAL.md) | code walkthrough; sections 9 to 11 are the full story of the experiments, including the ones that failed |
| [`results/baseline_2026-09-28/`](Mechanical/robot_simulator_8/results/baseline_2026-09-28) | frozen baseline of every model and the experiment log |
| [`tests/`](Mechanical/robot_simulator_8/tests) | about 70 tests, run on every push |
| [`PARTNER_DATA_SPEC.md`](Mechanical/robot_simulator_8/PARTNER_DATA_SPEC.md) | how a partner should record and hand over robot data |

## Datasets

Raw third-party data is **not** redistributed here. See
**[DATASETS.md](DATASETS.md)** for the authoritative source of every dataset
used (XJTU-SY, IEEE PHM 2012 / PRONOSTIA, SEU Drivetrain, CWRU), what to cite,
and where each loader expects the files. `data/external/` is gitignored.

Committed derived artifacts: cached features in `data/processed/`, trained
models in `models/`, and simulator-generated telemetry in `data/simulation/`.

## Running

```
cd Mechanical/robot_simulator_8
pip install -r requirements-py310.txt      # Python 3.10; see the README there for Windows / Python 3.7
python main.py                              # the simulator and dashboards
python -m tools.rul compare                 # train and compare the RUL models (needs the datasets)
python -m pytest tests -q                   # the tests (no datasets needed)
```
