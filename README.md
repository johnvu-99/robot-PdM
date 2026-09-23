# robot-PdM

Predictive maintenance AI for factories' one-arm robots.

A KUKA iiwa simulator built for smooth, deterministic motion first and features
second: fixed 240 Hz physics on a dedicated thread, offscreen PyBullet
rendering, PyQt5 front end — grown phase by phase into a condition-monitoring
and remaining-useful-life (RUL) research platform.

Each phase is kept as its own self-contained snapshot rather than being
squashed into one history, so the progression stays inspectable.

## Repository layout

| Folder | Phase | What it adds |
| --- | --- | --- |
| [`RS1/`](RS1/robot_simulator) | 1 | Core simulator: physics thread, URDF loading, joint control, trajectories, camera, UI |
| [`RS2/`](RS2/robot_simulator_2) | 2 | Fault injection and trajectory work on the core simulator |
| [`RS3/`](RS3/robot_simulator_3) | 3 | Telemetry ring buffer, feature extraction, CSV logging (`monitoring/`, `storage/`) |
| [`RS4/`](RS4/robot_simulator_4) | 4 | Baselines and labelled dataset generation |
| [`RS5/`](RS5/robot_simulator_5) | 5 | Anomaly detection: rule-based detector and Isolation Forest (`models/`) |
| [`RS6/`](RS6/robot_simulator_6) | 6 | Real mechanical dataset adapters and RUL (`datasets/`, `predictive/`) |
| [`RS7/`](RS7/robot_simulator_7) | 7 | RUL pipeline refinement: leakage controls, cross-dataset validation |
| [`Mechanical/`](Mechanical/robot_simulator_8) | 7 + partner data | Current line of work: partner dataset ingestion, `tools/`, trained RUL models |

Start with [`Mechanical/robot_simulator_8/README.md`](Mechanical/robot_simulator_8/README.md)
— it is the most complete, and covers the full architecture, the fault
injection model, the monitoring pipeline and the RUL results.

## Datasets

Raw third-party data is **not** redistributed here. See
**[DATASETS.md](DATASETS.md)** for the authoritative source of every dataset
used (XJTU-SY, IEEE PHM 2012 / PRONOSTIA, SEU Drivetrain, CWRU), what to cite,
and where each loader expects the files. `data/external/` is gitignored.

Committed derived artifacts: cached features in `data/processed/`, trained
models in `models/`, and simulator-generated telemetry in `data/simulation/`.

## Running

Each phase folder is independently runnable:

```
cd Mechanical/robot_simulator_8
pip install -r requirements.txt
python main.py
```

Key docs in the current phase:

- [`README.md`](Mechanical/robot_simulator_8/README.md) — architecture and results
- [`CODE_GUIDE_MECHANICAL.md`](Mechanical/robot_simulator_8/CODE_GUIDE_MECHANICAL.md) — code walkthrough
- [`PARTNER_DATA_SPEC.md`](Mechanical/robot_simulator_8/PARTNER_DATA_SPEC.md) — data hand-over format
