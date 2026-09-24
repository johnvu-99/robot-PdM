# robot-PdM

Predictive maintenance AI for factories' one-arm robots.

A KUKA iiwa simulator built for smooth, deterministic motion first and features
second: fixed 240 Hz physics on a dedicated thread, offscreen PyBullet
rendering, PyQt5 front end — grown phase by phase into a condition-monitoring
and remaining-useful-life (RUL) research platform.

Each published phase is kept as its own self-contained snapshot. Phases 1 to 6
(core simulator, fault injection, telemetry, baselines, anomaly detection, first
dataset adapters) are included in both folders below and are not published
separately.

## Repository layout

| Folder | Phase | What it adds |
| --- | --- | --- |
| [`RS7/`](RS7/robot_simulator_7) | 1 - 7 | Full simulator through phase 7: RUL pipeline refinement, leakage controls, cross-dataset validation |
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
