# DATASETS: authoritative sources

None of the raw third-party data is redistributed in this repository. Every
dataset below is licensed and published by its originating institution, and
`.gitignore` excludes `data/external/` so the raw files stay where their owners
put them. This file is the authority link for each one: where the data really
comes from, who published it, what to cite, and where the loader expects it.

Check each source's own terms before use. Several of these are free for
academic and research use but are **not** open-licensed for redistribution,
which is why they are linked rather than vendored.

| Dataset | Authority | Role in this project | Adapter |
| --- | --- | --- | --- |
| XJTU-SY Bearing Datasets | [WangBiaoXJTU/xjtu-sy-bearing-datasets](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets) | Degradation learning and RUL training | [`xjtu_sy.py`](Mechanical/robot_simulator_8/datasets/xjtu_sy.py) |
| IEEE PHM 2012 / PRONOSTIA | [FEMTO-ST Institute](https://www.femto-st.fr/) · mirror: [Lucky-Loek/ieee-phm-2012-data-challenge-dataset](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset) | Independent RUL benchmark and validation | [`phm2012.py`](Mechanical/robot_simulator_8/datasets/phm2012.py) |
| SEU Drivetrain Dynamics Simulator ("Mechanical-datasets") | [cathysiyu/Mechanical-datasets](https://github.com/cathysiyu/Mechanical-datasets) | Fault detection and anomaly validation | [`mechanical_dataset.py`](Mechanical/robot_simulator_8/datasets/mechanical_dataset.py) |
| CWRU Bearing Data | [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | Fault detection (fan-end accelerometer `.mat` files) | [`mechanical_dataset.py`](Mechanical/robot_simulator_8/datasets/mechanical_dataset.py) |

**These datasets are not robot telemetry.** They come from gearbox and rolling
bearing test rigs, and are used to develop and validate the condition
monitoring features, anomaly detection and the RUL method. A production KUKA
model would need real KUKA sensor data.

---

## XJTU-SY Bearing Datasets

- **Authority:** https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets
- **Publisher:** Institute of Design Science and Basic Components, Xi'an
  Jiaotong University (XJTU), with Changxing Sumyoung Technology Co., Ltd. (SY).
- **Contents:** 15 rolling-element bearings run to failure under three
  operating conditions (35Hz/12kN, 37.5Hz/11kN, 40Hz/10kN). One 1.28 s
  snapshot per minute at 25.6 kHz, two channels (horizontal, vertical).
- **Note:** the data is not stored in the GitHub repository itself — the README
  there links to the download mirrors (Google Drive, Dropbox, MEGA, etc.).
- **Cite:** B. Wang, Y. Lei, N. Li and N. Li, "A Hybrid Prognostics Approach
  for Estimating Remaining Useful Life of Rolling Element Bearings," *IEEE
  Transactions on Reliability*, vol. 69, no. 1, pp. 401–412, 2020.
- **Expected layout** (any extra wrapping folder is fine; folders are found
  recursively):

  ```
  data/external/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/1.csv
  data/external/XJTU-SY_Bearing_Datasets/37.5Hz11kN/Bearing2_1/...
  data/external/XJTU-SY_Bearing_Datasets/40Hz10kN/Bearing3_1/...
  ```

## IEEE PHM 2012 Data Challenge / PRONOSTIA

- **Authority:** the PRONOSTIA platform and the challenge data are from the
  FEMTO-ST Institute (Besançon, France) — https://www.femto-st.fr/
- **Working mirror used here:**
  https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset
- **Contents:** accelerated bearing run-to-failure tests under three
  conditions (1800rpm/4000N, 1650rpm/4200N, 1500rpm/5000N). Each `acc_*.csv`
  holds 2560 rows of hour, minute, second, microsecond, horizontal and
  vertical acceleration, recorded every 10 s.
- **Cite:** P. Nectoux et al., "PRONOSTIA: An experimental platform for
  bearings accelerated degradation tests," *IEEE International Conference on
  Prognostics and Health Management (PHM'12)*, Denver, 2012.
- **Usage in this project:** `Learning_set` (6 runs) and `Full_Test_Set`
  (11 runs) only. `Test_set` is truncated and deliberately unused, because its
  true RUL is not published with the data. `temp_*.csv` files are ignored.
- **Expected layout:**

  ```
  data/external/ieee-phm-2012-data-challenge-dataset/Learning_set/Bearing1_1/acc_00001.csv
  data/external/ieee-phm-2012-data-challenge-dataset/Full_Test_Set/Bearing1_3/acc_00001.csv
  ```

## SEU Drivetrain Dynamics Simulator ("Mechanical-datasets")

- **Authority:** https://github.com/cathysiyu/Mechanical-datasets
- **Publisher:** Southeast University (SEU), released alongside the transfer
  learning paper below.
- **Contents:** `gearbox/gearset` and `gearbox/bearingset` CSVs. 16 header
  lines, then 8 channels: motor vibration, planetary gearbox vibration x/y/z,
  motor torque, parallel gearbox vibration x/y/z. Tab or comma separated,
  5120 Hz. Conditions `20_0` and `30_2` (speed–load setting).
- **Cite:** S. Shao, S. McAleer, R. Yan and P. Baldi, "Highly Accurate Machine
  Fault Diagnosis Using Deep Transfer Learning," *IEEE Transactions on
  Industrial Informatics*, vol. 15, no. 4, pp. 2446–2455, 2019.
- **Expected layout:**

  ```
  data/external/Mechanical-datasets/gearbox/gearset/*.csv
  data/external/Mechanical-datasets/gearbox/bearingset/*.csv
  ```

## CWRU Bearing Data

- **Authority:** https://engineering.case.edu/bearingdatacenter
- **Publisher:** Case Western Reserve University Bearing Data Center.
- **Contents:** `.mat` bearing files; this project reads the **fan-end**
  accelerometer channel.
- **Terms:** CWRU asks that the Bearing Data Center be acknowledged as the
  source in any publication that uses the data.
- **Expected layout:** distributed inside the Mechanical-datasets clone as
  `data/external/Mechanical-datasets/dataset/*.mat`.

---

## Partner (robot) data

Partner-supplied robot telemetry is **not** third-party public data and is not
committed here. `data/external/partner/` is gitignored.

The recording and hand-over format is specified in
[`PARTNER_DATA_SPEC.md`](Mechanical/robot_simulator_8/PARTNER_DATA_SPEC.md).
A complete synthetic example of the exact file layout can be generated with:

```
python -m tools.partner_data template --out data/external/partner_example --mode fault_labels
```

---

## Getting the data

From `Mechanical/robot_simulator_8/`:

```
mkdir -p data/external
cd data/external
git clone https://github.com/cathysiyu/Mechanical-datasets
git clone https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset
# XJTU-SY: download via the links in its README, extract to
# XJTU-SY_Bearing_Datasets/ here
```

All three root paths are configurable in
[`config.py`](Mechanical/robot_simulator_8/config.py) or in the RUL tab of
the UI.

## What *is* committed

- `data/processed/` — cached extracted features (`.npz`), derived from the raw
  data above. Small, reproducible, keyed on source file size and mtime.
- `models/` — trained RUL model artifacts (`.joblib`).
- `data/simulation/` — telemetry produced by this simulator itself.
