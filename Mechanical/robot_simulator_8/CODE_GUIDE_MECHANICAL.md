# Code guide: the mechanical data side

This guide explains every file, class and function used to load machine
datasets, compute features, train models and evaluate them. It covers:

```
datasets/     read dataset files and turn raw signals into feature numbers
predictive/   the machine learning: anomaly detection, fault classification, RUL
tools/        command line programs you run to train, evaluate and score
config.py     every setting the above uses (paths, sample rates, thresholds)
```

The simulator (`simulator/`, `rendering/`, `ui/`, `monitoring/`) is not covered
here. None of the files below import it: the dataset side runs on its own.

Contents

1. The big picture
2. Concepts used everywhere
3. `datasets/` file by file
4. `predictive/` file by file
5. `tools/` file by file
6. Settings in `config.py`
7. Known limitations
8. Where to change what
9. Improving XJTU → PHM: every experiment and the thinking behind it

---

## 1. The big picture

### Three kinds of data, three jobs

| Data | Where it comes from | What we do with it |
| --- | --- | --- |
| **Mechanical-datasets** | SEU gearbox test rig (CSV) and CWRU bearings (`.mat`) | Anomaly detection and fault classification: is this machine healthy, and which fault? |
| **XJTU-SY** and **PHM 2012** | Bearings run until they broke, a short vibration snapshot every minute (XJTU) or every 10 s (PHM) | Remaining useful life (RUL): how much of its life is left? |
| **Partner data** | Your partner's robot recordings (telemetry, vibration, current, power, temperature) | Whatever the labels allow: anomaly detection, classification, RUL |

All three are **real machine data**, never robot simulation data. The code
keeps them apart and labels them `REAL_MACHINE`.

### The same four steps for every dataset

```
 1. FIND        which files exist, what each one is (label, condition, run)
      |         -> adapter.discover()  returns Recording objects
 2. LOAD        read the numbers from the file
      |         -> load_signals() / read_snapshot() / load_file()
 3. FEATURES    cut the signal into pieces, compute ~10 numbers per piece per channel
      |         -> extract_features() / extract_run() / extract()
 4. LEARN       train on some pieces, test on pieces the model never saw
                -> AnomalyModel, RULModel, RandomForestClassifier
```

An **adapter** is a class that knows one dataset's file layout and does steps
1 to 3. Everything after step 3 only sees a table of numbers, so the learning
code is shared between datasets.

### Who calls whom

```
tools/mechanical_anomaly.py ──> datasets/mechanical_dataset.py ──> datasets/signal_features.py
          │                                                              ▲
          └──────────────> predictive/anomaly_engine.py                  │
                                                                         │
tools/rul.py ──> predictive/dataset_jobs.py ──> datasets/xjtu_sy.py ─────┤
                         │                      datasets/phm2012.py ─────┘
                         │                        (both via datasets/run_to_failure.py)
                         ├──> predictive/rul_model.py
                         └──> predictive/fault_validation.py

tools/partner_data.py ──> datasets/partner_dataset.py ──> datasets/partner_schema.py
          │                                         └──> datasets/partner_features.py
          └──> predictive/partner_models.py ──> predictive/anomaly_engine.py
                                            └──> predictive/rul_model.py
```

The app's **RUL tab** (`ui/rul_panel.py`) runs the same `predictive/dataset_jobs.py`
functions as `tools/rul.py`, in a separate process, so both give identical results.

---

## 2. Concepts used everywhere

### Segments, windows and snapshots

A raw signal has thousands of samples per second. Models cannot use that
directly, so the signal is cut into short pieces and each piece is summarised
by a few numbers (features).

- **Segment** (Mechanical-datasets): 2048 consecutive samples, about 0.4 s of SEU
  data or 0.17 s of CWRU data. Segments do not overlap.
- **Snapshot** (XJTU-SY, PHM 2012): one file is one short recording, and is used
  whole: 1.28 s (XJTU) or 0.1 s (PHM).
- **Window** (partner data): a sliding window of `window_seconds`, moving forward
  by `hop_seconds`. Windows overlap when the hop is shorter than the window.

### Features

Numbers computed from one piece of signal. The ones used most:

| Feature | Formula | What it tells you |
| --- | --- | --- |
| RMS | sqrt(mean(x²)) | Overall vibration energy. Rises as a machine wears |
| std | standard deviation | Spread around the mean |
| p2p (peak to peak) | max − min | Largest swing |
| crest factor | peak / RMS | Spikiness. Early bearing damage makes short impacts, so this rises before RMS does |
| kurtosis | mean(z⁴), z = (x − mean)/std | Also spikiness. About 3 for normal random noise, higher with impacts |
| skewness | mean(z³) | Asymmetry of the signal |
| energy | sum(x²) | Like RMS but grows with segment length |
| trend | least squares slope | Is the signal drifting up or down within the piece? |
| dominant frequency | frequency with the largest spectrum peak | Often the shaft speed or a gear mesh frequency |
| spectral centroid | spectrum weighted average frequency | Shifts upward when damage adds high frequency content |
| band energies | share of spectrum energy in 4 frequency bands | Where the energy sits: 0–10%, 10–25%, 25–50%, 50–100% of Nyquist |

The spectrum is computed with an FFT after multiplying by a Hanning window,
which reduces smearing at the segment edges. The first bin (0 Hz, the average
value) is removed before the frequency features are computed.

**Nyquist** is half the sample rate: the highest frequency a sampled signal can
contain. At 5120 Hz, Nyquist is 2560 Hz.

### Leakage, and why the splits look the way they do

**Leakage** means test data influenced training, which makes results look better
than they really are. Neighbouring pieces of one recording are almost identical,
so a random shuffle would put near copies into both training and testing. The
code prevents that in three ways:

1. **Split by time, never shuffle** (Mechanical-datasets): the first ~65% of
   each recording trains, then a gap of 5 segments, then the last 30% tests.
2. **Split by whole run** (RUL, partner data): a bearing or recording session is
   entirely training or entirely testing. Leave-one-run-out cross validation
   trains on all runs but one and tests on that one, for every run in turn.
3. **Scalers fit on training data only**, inside the model pipeline.

### Anomaly detection without fault examples

The anomaly models learn only **healthy** data. Anything that does not look
healthy is flagged, including fault types never seen before. That matches real
maintenance, where you rarely have examples of every fault.

- **Isolation Forest**: builds random trees that split the data. Unusual points
  get isolated in few splits, so they get a high anomaly score.
- **Threshold**: the 95th percentile of scores on healthy data the forest did not
  train on. By construction, about 5% of healthy windows score above it.
- **Robust z score**: (x − median) / (1.4826 × MAD), where MAD is the median
  absolute deviation. Like a normal z score but not stretched by a few odd
  windows. Used to explain *which* features made a window anomalous.

### One model per setup

Healthy vibration depends on the machine, its speed and its load. A single
model trained on every setup at once learns that "healthy" covers everything,
and then detects nothing. So every anomaly model is trained for one **setup**
only:

- Mechanical-datasets: `source | subset | condition`, e.g. `SEU | gearset | SEU_20-0`
- Partner data: `condition | joint N`, e.g. `load_5kg | joint 2`

### RUL fraction

```
life_fraction = time since start / time of failure       0 at the start, 1 at failure
RUL_fraction  = 1 − life_fraction                        1 when new, 0 when failed
```

A fraction instead of hours, so bearings with very different lifetimes (38
minutes to 7 hours) can be compared and trained together.

---

## 3. `datasets/` file by file

### `datasets/base_adapter.py` — shared building blocks

Used by the Mechanical, XJTU-SY and PHM 2012 adapters.

**`DOMAIN_REAL_MACHINE = "REAL_MACHINE"`**
A label stamped on every recording so real data is never mistaken for simulation.

**`natural_key(path)`**
Sort key that orders file names the way a person would: `acc_9.csv` before
`acc_10.csv`. Plain text sorting would put `10` before `9`. It splits the name
into text and number pieces and compares numbers as numbers.

**`file_number(path)`**
Returns the last number in a file name (`acc_00123.csv` → 123, `57.csv` → 57).
Used to compute each snapshot's time, so the timing stays correct even if some
files are missing.

**`class Recording`**
Describes one recording, without loading its data.

| Attribute | Meaning |
| --- | --- |
| `dataset` | `MECHANICAL`, `XJTU_SY` or `PHM2012` |
| `recording_id` | unique name, used for the cache file name |
| `run_id` | the leakage unit: never split between training and testing |
| `system` | physical machine, e.g. "rolling bearing" |
| `domain` | always `REAL_MACHINE` |
| `condition` | operating setting, e.g. `SEU_20-0`, `1800rpm_4000N` |
| `label` | `HEALTHY`, a fault name, or `RUN_TO_FAILURE` |
| `files` | list of file paths (one for Mechanical, hundreds for a bearing run) |
| `sample_rate` | samples per second |
| `channels` | channel names |
| `snapshot_interval` | seconds between snapshots (run-to-failure only) |
| `meta` | extra facts, e.g. `source`, `subset`, fault diameter |

- `signature()` — a fingerprint of the files (names, sizes, modification
  times). If any file changes, the fingerprint changes and cached features are
  recomputed automatically.
- `describe()` — the attributes as a dictionary, for reports and the UI.

**`class DatasetAdapter`** — the parent class every adapter extends.

- `name`, `role`, `system` — text shown in reports.
- `available()` — does the dataset folder exist?
- `discover()` — each adapter must implement this: find the recordings.
- `summary(recordings)` — counts recordings, files, labels and conditions.
- `cache_path(recording, kind)` — where the feature cache for one recording is
  stored: `data/processed/<DATASET>/<recording>_<kind>.npz`.
- `load_cache(path, signature)` — returns the cached arrays, or `None` if there
  is no cache or the files changed since it was written.
- `save_cache(path, signature, **arrays)` — writes the cache. It writes to a
  temporary file first and renames it, so a crash never leaves half a cache.

Why caching matters: parsing the full Mechanical-datasets CSVs (80 MB each) or
thousands of XJTU snapshots takes minutes. The second run takes seconds.

---

### `datasets/signal_features.py` — vibration and torque features

**`VIBRATION_STATS`, `TORQUE_STATS`**
The ordered names of the features computed per channel.

**`band_names()`**
`band_energy_0` … `band_energy_3`, one per entry in `config.SPECTRAL_BANDS`.

**`vibration_feature_names(prefix)`**, **`torque_feature_names(prefix)`**
Full column names, e.g. `planetary_x_rms`, `torque_trend`. The order always
matches what the functions below return.
**`vibration_features(signal, sample_rate)`**
Computes the 14 vibration features for one channel of one piece of signal:

1. Subtracts the mean, so a sensor offset does not affect anything.
2. Time domain: RMS, std, peak, kurtosis, skewness, energy. Kurtosis and
   skewness are set to 0 for a flat signal instead of dividing by zero.
3. Frequency domain: FFT of the Hanning windowed signal, power spectrum,
   0 Hz bin removed, then dominant frequency, spectral centroid, total spectral
   energy, and the four band energy shares.
4. Returns them in the same order as `vibration_feature_names()`.

**`torque_features(signal, sample_rate)`**
Mean, RMS, std, peak (largest absolute value) and trend (slope in units per
second, by least squares) of the torque channel.

**`parse_numeric_text(text, columns)`**
Fast reader for files of numbers.
- Accepts comma, tab or semicolon separators (PHM uses both commas and
  semicolons; SEU uses tabs in some files and commas in others).
- Drops a trailing separator at the end of a line (SEU files have one).
- Drops the last line if the file does not end with a newline, because a
  partially downloaded file can be cut in the middle of a number.
- Returns an array with `columns` columns, dropping any incomplete final row.

---

### `datasets/mechanical_dataset.py` — Mechanical-datasets adapter

Repository: github.com/cathysiyu/Mechanical-datasets. Two sources inside:

| Folder | Source | Content |
| --- | --- | --- |
| `gearbox/gearset/*.csv` | SEU (Southeast University) | gear faults: health, chipped, miss, root, surface |
| `gearbox/bearingset/*.csv` | SEU | bearing faults: health, ball, inner, outer, comb |
| `dataset/*.mat` | CWRU (Case Western Reserve) | bearing faults at 4 loads: normal, B, IR, OR |

**`SEU_LABELS`, `CWRU_LABELS`**
Translate file name words into labels: `Chipped_20_0.csv` → `GEAR_CHIPPED_TOOTH`,
`IR007_0.mat` → `BEARING_INNER_RACE`.

**`class MechanicalDatasetAdapter(DatasetAdapter)`**

`discover()`
- **SEU**: file names look like `<fault>_<speed>_<load>.csv`. The regular
  expression `([A-Za-z]+)_(\d+)_(\d+)` pulls out the three parts. The condition
  becomes `SEU_20-0` or `SEU_30-2`. Files with unknown fault words are skipped.
- **CWRU**: names like `IR007_0.mat` (inner race, 0.007 inch fault, load 0),
  `OR007@6_0.mat` (outer race, `@6` is the fault position) and `normal_0.mat`.
  The condition is `CWRU_load0` … `CWRU_load3`.
- Each file becomes one Recording with its own `run_id`.

`load_signals(recording)` → an array of shape (samples, channels).

`_load_seu(path, max_rows)`
- Skips the 16 line text header until the line starting with `Data`.
- Reads at most `SEU_SEGMENT_SAMPLES × SEU_MAX_SEGMENTS_PER_FILE` rows
  (2048 × 240 = 491,520), enough for 240 segments, instead of all 1 million.
- Returns 8 columns: motor vibration, planetary x/y/z, motor torque, parallel x/y/z.

`_load_cwru(path)`
- Reads the `.mat` file and takes the fan end accelerometer (`..._FE_time`),
  falling back to the drive end (`..._DE_time`).
- **Resamples the healthy files.** `normal_*.mat` was recorded at 48 kHz, the
  fault files at 12 kHz. This was checked against the shaft speed (1796 rpm =
  29.9 Hz only lines up under those rates). Healthy files are decimated by 4 so
  every file is 12 kHz. Without this, every frequency feature differed between
  healthy and faulty because of the sample rate, not the damage.

`segment(recording, signals)`
Cuts the signal into non-overlapping pieces of 2048 samples, at most 240 (SEU)
or 120 (CWRU). Shape (segments, 2048, channels).

`feature_names(recording)`
- CWRU: 14 vibration features for the one channel.
- SEU: 14 vibration features for each of planetary x, y, z and motor vibration,
  plus 5 torque feat ures = 61 features.
  (The parallel gearbox channels are not used: the dataset README says rows 2 to
  4, the planetary gearbox, are the effective ones.)

`extract_features(recording)`
Returns (features array with one row per segment, feature names). Uses the
cache when the files have not changed. The cache is named `segments_v2`: the
`v2` was added when the CWRU resampling fix changed the results, so old caches
are ignored rather than reused.

---

### `datasets/run_to_failure.py` — shared code for XJTU-SY and PHM 2012

**`class RunToFailureAdapter(DatasetAdapter)`**

`snapshot_channels = ("horizontal", "vertical")` — both datasets have two
accelerometers.

`read_snapshot(path)` — implemented by each dataset: returns (samples, 2).

`feature_names()` — 14 vibration features × 2 channels = 28 names, e.g.
`horizontal_rms`, `vertical_kurtosis`.

`snapshot_times(recording)`
Time of each snapshot in seconds: (file number − first file number) ×
snapshot interval. Using the file number, not the position in the list, keeps
times correct when some files are missing.

`extract_run(recording, progress=None, cancel=None)`
Computes features for every snapshot of one bearing (cached), then the labels:

```
times          seconds since the first snapshot
life_fraction  times / last time            0 → 1
rul_fraction   1 − life_fraction            1 → 0
total_life_s   time of the last snapshot    = the failure time
```

It assumes the last snapshot is the failure. That is true for both datasets:
the tests were stopped when vibration exceeded a failure threshold.

---

### `datasets/xjtu_sy.py` — XJTU-SY adapter

Layout after download:
`XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/1.csv, 2.csv, …`
Three conditions (35 Hz 12 kN, 37.5 Hz 11 kN, 40 Hz 10 kN), 5 bearings each.
One snapshot per minute, 32,768 samples at 25.6 kHz.

**`discover()`**
Walks the folder tree (so an extra wrapping folder is fine), finds folders
named `Bearing<n>_<n>`, collects files named only with a number (`12.csv`), and
takes the condition from the parent folder name.

**`read_snapshot(path)`**
Tries to read the first line as numbers. If that fails, it is a header
(`Horizontal_vibration_signals,Vertical_vibration_signals`) and is skipped.
Returns 2 columns. Raises an error for files with fewer than 16 rows.

**Status:** written to the documented layout and tested on a synthetic folder
with that structure. Not yet run on the real download.

---

### `datasets/phm2012.py` — PHM 2012 / PRONOSTIA adapter

Layout: `Learning_set/Bearing1_1/acc_00001.csv …` and `Full_Test_Set/Bearing1_3/…`.
One snapshot every 10 s, 2560 samples at 25.6 kHz.

**`PHM_CONDITIONS`** — the first digit of the bearing name is the condition:
1 = 1800 rpm 4000 N, 2 = 1650 rpm 4200 N, 3 = 1500 rpm 5000 N.

**`discover()`**
Looks only in `Learning_set` and `Full_Test_Set` (`config.PHM_COMPLETE_SETS`).
`Test_set` is skipped on purpose: its runs are cut short before failure, and
the true remaining life is not in the repository, so they cannot be labelled.
`temp_*.csv` files are ignored. The subset is stored in `meta["subset"]`, which
the `PHM_LEARNING_TO_FULL_TEST` protocol uses to split train and test.

**`read_snapshot(path)`**
Each row has 6 numbers: hour, minute, second, microsecond, horizontal,
vertical. Returns only the last two. Handles comma and semicolon files.

---

### `datasets/feature_schema.py` — common vocabulary for both domains

Simulation telemetry and real machines measure different things (the simulator
has tracking error but no vibration; bearing rigs are the opposite). This file
maps both onto shared categories without inventing missing values.

**`CATEGORIES`** — `LOAD, MOTION_ERROR, VIBRATION, ENERGY, DEGRADATION, ANOMALY`.

**`CATEGORY_OF`** — which category each abstract feature belongs to, e.g.
`vibration_kurtosis → VIBRATION`, `torque_RMS → LOAD`.

**`SIMULATION_SOURCE`** — which simulator monitoring feature supplies each
abstract feature.

**`class ConditionFeatures`**
- `set(name, value)` — stores a value; `None` or NaN removes it (missing stays missing).
- `get(name)`, `category(name)`, `available_categories` — look up values.
- `vector(names)` — returns (values, mask). Missing values are NaN with mask
  False. They are never filled with a guess.
- `from_simulation(...)`, `from_vibration(...)` — build one from either domain.
- `to_dict()` — for reports.

This is a design layer for comparing domains later. The current training code
works directly on each dataset's own features and does not need it.

---

### `datasets/partner_schema.py` — reads the partner's description files

A partner dataset folder contains `dataset.json` (what was measured),
`recordings.csv` (one row per file) and the data files.

**Constants**
- `CHANNEL_TYPES` — allowed channel types: position, target_position, velocity,
  acceleration, torque, current, voltage, power, vibration, temperature,
  ambient_temperature, force, other.
- `LABEL_MODES` — `fault_labels`, `run_to_failure`, `unlabeled`.
- `FORMATS` — `csv`, `xlsx`, `mat`.
- `RECORDING_COLUMNS` — expected columns of `recordings.csv`. Only `file` and
  `run_id` are required.

**`class SchemaError(ValueError)`** — raised with a readable message for any
problem in the description files.

**`class Channel`** — one channel from `dataset.json`: `name`, `column` (the
column name in the data file), `type`, `unit`, `scale` (multiplier, e.g. 0.001
to convert mA to A). Rejects unknown types.

**`class Recording`** — one row of `recordings.csv`: file path, run, asset,
condition, label, `fault_joint`, `start_time_s`, `failure_time_s`, notes.
`fault_joint` says which joint the fault is on; other joints in the same file
are then treated as healthy.

**`_optional_float(value)`** — empty cell → `None`, otherwise a number.

**`class DatasetSchema`** — all of `dataset.json` with defaults filled in from
`config.py`: format, sample rate, time and joint columns, delimiter, Excel sheet,
MAT variable, `window_seconds`, `hop_seconds`, `temperature_window_seconds`,
`warmup_skip_seconds`, label mode, healthy label, channels, recordings.

**`load_schema(root)`** — reads and checks both files. Errors reported:
missing files, invalid JSON, unknown format or label mode, no channels, neither
sample rate nor time column, duplicate channel names, missing required columns,
and (for `fault_labels`) recordings with no label.

---

### `datasets/partner_features.py` — features chosen by channel type

**Constants**
- `GENERIC` — 9 features for every channel: mean, std, rms, min, max, p2p,
  kurtosis, skewness, trend.
- `SPECTRAL` — dominant frequency, spectral centroid and 4 band energies, added
  for `vibration`, `current`, `torque` and `force` channels, but only when the
  sample rate is at least 200 Hz (`PARTNER_SPECTRAL_MIN_RATE_HZ`).
- `THERMAL` — mean, max, rate in °C per minute, rise since the start of the
  recording. Plus `rise_over_ambient` if an ambient channel exists, and
  `rise_per_watt` if a power channel exists.

**`thermal_feature_names(name, has_ambient, has_power)`** — the column names.

**`thermal_features(values, times, start_value, ambient, power)`**
Computed over the longer temperature window:
- `rate_c_per_min` — slope of temperature × 60.
- `rise_since_start` — current level minus the level at the start of the
  recording. Only uses the past, so it could run live.
- `rise_over_ambient` — motor temperature minus room temperature.
- `rise_per_watt` — the rise divided by mean mechanical power. A motor working
  harder is hotter, and that is normal; a motor that is hotter *for the same
  power* has a cooling problem. This separates the two.

**`channel_feature_names(name, type, sample_rate)`** — the feature names for
one channel. Ambient temperature only gets its mean.

**`_generic(x, t)`** — the 9 generic statistics. Trend is the least squares
slope over time.

**`_spectral(x, sample_rate)`** — the frequency features, same method as
`signal_features.vibration_features`.

**`window_features(signals, times, channels, sample_rate)`** — all features for
one window, as (name, value) pairs in a fixed order. The spectrum is only
computed when a channel needs it.

---

### `datasets/partner_dataset.py` — loads, checks and windows partner data

**`class Stream`** — the continuous signals of one joint from one recording:
`times` and a dictionary of channel arrays.
- `group` — the `run_id` (the leakage unit).
- `stream_id` — `run|asset|joint`, unique per stream.

**`class PartnerDatasetAdapter`**

`__init__(root)` — reads the description through `load_schema`.

Loading:
- `load_file(recording)` — picks the reader for the format.
- `_load_csv(path)` — header row + data rows; blank rows skipped.
- `_load_xlsx(path)` — first sheet, or the one named in `dataset.json`.
- `_table(header, rows)` — converts cells to numbers. Text that is not a number
  becomes NaN (missing). The joint column is kept as text.
- `_load_mat(path)` — either one matrix variable (`mat_variable`, transposed
  automatically if stored the other way round) or one variable per column.

Streams and time:
- `streams(recording)` — checks every channel's column exists, then splits the
  rows by joint (files may contain several joints, one block after another).
  Applies each channel's `scale`.
- `_times(header, rows, recording)` — from the time column (shifted to start at
  0, plus `start_time_s`) or from the sample rate.

Quality checks:
- `validate()` — runs over every recording and returns errors (must fix) and
  warnings (results will be weaker). Never stops at the first bad file.
  Beyond the per stream checks below, it warns when there is no healthy data,
  when a label appears in only one run (it can then never be tested on an
  unseen run), when a run-to-failure dataset has fewer than 3 runs or no
  failure time, and when there is only one run in total.
- `_stream_quality(stream, report)` — per stream: time must increase; measured
  sample rate must match the declared one within 5%; temperature must not still
  be rising more than 1 °C/min after the warm-up skip; a missing ambient channel
  is reported; channels with more than 5% missing values or a constant value
  (a disconnected sensor) are reported; streams shorter than one window are
  reported.

Features:
- `sample_rate(stream)` — declared rate, or the median of the time steps.
- `channel_list()` — channels windowed the normal way (temperature channels are
  handled separately), plus a derived `tracking_error` channel when both
  position and target position exist.
- `thermal_channels()`, `_named(type)` — find channels by type.
- `feature_names(rate)` — every column name, in the order `extract` fills them.
- `extract(progress, cancel)` — the main loop. For every stream:
  1. Adds `tracking_error = target − position` when possible.
  2. Slides a window of `window_seconds` forward by `hop_seconds`.
  3. Skips windows that end inside the warm-up period.
  4. Skips windows containing any missing value (and counts them).
  5. Computes the normal features, then the temperature features over the
     longer trailing temperature window.
  6. Records the run, stream, label, setup (`condition | joint N`), asset,
     joint and window end time.

  For `run_to_failure`, it also computes `life_fraction` per run, from the
  declared `failure_time_s` (or the last sample if none was given).
  Returns a dictionary with the feature matrix `X` and all those labels.
- `stream_label(recording, stream)` — the label for one joint: when
  `fault_joint` is set, only that joint gets the fault label; the others are
  healthy.
- `schema_hash()` — short fingerprint of the channel layout, stored in trained
  models, so scoring refuses data with different channels.

---

## 4. `predictive/` file by file

### `predictive/anomaly_engine.py` — shared anomaly detector

Used by both the Mechanical-datasets tool and the partner pipeline.

**`class AnomalyModel`** — one model for one setup.

`__init__(setup, feature_names, …)` — stores the settings and finds the
temperature columns (names ending in `_rise_over_ambient`, `_rise_per_watt` or
`_rise_since_start`).

`fit(healthy)` — trains on healthy data only.
1. Accepts one array or a list of arrays (one per recording). From **each**
   recording, the first 70% goes to fitting and the last 30% to calibration, so
   every healthy recording helps set the threshold.
2. `RobustScaler` (scales by median and interquartile range, not mean and std,
   so a few odd windows do not distort it) then `IsolationForest` with 300 trees.
3. Stores each feature's median and MAD for explanations. The MAD is floored at
   half the std and 5% of the median, because features that are nearly constant
   when healthy (such as a dominant frequency locked to shaft speed) would
   otherwise produce z scores in the thousands.
4. Threshold = 95th percentile of scores on the calibration part.
5. For temperature columns, stores the healthy median as a reference.
6. Refuses to train with fewer than about 15 healthy windows.

`raw_scores(x)` — the anomaly score of each window (higher = more unusual).

`thermal_ratios(x)` — for each window, the largest ratio of a temperature
feature to its healthy median. 1.0 is normal; 2.0 means twice the usual rise.

`score(x, top=3)` — per window: the score, score / threshold, whether it is
anomalous, the thermal ratio, and the 3 features that deviate most. A window is
anomalous if the forest score is above the threshold **or** the thermal ratio
reaches 1.6. The thermal check exists because a cooling fault moves a few slow
features moderately, and a forest spread over ~100 features averages that
away: on test data the forest alone caught 6.6% of cooling windows, with the
thermal check 75%.

`evaluate(healthy_test, faulty_by_label)` — reports:
- false alarm rate: share of unseen healthy windows flagged;
- per fault label: detection rate, share caught by the thermal check, ROC AUC,
  median score / threshold, and the most deviating features.

ROC AUC is computed from the forest score only. It measures how well scores
separate healthy from faulty without choosing a threshold: 1.0 is perfect, 0.5
is guessing.

**`_top_features(model, x)`** — the 5 features with the largest median robust z
across a group of windows.

**`time_blocks(n, test_fraction, gap)`** — indices for the time based split:
train block, gap, test block. Never shuffled.

**`save_models(path, models, metadata)`**, **`load_models(path)`** — save and
load a dictionary of setup → model with joblib. Loading checks the file really
is an anomaly bundle.

---

### `predictive/fault_validation.py` — classification and anomaly check on Mechanical-datasets

Called by the RUL tab's **Fault Validation** button.

**`split_blocks(n)`** — same idea as `time_blocks`: first block trains, last 30%
tests, 5 segment gap.

**`_setup_key(info)`** — `system | condition`, the unit for anomaly models.

**`_anomaly_by_setup(items, scaler)`** — one Isolation Forest per setup, trained
on 70% of the healthy training block, threshold from the other 30%, then
detection rate and ROC AUC per fault label on the test blocks.

**`validate(recordings_features)`**
Groups recordings with the same feature layout (SEU and CWRU have different
features, so they are validated separately), then:
1. Time based train/test split per recording.
2. `StandardScaler` fitted on training segments only.
3. `RandomForestClassifier` (300 trees) to predict the fault label. Reports
   accuracy, macro F1 (the average F1 over labels, so rare labels count as much
   as common ones), the confusion matrix and the 8 most important features.
4. The anomaly check above.

---

### `predictive/rul_model.py` — remaining useful life

**Constants**
- `BASE_QUANTITIES` — the 10 vibration features used for RUL per channel.
- `LOG_RATIO` — features converted to log(value / starting value): rms, p2p,
  spectral energy. Logarithms turn "doubled" into a fixed step, whatever the
  starting level.
- `SMOOTHED` — rms and kurtosis also get smoothed versions.

**`_ema(values, alpha)`** — exponential moving average: each output is
`alpha × new + (1 − alpha) × previous`. Smooths noise while only using the past.

**`rul_features(run, loudness_reference)`** — turns a run's snapshot features
into RUL features.
1. **Baseline**: the mean of the first 10 minutes (`RUL_BASELINE_SECONDS`), or
   a quarter of the run if shorter. Each bearing is compared with its own
   healthy start, so bearings with different vibration levels become
   comparable. This uses only early data, which is what you would have for a
   real machine.
2. Per feature: log ratio, difference (band energies) or ratio (the rest) to
   the baseline. **Exception, the loudness features** (rms, p2p, spectral
   energy) when `loudness_reference` is `"QUIETEST"`: they are measured against
   the quietest the bearing has been so far (running minimum of a 5 minute
   EMA) instead of its start. See section 9, Fix 1. `"BOTH"` keeps both
   versions. Optional extras (off by default, section 9, Fix 2):
   `RUL_TREND_FEATURES` adds fast EMA − slow EMA, `RUL_DROP_LOUDNESS_FEATURES`
   removes the loudness features.
3. For rms and kurtosis: a fast EMA (alpha 0.30), a slow EMA (0.05) and the
   running maximum of the slow EMA. The running maximum never goes down, which
   matches wear: a bearing does not repair itself.
4. **Health indicator** (for plots): `exp(−running max of the slow log RMS ratio)`,
   1.0 when new, falling toward 0 as vibration grows.

Every feature only uses snapshots up to that time. Nothing from the future
leaks into an earlier prediction.

**`_make_estimator(kind)`** — builds a pipeline `StandardScaler → model`:
- `LINEAR_REGRESSION` — a weighted sum of features. Simple, hard to overfit.
- `RANDOM_FOREST` — 200 trees, each leaf needs 5 samples.
- `GRADIENT_BOOSTING` — 300 shallow trees added one after another.

Because the scaler is inside the pipeline, it is fitted on training runs only.

**`_stack(runs)`** — joins all runs into one table plus the RUL targets and
run ids.

**`assert_disjoint(train, test)`** — stops with an error if any bearing is in
both training and testing.

**`detect_fpt(run, matrix, names)`** — the "first predicting time": the first
snapshot where degradation has started. This bearing's first 10 minutes define
its normal range; degradation = the fast EMA of RMS growth or of kurtosis stays
above `mean + RUL_FPT_K × std` of that range for `RUL_FPT_SUSTAIN_S` seconds.
Uses only past snapshots. Returns the run length if it never happens.

**`postprocess_prediction(prediction, run, training_dataset)`** — cleans up a
predicted RUL curve, using only the past. If the run comes from a dataset the
model was NOT trained on, it is smoothed with a 15 minute EMA
(`RUL_CROSS_DATASET_SMOOTHING_TAU_S`). Optional running minimum
(`RUL_MONOTONE_PREDICTION`, off; section 9 explains why).

**`class RULModel`**
- `fit(runs, loudness_reference, fpt_gate)` — trains and remembers which
  bearings and datasets it used, the loudness reference, and, if `fpt_gate`,
  the **FPT floor**: the median RUL the training bearings had at their FPT.
- `predict_run(run)` — predicted RUL fraction along one bearing (clipped to
  0–1), plus the health indicator. Steps: features with the model's own
  loudness reference → model → if the model has an FPT floor, predictions
  before this bearing's FPT are raised to at least the floor (the "gate") →
  `postprocess_prediction`. Refuses a run with a different feature layout.
- `save(path)`, `load(path)` — joblib files marked `target: RUL_fraction`,
  including `loudness_reference` and `fpt_floor`. Files saved before these
  existed load as `START` and no gate, so old models keep working unchanged.

**`run_metrics(run, prediction)`**
- `mae` — mean absolute error of the RUL fraction. 0.15 means off by 15% of the
  bearing's life on average.
- `rmse` — like MAE but punishes large errors more.
- `r2` — 1.0 perfect, 0 no better than predicting the average, negative worse
  than that.
- `late_life_time_error_percent_life` — converts the predicted fraction f into
  remaining time with `elapsed × f / (1 − f)`, from 50% of life onwards. This
  formula explodes when f is close to 1, so treat large values with caution.

**`aggregate(metrics)`** — averages the metrics over bearings.

**`leave_one_run_out(runs, kind, loudness_reference, fpt_gate)`** — for each
bearing: train on all others (with the same settings as the final model), test
on it. The standard way to test with few bearings.

---

### `predictive/dataset_jobs.py` — the jobs behind the RUL tab and `tools/rul.py`

**`class Cancelled`** — raised when the user presses Cancel.

**`adapters(roots)`** — creates the three dataset adapters, with custom folders
if given.

**`class JobContext`** — passes progress messages to a queue (the app's process
queue, or the printer in `tools/rul.py`) and checks for cancellation.

Jobs (each returns a dictionary):

| Function | What it does |
| --- | --- |
| `job_load` | lists what each dataset folder contains |
| `job_preprocess` | reads every recording once to catch unreadable files; writes `data/processed/<DATASET>/index.json` |
| `_runs` | features for every bearing of one adapter |
| `job_extract` | computes and caches all features |
| `_protocol_runs` | picks training and validation bearings for the chosen protocol, and checks they do not overlap |
| `job_train_rul` | leave-one-run-out cross validation (3 or more training bearings), then fits on all training bearings and saves the model |
| `job_evaluate_rul` | loads a model, refuses if it was trained on any validation bearing, returns per bearing metrics and curves |
| `job_mechanical` | features for Mechanical-datasets, then `fault_validation.validate` |

Protocols:
- `XJTU_TO_PHM` — train on XJTU-SY, validate on PHM 2012. The strongest test:
  different machines, different lab.
- `PHM_LEARNING_TO_FULL_TEST` — train on PHM `Learning_set`, validate on
  `Full_Test_Set`. For use until XJTU-SY is downloaded.

**`run_job(name, params, queue, cancel_event)`** — runs one job and reports
`finished`, `cancelled` or `error` on the queue.

**`process_main(...)`** — entry point when a job runs in a separate process.

---

### `predictive/partner_models.py` — models for partner data

The models chosen depend on `label_mode`:

| label_mode | Models |
| --- | --- |
| `fault_labels` | anomaly detection (healthy recordings only) + fault classification |
| `unlabeled` | anomaly detection, assuming most recordings are normal |
| `run_to_failure` | anomaly detection (first 30% of life counts as healthy) + RUL |

**`group_split(groups, test_fraction, seed, stratify)`** — holds out whole runs
for testing. With `stratify`, it holds out runs from every label, so each label
also appears in training.

**`train_anomaly(data)`** — one `AnomalyModel` per setup. Normal runs are split
by run. If there is only one normal run, it falls back to an early / late time
split and says so ("weaker evidence"). For unlabelled data it also reports the
share of anomalous windows per run.

**`train_classifier(data)`** — `StandardScaler → RandomForestClassifier`,
trained and tested on different runs. Reports accuracy, macro F1, confusion
matrix and top features, and warns about labels that only exist in test runs.

**`rul_features(data)`** — like `rul_model.rul_features`, per stream: each value
relative to that stream's first 10 windows.

**`train_rul(data, kind)`** — leave-one-run-out RUL, reusing `_make_estimator`
from `rul_model.py`. Needs at least 3 runs.

**`train_all(data)`** — runs whichever of the above apply and returns the model
bundle and the report.

**`score(bundle, data)`** — applies a saved bundle to new data. Refuses data
with a different channel layout (`schema_hash`). Per stream: share of anomalous
windows, the worst window and its top features, the predicted fault (majority
vote over windows) and the latest RUL.

---

### Simulator-only files in `predictive/`

These three serve the simulator dashboard, not dataset training:

- `health_model.py` — combines simulator evidence into a 0–100 joint health score.
- `maintenance_engine.py` — turns evidence patterns into maintenance recommendations.
- `event_log.py` — the dashboard's event log and its CSV export.

---

## 5. `tools/` file by file

All tools run from the project root with the `mechanical` env active:
`python -m tools.<name> <command> --help` shows every option.

### `tools/mechanical_anomaly.py`

```
python -m tools.mechanical_anomaly train    [--root FOLDER] [--model FILE]
python -m tools.mechanical_anomaly evaluate [--model FILE]
python -m tools.mechanical_anomaly score    FILE [FILE ...]
```

- `setup_key(recording)` — `source | subset | condition`.
- `load_features(root)` — discovers files and extracts (cached) features,
  grouped by setup.
- `build(setups)` — for each setup with healthy data: time based split, trains
  an `AnomalyModel` on the healthy training blocks, evaluates on the test blocks
  of healthy and faulty recordings.
- `print_reports(reports)` — the table you see.
- `command_train` — the above, then saves `models/mechanical_anomaly.joblib`
  and a JSON report next to it.
- `command_evaluate` — re-runs the test block evaluation for a saved model.
- `command_score` — scores any Mechanical-datasets file window by window and
  shows the worst window's deviating features. The file name must follow the
  dataset's naming, because that is how the setup is found.

### `tools/rul.py`

```
python -m tools.rul info
python -m tools.rul train    [--protocol P] [--model M] [--out FILE]
python -m tools.rul evaluate --model-file FILE [--protocol P] [--curves CSV]
python -m tools.rul compare  [--protocol P]
python -m tools.rul predict  --model-file FILE BEARING_FOLDER
```

- `_Printer` — shows job progress on one line in a terminal, one line per step
  elsewhere (VS Code output panel, log files).
- `_run(job, params)` — runs a `dataset_jobs` job and stops on error.
- `_roots(args)` — dataset folders from `--xjtu` and `--phm`.
- `_fmt`, `_print_metrics`, `_print_summary` — formatting.
- `command_info` — bearings found per dataset.
- `command_train` — cross validation table, then the saved model path and the
  command to evaluate it.
- `command_evaluate` — per bearing validation table; `--curves` writes true and
  predicted RUL along each bearing to CSV for plotting.
- `command_compare` — trains and evaluates all three model types and names the
  one with the lowest validation MAE.
- `_recording_for_folder(folder)` — recognises a PHM folder (`acc_*.csv`) or an
  XJTU-SY folder (`1.csv, 2.csv …`).
- `command_predict` — predicted RUL at 10, 25, 50, 75, 90 and 100% of one
  bearing's recording. Warns if that bearing was used for training.

### `tools/partner_data.py`

```
python -m tools.partner_data template --out FOLDER [--mode fault_labels|unlabeled|run_to_failure]
python -m tools.partner_data validate --root FOLDER
python -m tools.partner_data train    --root FOLDER [--model FILE]
python -m tools.partner_data score    --root FOLDER --model FILE
```

- `EXAMPLE_CHANNELS` — the channels in the example dataset.
- `_thermal(...)`, `_synthetic(...)`, `_write_csv(...)` — generate **synthetic**
  example signals (sine motion, friction, bearing impacts, backlash lag, a
  cooling fault, warm-up). Only for showing the file format and testing the
  code; never mix them with real data.
- `command_template` — writes `dataset.json`, `recordings.csv` and CSV files.
- `command_validate` — prints the `validate()` report; exits with an error code
  when there are errors.
- `command_train` — validates, extracts features, `train_all`, saves the model
  and a JSON report, prints the results.
- `_print_report` — the results table.
- `command_score` — applies a saved model to a new dataset folder.
- `_adapter(root)` — turns schema errors into a readable message.

---

## 6. Settings in `config.py`

### Paths

| Setting | Value |
| --- | --- |
| `EXTERNAL_DATA_DIR` | `data/external` — downloaded datasets |
| `PROCESSED_DIR` | `data/processed` — feature caches (safe to delete) |
| `MECHANICAL_DATASET_DIR` | `data/external/Mechanical-datasets` |
| `XJTU_SY_DIR` | `data/external/XJTU-SY_Bearing_Datasets` |
| `PHM2012_DIR` | `data/external/ieee-phm-2012-data-challenge-dataset` |
| `PARTNER_DATA_DIR` | `data/external/partner` |
| `MODEL_DIR` | `models` — trained models |

### Mechanical-datasets

| Setting | Value | Meaning |
| --- | --- | --- |
| `SEU_SAMPLE_RATE_HZ` | 5120 | from the file header: 2000 Hz frequency limit, 1600 lines = 2.56 × 2000 |
| `SEU_CHANNELS` | 8 names | the column order in SEU files |
| `SEU_VIBRATION_CHANNELS` | planetary x, y, z | channels used for vibration features (plus motor vibration) |
| `SEU_SEGMENT_SAMPLES` | 2048 | samples per segment |
| `SEU_MAX_SEGMENTS_PER_FILE` | 240 | caps reading time |
| `CWRU_SAMPLE_RATE_HZ` | 12000 | fault files |
| `CWRU_NORMAL_SAMPLE_RATE_HZ` | 48000 | healthy files, resampled to 12000 |
| `CWRU_SEGMENT_SAMPLES` | 2048 | |
| `CWRU_MAX_SEGMENTS_PER_FILE` | 120 | |
| `MECH_TEST_FRACTION` | 0.30 | last 30% of each recording is test data |
| `MECH_GAP_SEGMENTS` | 5 | segments dropped between train and test |

### Run-to-failure and RUL

| Setting | Value | Meaning |
| --- | --- | --- |
| `XJTU_SAMPLE_RATE_HZ` / `PHM_SAMPLE_RATE_HZ` | 25600 | |
| `XJTU_SNAPSHOT_INTERVAL_S` | 60 | one snapshot per minute |
| `PHM_SNAPSHOT_INTERVAL_S` | 10 | one snapshot every 10 s |
| `PHM_COMPLETE_SETS` | Learning_set, Full_Test_Set | Test_set is excluded |
| `SPECTRAL_BANDS` | 0–10%, 10–25%, 25–50%, 50–100% of Nyquist | band energy edges |
| `RUL_BASELINE_SECONDS` | 600 | healthy reference = the first 10 minutes of each run |
| `RUL_EMA_FAST_TAU_S` / `RUL_EMA_SLOW_TAU_S` | 168 / 1170 s | smoothing time constants |
| `RUL_LOUDNESS_REFERENCE_BY_PROTOCOL` | XJTU_TO_PHM: QUIETEST, others: START | where loudness growth is measured from (section 9, Fix 1) |
| `RUL_QUIETEST_TAU_S` | 300 s | smoothing before taking the running minimum |
| `RUL_FPT_GATE_BY_PROTOCOL` | XJTU_TO_PHM: on, others: off | no prediction below the floor before degradation starts (section 9, Fix 4) |
| `RUL_FPT_K` / `RUL_FPT_SUSTAIN_S` / `RUL_FPT_HEALTHY_S` | 5 / 120 s / 600 s | degradation start = 5 std above normal for 2 minutes, normal = first 10 minutes |
| `RUL_CROSS_DATASET_SMOOTHING_TAU_S` | 900 s | smooth predictions on a dataset the model was not trained on |
| `RUL_MONOTONE_PREDICTION` | False | force RUL to only go down (rejected, section 9) |
| `RUL_TREND_FEATURES` / `RUL_DROP_LOUDNESS_FEATURES` | False / False | Fix 2 options (rejected, section 9) |
| `RUL_KURTOSIS_NEEDS_GROWTH` | 0 (off) | kurtosis only counts once RMS has grown this much (tested, not adopted, section 9) |
| `HEALTH_INDICATOR_SENSITIVITY` | 1.0 | only affects the health indicator plot |
| `RUL_MODELS` | 3 model types | |
| `RUL_DEFAULT_MODEL` | RANDOM_FOREST | |
| `RUL_RANDOM_STATE` | 11 | fixed seed, so results repeat exactly |
| `RUL_DEFAULT_PROTOCOL` | XJTU_TO_PHM | also available: PHM_LEARNING_TO_FULL_TEST, XJTU_PLUS_PHM_TO_FULL_TEST |
| `RUL_TIME_METRICS_FROM_LIFE` | 0.5 | time error only reported from half of life |

### Partner data and the anomaly engine

| Setting | Value | Meaning |
| --- | --- | --- |
| `PARTNER_DEFAULT_WINDOW_S` / `PARTNER_DEFAULT_HOP_S` | 1.0 / 0.5 | used when `dataset.json` does not set them |
| `PARTNER_SPECTRAL_MIN_RATE_HZ` | 200 | no spectral features below this sample rate |
| `PARTNER_MIN_WINDOW_SAMPLES` | 16 | smallest allowed window |
| `PARTNER_TEMPERATURE_WINDOW_S` | 120 | trailing window for temperature features |
| `PARTNER_DEFAULT_WARMUP_SKIP_S` | 0 | seconds dropped at the start of each recording |
| `PARTNER_WARMUP_RATE_C_PER_MIN` | 1.0 | warn if still warming faster than this |
| `PARTNER_TEST_FRACTION` | 0.3 | share of runs held out |
| `PARTNER_SPLIT_SEED` | 17 | fixed seed for run selection |
| `PARTNER_RUL_BASELINE_WINDOWS` | 10 | |
| `PARTNER_ANOMALY_CALIBRATION_FRACTION` | 0.3 | share of healthy data used for the threshold |
| `PARTNER_ANOMALY_THRESHOLD_PERCENTILE` | 95 | about 5% of healthy windows score above the threshold |
| `PARTNER_MAX_NAN_FRACTION` | 0.05 | warn above 5% missing values |
| `PARTNER_RATE_TOLERANCE` | 0.05 | allowed mismatch between timestamps and declared rate |
| `THERMAL_RATIO_WARN` | 1.6 | thermal alarm at 1.6 × the healthy temperature rise |

The anomaly engine also uses the partner threshold and calibration settings
when it runs on Mechanical-datasets.

---

## 7. Known limitations

- **Mechanical-datasets has one recording per fault per condition.** Train and
  test come from different time blocks of the same recording, not from
  different machines. High scores there show the features separate the faults,
  not that the model would work on a new machine.
- **SEU explanations are dominated by the motor vibration dominant frequency.**
  It may partly reflect differences between recording sessions rather than
  damage. Check it on the full files.
- **The smallest CWRU ball fault** (B007) is detected in only about 58% of
  windows. That is a known hard case, not a bug.
- **RUL smoothing is now time based** (`RUL_TIME_AWARE_SMOOTHING = True`): the
  EMAs use time constants in seconds and the healthy baseline is the first
  `RUL_BASELINE_SECONDS`, so XJTU-SY (a snapshot per minute) and PHM 2012 (every
  10 s) are smoothed over the same real time. Measured effect on the real
  XJTU → PHM validation with linear regression: MAE 0.314 → 0.266, R2 -0.69 →
  -0.30. Set the flag to False to get the old per snapshot behaviour.
- **Transfer between datasets is improved, not solved.** XJTU → PHM went from
  MAE 0.216, R2 0.15, rank 0.53 to MAE 0.181, R2 0.39, rank 0.82 (section 9),
  and now beats training on PHM itself (0.198 / 0.22 / 0.83). Still weak:
  Bearing2_7 (38 minute life, R2 -0.24), Bearing1_1 (rank 0.19) and
  Bearing2_3 (rank 0.32). `rank_correlation` (Spearman) is reported next to
  R2 because an alarm threshold needs the ordering more than the absolute
  number.
- **Some bearings give almost no warning.** On several PHM bearings the only
  clear sign of damage is in the last 10% of life. No vibration-only model can
  predict well before that; this is a limit of the data.
- **Settings were chosen on 6 dev bearings.** That is few. Fix 1 looked good on
  the PHM → PHM dev bearings and then hurt its test bearings badly, which is
  why every setting is also checked on the 11 test bearings and why small dev
  gains are not trusted (section 9).
- **How much data you test on changes which model wins.** On a partial PHM
  download (3 training bearings) linear regression was best; with all 6 PHM
  training bearings, and with the 15 XJTU-SY bearings, the random forest has
  the lowest validation error (see the tables in README.md). Do not carry a
  model choice over from a smaller experiment.
- **Partner results so far come from synthetic data** and say nothing about real
  performance.

---

## 8. Where to change what

| I want to… | Change |
| --- | --- |
| use datasets stored somewhere else | the `*_DIR` paths in `config.py`, or `--root` / `--phm` / `--xjtu` |
| use longer or shorter segments | `SEU_SEGMENT_SAMPLES`, `CWRU_SEGMENT_SAMPLES` (delete `data/processed/` afterwards) |
| add a feature | `datasets/signal_features.py`: add the name to `VIBRATION_STATS` and the value to `vibration_features` in the same position |
| make anomaly detection stricter or looser | `PARTNER_ANOMALY_THRESHOLD_PERCENTILE` (99 = fewer false alarms, fewer detections) |
| try another RUL model | add a branch in `_make_estimator` and its name to `RUL_MODELS` |
| change which bearings train and test | `_protocol_runs` in `predictive/dataset_jobs.py` and `RUL_PROTOCOLS` in `config.py` |
| support a new dataset | a new class extending `DatasetAdapter` (or `RunToFailureAdapter` for run-to-failure data) with `discover()` and a reader |
| force features to be recomputed | delete `data/processed/` |
| change how RUL features are referenced, gated or smoothed | the `RUL_LOUDNESS_*`, `RUL_FPT_*` and `RUL_*SMOOTHING*` settings in `config.py` (section 9) |
| try a new RUL idea without touching the main code | add a variant to `results/experiments/xjtu_to_phm_rank.py` |

---

## 9. Improving XJTU → PHM: every experiment and the thinking behind it

This section is the story of how the XJTU → PHM result was improved on
28–29 September 2026, including the ideas that failed. It is written so you can
follow the reasoning and explain it in an interview.

### The starting point

Train on all 15 XJTU-SY bearings, predict the 17 PHM 2012 bearings (a different
lab, machine, speed and load). Random forest, full data:

| | MAE | R2 | Rank |
| --- | --- | --- | --- |
| Always guess 0.5 (the "dumb" baseline) | 0.250 | 0.00 | — |
| **Start** | 0.216 | 0.15 | 0.53 |
| Training on PHM itself (PHM → PHM), for comparison | 0.198 | 0.22 | 0.83 |

The model barely beat guessing. Full outputs are frozen in
`results/baseline_2026-09-28/`.

How to read the metrics:
- **MAE**: average error as a fraction of life. 0.20 = off by 20% of life.
- **R2**: 1 = perfect, 0 = no better than guessing the average, < 0 = worse.
- **Rank**: does the prediction fall in the right order as the bearing ages?
  1 = perfect order. An alarm needs this most.

### The method (the same for every idea)

1. **Observe** a failure in the numbers.
2. **Explain** it with a guess (hypothesis).
3. **Idea** that fixes that specific cause.
4. **Test**: change one thing, compare with the current best.
5. **Decide** and write it down, including failures.

Two rules keep the results honest:

- **Choose on dev, confirm on test.** The 17 PHM bearings are split:
  - **dev** = the 6 PHM Learning_set bearings, used to CHOOSE between ideas;
  - **test** = the 11 Full_Test_Set bearings, never used to choose, only to
    confirm.

  If you try 20 ideas and keep whichever scores best on the test bearings, you
  have fitted the test set and the number is no longer honest.
- **Small gains are noise.** With 6 dev bearings, a difference like MAE −0.003
  is not real. An idea is kept only if it is clearly better on dev AND the test
  bearings agree.

All experiments run through `results/experiments/xjtu_to_phm_rank.py`, which
loads the data once and scores each variant on dev, test and all 17 bearings.
Run it from the project root, for example
`python results/experiments/xjtu_to_phm_rank.py baseline quiet300`
(add `PER_BEARING=1` in front to see every bearing).

### Step 1 — Noise: smooth the predictions (kept, cross-dataset only)

**Observe.** The model judges each snapshot on its own. A PHM snapshot is only
0.1 s of vibration, so it wobbles for reasons that have nothing to do with wear
(ball position, small shocks, sensor noise). On Bearing3_3 the raw prediction
jumped 5% of life between snapshots 10 s apart. Real wear cannot change that
fast.

**Idea.** Average the predictions over the last few minutes (a time based EMA,
using only the past), like judging your weight by a weekly average.

**Rejected on the way: forcing RUL to only go down.** "Damage does not heal",
so a running minimum sounded right. Rank jumped to 0.94, but:
- every model got rank 0.94–1.00, even models with negative R2. When a trick
  makes every model look perfect, the metric has stopped measuring anything;
- Bearing1_5 had rank 0.94 with R2 −0.77: its prediction dropped far too early
  and then was not allowed to recover;
- MAE did not improve.

**Result** (smoothing over 15 minutes, 900 s, chosen on dev):

| | dev MAE / R2 / rank | test MAE / R2 / rank |
| --- | --- | --- |
| before | 0.240 / 0.02 / 0.51 | 0.203 / 0.22 / 0.54 |
| smoothing 900 s | 0.213 / 0.22 / 0.76 | 0.197 / 0.29 / 0.61 |

**Side effect found.** On PHM → PHM, smoothing made things worse (MAE 0.198 →
0.216): a model trained on PHM is already on PHM's time scale, so smoothing
only adds delay. **Decision:** smooth only when a model predicts on a dataset
it was not trained on (`RUL_CROSS_DATASET_SMOOTHING_TAU_S`).

### Step 2 — Why the bad bearings are bad (looking at the data)

Smoothing fixed noise, but some bearings were still in the wrong order
(Bearing2_5 rank −0.81). Averaging each tenth of life showed why. Vibration
growth on a log scale (0 = same as at the start, negative = quieter):

| Bearing | 0–10% | 30% | 50% | 70% | 90–100% |
| --- | --- | --- | --- | --- | --- |
| Bearing2_5 | 0.03 | −0.79 | −0.85 | −0.90 | −0.89 |
| Bearing1_5 | 0.00 | −0.23 | −0.30 | −0.34 | −0.13 |
| Bearing1_7 | −0.09 | −0.37 | −0.15 | −0.14 | +0.22 |
| XJTU bearings at failure | | | | | +1.2 to +2.4 |

Three findings:
1. **Many PHM bearings get quieter, not louder** — probably run-in: a new
   bearing is rough and settles. The reference ("the first 10 minutes") falls
   in that rough period, so the bearing looks healthier than new for most of
   its life. Bearing2_5 ends at about 40% of its starting vibration; the model
   read "getting quieter" as "getting healthier" and its prediction ROSE.
2. **The warning is late**: kurtosis (sharp spikes from cracks) jumps only in
   the last 10% of life.
3. **XJTU failures are much louder** (3–11× the start), so the model learned
   "failure = loud".

Each finding became one fix below.

### Fix 1 — Measure loudness from the quietest point so far (kept, XJTU → PHM only)

**Idea.** Compare each snapshot not with the first 10 minutes but with the
quietest the bearing has been so far (running minimum of a smoothed signal).
Still uses only the past. A bearing that starts rough and settles now reads ~0
for most of its life and rises at the end — the right shape.

**Prediction written down first:** it should help Bearing2_5, 1_5 and 1_7.

**Result** (smoothing window for the minimum chosen on dev: 300 s was the only
setting at least as good as before on all three dev metrics):

| | dev MAE / R2 / rank | test MAE / R2 / rank |
| --- | --- | --- |
| before (Step 1) | 0.213 / 0.22 / 0.76 | 0.197 / 0.29 / 0.61 |
| **Fix 1** | 0.212 / 0.24 / 0.76 | **0.187 / 0.35 / 0.84** |

The predicted bearings improved the most — the diagnosis was right:

| Bearing | Rank before → after | R2 before → after |
| --- | --- | --- |
| Bearing2_5 | −0.81 → 0.57 | −0.52 → −0.02 |
| Bearing1_5 | −0.08 → 0.88 | −0.52 → 0.42 |
| Bearing1_7 | 0.30 → 0.83 | −0.18 → 0.33 |
| Bearing2_1 | 0.33 → 0.91 | −0.01 → 0.16 |

**Side effect found.** On PHM → PHM, Fix 1 was much worse (MAE 0.198 → 0.279),
even though its own dev check (cross validation) said it helped (0.200 →
0.188). Likely reason (a hypothesis, not proven): all PHM bearings share the
same run-in, so a model trained and tested on PHM uses "how much quieter than at
the start" as a clock for where the bearing is in its life. Fix 1 removes that
clock. A model moving to a new machine cannot rely on that pattern anyway.
**Decision:** the reference is set per protocol
(`RUL_LOUDNESS_REFERENCE_BY_PROTOCOL`) and stored in the model file, so an old
model is never fed features that mean something different. **Lesson:** 6 dev
bearings can mislead; that is why the test bearings are always checked too.

### Fix 2 — Features that do not depend on "how loud" (rejected)

**Idea.** Loudness differs between machines; the SHAPE of the signal
(kurtosis, crest factor, spectral centroid, band energies) might transfer
better. Also tried: trend features (fast EMA − slow EMA: "is it rising right
now?").

| | dev MAE / R2 / rank | test MAE / R2 / rank |
| --- | --- | --- |
| Fix 1 (current) | 0.212 / 0.24 / 0.76 | 0.187 / 0.35 / 0.84 |
| + trend features | 0.209 / 0.26 / 0.77 | 0.187 / 0.36 / 0.82 |
| shape only | 0.256 / −0.25 / 0.45 | 0.197 / 0.23 / 0.74 |
| shape only + trend | 0.255 / −0.24 / 0.48 | 0.197 / 0.24 / 0.74 |

**Decision.**
- **Shape only: rejected.** Once loudness is measured from the right reference
  (Fix 1), it is useful information; throwing it away hurts.
- **Trend features: rejected.** The dev gain (MAE −0.003) is noise with 6
  bearings, and test rank went slightly down. Extra features that do not
  clearly help only add complexity.

Both stay available in `config.py`, switched off.

### Fix 3 — Train on XJTU + PHM Learning together (kept as a protocol, not default)

**Idea.** The model had never seen a quiet, PHM-style failure. Add the 6 PHM
Learning bearings to the 15 XJTU bearings (21 training bearings), test on the
11 Full_Test bearings. New protocol: `XJTU_PLUS_PHM_TO_FULL_TEST`.

**Note:** this changes the question. It is no longer "a machine it has never
seen", so it is reported as its own protocol, not as an XJTU → PHM result.

The loudness reference was chosen by cross validation over the 21 training
bearings: START (MAE 0.205) beat QUIETEST (0.218). On the 11 test bearings:

| Setup (random forest, same 11 test bearings) | MAE / R2 / rank |
| --- | --- |
| PHM → PHM | 0.198 / 0.22 / 0.83 |
| **Fix 3: XJTU + PHM** | 0.180 / 0.37 / 0.65 |
| XJTU → PHM with Fix 1 | 0.187 / 0.35 / 0.84 |
| XJTU → PHM with Fix 1 + Fix 4 | 0.172 / 0.42 / 0.85 |

**Decision.** Lower error than PHM alone, but rank drops to 0.65, and it does
not beat XJTU → PHM with Fixes 1 and 4 — which never saw a PHM bearing. Kept as
an extra protocol for comparison; not the default.

### Fix 4 — Do not predict degradation before degradation starts (kept as a "gate", XJTU → PHM only)

**Idea (two stages).**
1. Detect the moment degradation starts, the "first predicting time" (FPT):
   the fast EMA of RMS growth or kurtosis stays more than k standard deviations
   above this bearing's own first 10 minutes, for 2 minutes. Uses only the past.
2. Before the FPT: do not guess. After it: predict remaining life.

Two versions were tested:
- **full two-stage**: the second model is trained only on the degradation part
  of the training bearings;
- **gate only**: keep the normal model, but before the FPT never let the
  prediction drop below the "floor" — the median RUL the training bearings had
  at their FPT (0.72 for XJTU).

| | dev MAE / R2 / rank | test MAE / R2 / rank |
| --- | --- | --- |
| Fix 1 (current) | 0.212 / 0.24 / 0.76 | 0.187 / 0.35 / 0.84 |
| full two-stage, k = 3 | 0.221 / 0.02 / 0.81 | 0.181 / 0.37 / 0.84 |
| full two-stage, k = 5 | 0.213 / 0.26 / 0.76 | 0.184 / 0.38 / 0.80 |
| full two-stage, k = 8 | 0.217 / 0.23 / 0.84 | 0.193 / 0.35 / 0.80 |
| **gate only, k = 5** | **0.197 / 0.33 / 0.78** | **0.172 / 0.42 / 0.85** |

Two surprises:
- **The detector fires early**: at 3–50% of life, usually 10–20%, not in the
  last 10% as expected. A 10 minute "normal" window is short, so small changes
  already look abnormal.
- **The full two-stage version did not help**: training only on degradation
  data gave about the same numbers as Fix 1.

**Why "gate only" works.** It blocks early false alarms. Before any
degradation is detected, the model is not allowed to say "70% of life is gone"
— which is exactly the Bearing1_5 mistake from Step 1. It improved all three
dev metrics clearly, and the test bearings agree.

**Side effect check.** Gate forced on for PHM → PHM: validation MAE 0.199 vs
0.198 and a dev gain too small to count (the same rule that rejected the trend
features). **Decision:** gate on for XJTU → PHM only
(`RUL_FPT_GATE_BY_PROTOCOL`); the floor is stored in the model file.

### Final result

`python -m tools.rul compare --protocol XJTU_TO_PHM`, all 17 PHM bearings:

| Model | Start: MAE / R2 / rank | Now: MAE / R2 / rank |
| --- | --- | --- |
| Always guess 0.5 | 0.250 / 0.00 / — | 0.250 / 0.00 / — |
| Linear regression | 0.311 / −0.83 / 0.74 | 0.219 / −0.02 / 0.90 |
| **Random forest** | 0.216 / 0.15 / 0.53 | **0.181 / 0.39 / 0.82** |
| Gradient boosting | 0.268 / −0.35 / 0.65 | 0.194 / 0.25 / 0.80 |

- XJTU → PHM now **beats training on PHM itself** (0.198 / 0.22 / 0.83), even
  though the model never saw a PHM bearing.
- **PHM → PHM is exactly unchanged** (every new setting is off for it).
- Numbers from the real code match the experiment script exactly.

**Still weak** (random forest): Bearing2_7 (a 38 minute life, R2 −0.24),
Bearing1_1 (rank 0.19), Bearing2_3 (rank 0.32). Bearings whose only warning is
in the last 10% of life stay hard for any vibration-only model.

### The three weak bearings: why, and five more ideas (none adopted)

**Why they are weak** (averages over each tenth of life, current model):

- **Bearing2_7** lives only 38 minutes and its vibration barely changes until
  the last 4 minutes. The 15 minute smoothing from Step 1 is 40% of its whole
  life, so the prediction cannot fall fast enough (still 0.52 at failure).
- **Bearing1_1** wears in a textbook way (steady rise over 7.8 h), but by 20% of
  life the model already predicts 0.22. Cause, found by experiment B3 below:
  its kurtosis rises a little (1.19×) while loudness is still flat, and the
  model reads that as damage.
- **Bearing2_3** has strong kurtosis spikes at 10% and 40% of life (up to 3.2×)
  and slightly raised loudness right at the start (+0.20), then stays flat
  until failure. The model predicts ~0.25 from 10% of life onwards.

**Can we delete them?** No. They are real, difficult bearings, not broken
data. Removing the test bearings a model fails on makes the number better
without making the model better (cherry-picking), and a factory cannot choose
which bearings fail in an unusual way. A bearing may only be excluded for a
data fault (dead sensor, corrupted file), by a rule fixed before looking at
results.

**Five ideas tested** on top of the final system (dev 0.197 / 0.33 / 0.78,
test 0.172 / 0.42 / 0.85):

| Idea | dev MAE / R2 / rank | test MAE / R2 / rank | Target bearing |
| --- | --- | --- | --- |
| A1: smooth 15 min before degradation is detected, 5 min after | 0.210 / 0.15 / 0.66 | 0.181 / 0.36 / 0.75 | 2_7: MAE 0.285 → 0.217 |
| A1: same, 2 min after | 0.214 / 0.08 / 0.59 | 0.184 / 0.33 / 0.73 | 2_7: 0.285 → 0.178 |
| A2: smooth over 10% of the time lived so far | 0.223 / 0.07 / 0.61 | 0.179 / 0.38 / 0.78 | 2_7: 0.285 → 0.153 |
| A2: same, 25% | 0.219 / 0.16 / 0.72 | 0.175 / 0.41 / 0.81 | 2_7: 0.285 → 0.198 |
| B1: degradation detector listens to RMS growth only | 0.184 / 0.35 / 0.93 | 0.205 / 0.14 / 0.86 | 2_3: unchanged |
| B2: detector needs RMS growth AND kurtosis | 0.194 / 0.41 / 0.89 | 0.185 / 0.36 / 0.85 | 2_3: unchanged |
| B3: kurtosis only counts once RMS has grown 0.2 | 0.177 / 0.45 / 0.92 | 0.179 / 0.37 / 0.82 | 2_3: unchanged; 1_1: MAE 0.262 → 0.165 |
| B3: same, 0.5 | 0.215 / 0.09 / 0.86 | 0.183 / 0.34 / 0.73 | 2_3: slightly better |

**What the numbers say.**
- **A (less smoothing) fixes Bearing2_7 but costs everyone else.** Every A
  variant made Bearing2_7 much better (R2 −0.24 → up to 0.64) and made the
  overall result worse on dev and on test: the noise that Step 1 removed comes
  back for the other 16 bearings. One setting cannot suit a 38 minute life and
  a 7 hour life. **Not adopted.**
- **B1 and B2 help dev and hurt test.** With a stricter detector the gate stays
  closed too long on some bearings (two are never detected at all). **Not
  adopted** — again the dev bearings alone would have misled.
- **B3 (0.2) is the most interesting.** It fixes Bearing1_1 (rank 0.19 → 0.90),
  which shows that early kurtosis without loudness is what fooled the model
  there. But Bearing1_1 is a DEV bearing, most of the dev gain comes from it,
  and the 11 test bearings get slightly worse (MAE 0.172 → 0.179). Keeping it
  would be fitting the dev set. **Not adopted**; the option stays in
  `config.py` (`RUL_KURTOSIS_NEEDS_GROWTH`, 0 = off) for when more bearings are
  available to confirm it.
- **Nothing fixed Bearing2_3.** Its loudness is raised at the very start, so
  even "spikes confirmed by loudness" is satisfied early. Its vibration really
  does stay flat from 20% to 80% of life: there is nothing in the signal for a
  model to follow. This is a limit of the data.

**Lesson.** A fix aimed at one bearing is judged on all of them. Three ideas
here would each have "fixed" their target and made the system worse.

### What changed in the code

| File | Change |
| --- | --- |
| `predictive/rul_model.py` | `QUIETEST` / `BOTH` loudness reference in `rul_features`; `detect_fpt`; `postprocess_prediction`; FPT floor in `fit` / `predict_run`; model files store `loudness_reference` and `fpt_floor` |
| `predictive/dataset_jobs.py` | picks the loudness reference and gate per protocol; new protocol `XJTU_PLUS_PHM_TO_FULL_TEST` |
| `config.py` | the settings in the table in section 6, with comments |
| `results/experiments/` | the experiment script and every comparison output |
| `results/baseline_2026-09-28/README.md` | the experiment log: one row per idea |

### Lessons worth remembering

1. **Look at the data, not only the average.** Per-bearing tables found the
   run-in problem; the average never would have.
2. **Write the prediction down first.** Fix 1 was predicted to help Bearing2_5,
   1_5 and 1_7 — it did, which is strong evidence the diagnosis was right.
3. **Distrust a metric that suddenly becomes perfect.** The monotone trick made
   every model's rank ~1.
4. **Check side effects on everything else.** Two good ideas (smoothing, Fix 1)
   hurt PHM → PHM and had to be limited to where they help.
5. **Small dev sets mislead.** Always confirm on data you did not choose with.
6. **Simple beats clever when the data is small.** "Gate only" beat the full
   two-stage model.
