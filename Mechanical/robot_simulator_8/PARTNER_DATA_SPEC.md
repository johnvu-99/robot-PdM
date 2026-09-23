# Data collection guide for the robot condition monitoring project

This guide says how to record and hand over the robot data so it can be loaded
without manual cleaning. It covers joint telemetry, accelerometer (vibration),
motor current and power.

A complete example, with the exact file layout, can be generated at any time:

```
python -m tools.partner_data template --out data/external/partner_example --mode fault_labels
```

The example signals are synthetic. Use it for the format only.

## 1. Folder layout

```
partner/
  dataset.json       what was measured (fill in once)
  recordings.csv     one row per data file (add a row for every recording)
  raw/
    run_01.csv
    run_02.csv
    ...
```

## 2. Data files

One file per recording session. CSV is preferred; Excel (`.xlsx`) and MATLAB
(`.mat`) also work.

```
time_s,joint,position_rad,target_rad,velocity_rad_s,torque_nm,motor_current_a,power_w,accel_g
0.0000,1,-0.04677,-0.05103,-1.1929,30.346,14.608,-36.199,0.00018
0.0010,1,-0.04694,-0.05222,-1.1925,30.331,14.622,-36.171,0.00213
...
0.0000,2,...
```

- `time_s`: seconds, strictly increasing within each joint.
- `joint`: joint number. Several joints can share one file (long format).
- One row per sample. Use the same sample rate for all channels in one file.
- If the accelerometer runs much faster than the controller (for example
  10 kHz vs 1 kHz), record it in a separate dataset folder with its own
  `dataset.json`, rather than repeating or dropping samples.
- Decimal point `.`, no thousands separators, SI units where possible.
- Leave a value empty if it is missing. Do not write 0 for a missing value.

## 3. dataset.json

```json
{
  "name": "lab_robot_2026_09",
  "system": "Robot model, which joints, gearbox type, accelerometer model and mounting position",
  "format": "csv",
  "sample_rate_hz": 1000,
  "time_column": "time_s",
  "joint_column": "joint",
  "window_seconds": 4.0,
  "hop_seconds": 1.0,
  "label_mode": "fault_labels",
  "healthy_label": "HEALTHY",
  "temperature_window_seconds": 120.0,
  "warmup_skip_seconds": 600,
  "channels": [
    {"name": "position", "column": "position_rad",    "type": "position",        "unit": "rad"},
    {"name": "target",   "column": "target_rad",      "type": "target_position", "unit": "rad"},
    {"name": "velocity", "column": "velocity_rad_s",  "type": "velocity",        "unit": "rad/s"},
    {"name": "torque",   "column": "torque_nm",       "type": "torque",          "unit": "N m"},
    {"name": "current",  "column": "motor_current_a", "type": "current",         "unit": "A"},
    {"name": "power",    "column": "power_w",         "type": "power",           "unit": "W"},
    {"name": "accel",    "column": "accel_g",         "type": "vibration",       "unit": "g"},
    {"name": "motor_temp","column": "motor_temp_c",    "type": "temperature",     "unit": "C"},
    {"name": "ambient",  "column": "ambient_c",        "type": "ambient_temperature", "unit": "C"}
  ]
}
```

- Channel `type` must be one of: position, target_position, velocity,
  acceleration, torque, current, voltage, power, vibration, temperature,
  ambient_temperature, force, other.
- List only sensors that were really recorded. Missing sensors are fine.
- `scale` (optional, per channel) multiplies raw values, e.g. `"scale": 0.001`
  for mA to A.
- `window_seconds` must cover at least one full repetition of the robot's
  motion program (see section 5).
- `label_mode`: `fault_labels` (known healthy and faulty recordings),
  `run_to_failure` (the same joint recorded until it fails), or `unlabeled`.

## 4. recordings.csv

```
file,run_id,asset_id,condition,label,fault_joint,start_time_s,failure_time_s,notes
raw/run_01.csv,run_01,robot_A,load_5kg,HEALTHY,,,,new gearbox
raw/run_05.csv,run_05,robot_A,load_5kg,FRICTION,2,,,grease removed from joint 2
```

| Column | Meaning |
| --- | --- |
| file | path relative to the dataset folder |
| run_id | one physical session. Files that belong to the same session share it. It decides train/test splitting, so never reuse a run_id for a different session |
| asset_id | which robot |
| condition | payload, speed or program. Keep it identical for recordings that should be compared |
| label | HEALTHY or the fault name. Keep names consistent (always FRICTION, never Friction or friction_high) |
| fault_joint | the joint the fault is on. Other joints in the file are treated as healthy. Empty means all joints |
| start_time_s | run_to_failure only: when this file starts, in seconds since the run began |
| failure_time_s | run_to_failure only: when the joint failed, in seconds since the run began |
| notes | anything useful (what was changed, who recorded it) |

## 5. Temperature, if you record it

Temperature behaves differently from vibration, so it needs three extra things.

1. **Record ambient temperature too**, as a channel of type
   `ambient_temperature`. A motor at 45 C means something different in a 20 C
   room than in a 35 C room. Without it, a hot day looks like a fault.
2. **Let the robot warm up before the useful part of the recording.** Motors
   take 10 to 30 minutes to settle. Set `warmup_skip_seconds` to the time the
   machine needs (600 means the first 10 minutes are ignored), and record long
   enough that useful data remains afterwards.
3. **Record healthy data at the same duty cycle** as the faulty data. Hotter is
   normal when the robot works harder; the analysis divides the temperature rise
   by mechanical power to compensate, which only works if power is recorded too.

The temperature sensor may be slow (1 reading per second, or even slower). Write
the last known value on every row; repeated values are expected and handled.

Features computed per window: level, maximum, rate of change in C per minute,
rise since the start of the recording, rise over ambient, and rise per watt of
mechanical power. They use a longer window (`temperature_window_seconds`,
120 s by default) than the mechanical channels.

If your baseline data is recorded while the machine is still warming up, the
validator warns:
`motor_temp still rising 2.3 C/min after warmup_skip_seconds`.

## 6. How to record, so the data is usable

1. **Repeat one motion program.** Run the same trajectory, payload and speed
   for healthy and faulty recordings. Record at least 30 s per file, and make
   `window_seconds` at least one program cycle long.
2. **Record several separate healthy sessions** (at least 4, on different
   days if possible). Healthy variation between sessions is what the anomaly
   detector learns. One long healthy file is much weaker.
3. **Record every fault type in at least 2 separate sessions.** A fault seen in
   only one session cannot be tested on data the model has not seen.
4. **Keep conditions matched.** If payload or speed change, record healthy data
   for that condition too, and give it a different `condition` value.
5. **Write down exactly what the fault is** in `notes` (how much grease was
   removed, which bearing, how much load was added).
7. **Check the data before sending**:

   ```
   python -m tools.partner_data validate --root path/to/partner
   ```

   Fix every ERROR. WARNINGs explain what will make results weaker.
