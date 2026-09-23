"""
Feature windows from generated simulation CSV files (Phase 4 format).

Used to train the baseline and Isolation Forest offline from
data/simulation/normal. Same FeatureExtractor as the live path.
"""

import csv
import glob
import json
import os

import numpy as np

from monitoring.feature_extractor import SIGNAL_NAMES

_CSV_SIGNALS = {
    "torque": "motor_torque",
    "velocity": "velocity",
    "acceleration": "acceleration",
    "tracking_error": "tracking_error",
    "power": "power",
}


def load_run(path):
    """Returns (times (T,), signals (T, n, S), normal (T, n) bool, joint_names)."""
    by_time = {}
    joint_names = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            t = float(row["timestamp"])
            joint = int(row["joint_id"])
            joint_names[joint] = row["joint_name"]
            values = []
            for name in SIGNAL_NAMES:
                if name == "force":
                    fx = float(row["reaction_force_x"])
                    fy = float(row["reaction_force_y"])
                    fz = float(row["reaction_force_z"])
                    values.append((fx * fx + fy * fy + fz * fz) ** 0.5)
                else:
                    values.append(float(row[_CSV_SIGNALS[name]]))
            by_time.setdefault(t, {})[joint] = (values, row.get("fault_type", "NORMAL") == "NORMAL")
    joints = sorted(joint_names)
    times = np.array(sorted(by_time))
    signals = np.zeros((times.shape[0], len(joints), len(SIGNAL_NAMES)))
    normal = np.ones((times.shape[0], len(joints)), dtype=bool)
    for i, t in enumerate(times):
        sample = by_time[t]
        for j, joint in enumerate(joints):
            if joint in sample:
                signals[i, j] = sample[joint][0]
                normal[i, j] = sample[joint][1]
    return times, signals, normal, [joint_names[j] for j in joints]


def windows_from_run(path, extractor, mode, hop, skip_s):
    times, signals, normal, names = load_run(path)
    samples = []
    if times.shape[0] == 0:
        return samples, names
    end_time = times[0] + max(skip_s, extractor.longest_window)
    while end_time <= times[-1] + 1e-9:
        end = np.searchsorted(times, end_time, side="right")
        start = np.searchsorted(times, end_time - extractor.longest_window + 1e-9)
        if np.all(normal[start:end]):
            features = extractor.compute(times[start:end], signals[start:end])
            if features is not None:
                samples.append((mode, features))
        end_time += hop
    return samples, names


def normal_dataset_windows(root, extractor, hop, skip_s):
    """All windows from <root>/normal, with motion mode taken from manifests."""
    patterns = {}
    for manifest_path in glob.glob(os.path.join(root, "manifest_*.json")):
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        for run in manifest.get("runs", []):
            patterns[os.path.basename(run["file"])] = run.get("pattern", "ALL")
    samples = []
    names = None
    files = sorted(glob.glob(os.path.join(root, "normal", "*.csv")))
    for path in files:
        mode = patterns.get(os.path.basename(path), "ALL")
        run_samples, run_names = windows_from_run(path, extractor, mode, hop, skip_s)
        samples.extend(run_samples)
        names = names or run_names
    return samples, names, len(files)
