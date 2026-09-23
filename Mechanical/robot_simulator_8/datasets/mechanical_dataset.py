"""
Mechanical-datasets (github.com/cathysiyu/Mechanical-datasets).

Role: FAULT DETECTION AND ANOMALY VALIDATION on real rotating machinery.

gearbox/gearset/*.csv, gearbox/bearingset/*.csv
    Southeast University Drivetrain Dynamics Simulator. 16 header lines, then
    8 channels: motor vibration, planetary gearbox vibration x/y/z, motor
    torque, parallel gearbox vibration x/y/z. Tab or comma separated.
    Conditions 20_0 and 30_2 (speed - load setting).
dataset/*.mat
    Case Western Reserve University bearing data, fan end accelerometer.

This is gearbox and bearing test rig data, not KUKA joint telemetry.
"""

import glob
import os
import re

import numpy as np

import config
from datasets.base_adapter import DatasetAdapter, Recording
from datasets.signal_features import (
    parse_numeric_text, torque_feature_names, torque_features,
    vibration_feature_names, vibration_features,
)

SEU_LABELS = {
    "health": "HEALTHY",
    "chipped": "GEAR_CHIPPED_TOOTH",
    "miss": "GEAR_MISSING_TOOTH",
    "root": "GEAR_ROOT_CRACK",
    "surface": "GEAR_SURFACE_WEAR",
    "ball": "BEARING_BALL",
    "inner": "BEARING_INNER_RACE",
    "outer": "BEARING_OUTER_RACE",
    "comb": "BEARING_COMBINED_INNER_OUTER",
}
CWRU_LABELS = {"B": "BEARING_BALL", "IR": "BEARING_INNER_RACE", "OR": "BEARING_OUTER_RACE",
               "NORMAL": "HEALTHY"}


class MechanicalDatasetAdapter(DatasetAdapter):

    name = "MECHANICAL"
    role = "Fault detection and anomaly validation"
    system = "Drivetrain gearbox and bearing test rigs (SEU DDS, CWRU)"

    def __init__(self, root=None):
        DatasetAdapter.__init__(self, root or config.MECHANICAL_DATASET_DIR)

    # -- discovery ----------------------------------------------------------

    def discover(self):
        recordings = []
        for subset in ("gearset", "bearingset"):
            for path in sorted(glob.glob(os.path.join(self.root, "gearbox", subset, "*.csv"))):
                stem = os.path.splitext(os.path.basename(path))[0]
                match = re.match(r"([A-Za-z]+)_(\d+)_(\d+)$", stem)
                if not match:
                    continue
                label = SEU_LABELS.get(match.group(1).lower())
                if label is None:
                    continue
                condition = "SEU_%s-%s" % (match.group(2), match.group(3))
                recordings.append(Recording(
                    self.name, "SEU_%s_%s" % (subset, stem), "SEU_%s_%s" % (subset, stem),
                    "gearbox" if subset == "gearset" else "gearbox bearing",
                    condition, label, [path], config.SEU_SAMPLE_RATE_HZ, config.SEU_CHANNELS,
                    meta={"source": "SEU", "subset": subset}))
        for path in sorted(glob.glob(os.path.join(self.root, "dataset", "*.mat"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            match = re.match(r"(IR|OR|B)(\d{3})(?:@\d+)?_(\d)$", stem, re.IGNORECASE)
            if match:
                label = CWRU_LABELS[match.group(1).upper()]
                load = match.group(3)
                meta = {"source": "CWRU", "fault_diameter_in": int(match.group(2)) / 1000.0}
            elif stem.lower().startswith("normal"):
                label = "HEALTHY"
                load = stem.split("_")[-1]
                meta = {"source": "CWRU", "original_sample_rate_hz": config.CWRU_NORMAL_SAMPLE_RATE_HZ,
                        "resampled_to_hz": config.CWRU_SAMPLE_RATE_HZ}
            else:
                continue
            recordings.append(Recording(
                self.name, "CWRU_%s" % stem, "CWRU_%s" % stem, "rolling bearing",
                "CWRU_load%s" % load, label, [path], config.CWRU_SAMPLE_RATE_HZ,
                ("fan_end_vibration",), meta=meta))
        return recordings

    # -- loading and segmentation -------------------------------------------

    def load_signals(self, recording):
        """Returns (samples, channels) float array for one recording."""
        path = recording.files[0]
        if recording.meta.get("source") == "CWRU":
            return self._load_cwru(path)
        limit = config.SEU_SEGMENT_SAMPLES * config.SEU_MAX_SEGMENTS_PER_FILE
        return self._load_seu(path, limit)

    @staticmethod
    def _load_seu(path, max_rows):
        rows = []
        started = False
        with open(path, "r", errors="replace") as handle:
            for line in handle:
                if not started:
                    if line.split("\t")[0].split(",")[0].strip().lower() == "data":
                        started = True
                    continue
                rows.append(line)
                if len(rows) >= max_rows:
                    break
        if not started:
            raise ValueError("no 'Data' marker in %s" % path)
        return parse_numeric_text("".join(rows), len(config.SEU_CHANNELS))

    @staticmethod
    def _load_cwru(path):
        from scipy.io import loadmat
        data = loadmat(path)
        keys = [k for k in data if k.endswith("_FE_time")] or [k for k in data if k.endswith("_DE_time")]
        if not keys:
            raise ValueError("no accelerometer channel in %s" % path)
        signal = np.asarray(data[keys[0]], dtype=np.float64).reshape(-1)
        if os.path.basename(path).lower().startswith("normal"):
            from scipy.signal import decimate
            factor = int(round(config.CWRU_NORMAL_SAMPLE_RATE_HZ / config.CWRU_SAMPLE_RATE_HZ))
            if factor > 1:
                signal = decimate(signal, factor, ftype="fir", zero_phase=True)
        return signal.reshape(-1, 1)

    def segment(self, recording, signals):
        if recording.meta.get("source") == "CWRU":
            length, limit = config.CWRU_SEGMENT_SAMPLES, config.CWRU_MAX_SEGMENTS_PER_FILE
        else:
            length, limit = config.SEU_SEGMENT_SAMPLES, config.SEU_MAX_SEGMENTS_PER_FILE
        count = min(signals.shape[0] // length, limit)
        return signals[:count * length].reshape(count, length, signals.shape[1])

    # -- features -----------------------------------------------------------

    def feature_names(self, recording):
        if recording.meta.get("source") == "CWRU":
            return vibration_feature_names("vibration")
        names = []
        for channel in config.SEU_VIBRATION_CHANNELS + ("motor_vibration",):
            names.extend(vibration_feature_names(channel))
        names.extend(torque_feature_names("torque"))
        return tuple(names)

    def extract_features(self, recording):
        """(segments, features) with a cache keyed on the file signature."""
        cache = self.cache_path(recording, "segments_v2")
        signature = recording.signature()
        cached = self.load_cache(cache, signature)
        names = self.feature_names(recording)
        if cached is not None and tuple(cached["names"]) == names:
            return cached["features"], names
        segments = self.segment(recording, self.load_signals(recording))
        rate = recording.sample_rate
        rows = []
        for segment in segments:
            if recording.meta.get("source") == "CWRU":
                rows.append(vibration_features(segment[:, 0], rate))
                continue
            row = []
            for channel in config.SEU_VIBRATION_CHANNELS + ("motor_vibration",):
                row.extend(vibration_features(segment[:, config.SEU_CHANNELS.index(channel)], rate))
            row.extend(torque_features(segment[:, config.SEU_CHANNELS.index(config.SEU_TORQUE_CHANNEL)], rate))
            rows.append(row)
        features = np.array(rows, dtype=np.float64).reshape(len(rows), len(names))
        self.save_cache(cache, signature, features=features, names=np.array(names))
        return features, names
