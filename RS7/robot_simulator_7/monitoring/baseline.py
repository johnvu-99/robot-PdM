"""
Normal behaviour baseline.

Collects feature vectors from windows that are known to be normal (ground truth
label NORMAL, no mode transition) and summarises them per joint and per motion
mode: mean, std, 5th/50th/95th percentile. Feature distributions differ a lot
between holding a pose and a stress sweep, so each mode keeps its own baseline
with a pooled "ALL" fallback.
"""

import numpy as np

import config
from monitoring.feature_extractor import signal_of

ALL_MODES = "ALL"


class Baseline(object):

    def __init__(self, feature_names, joint_names):
        self.feature_names = list(feature_names)
        self.joint_names = list(joint_names)
        self.stats = {}          # mode -> dict of arrays (n_joints, n_features)
        self.window_counts = {}  # mode -> windows used

    @property
    def trained(self):
        return bool(self.stats)

    @classmethod
    def fit(cls, feature_names, joint_names, samples):
        """samples: list of (mode, features (n_joints, n_features))."""
        baseline = cls(feature_names, joint_names)
        groups = {}
        for mode, features in samples:
            groups.setdefault(mode, []).append(features)
            groups.setdefault(ALL_MODES, []).append(features)
        for mode, rows in groups.items():
            if len(rows) < config.BASELINE_MIN_WINDOWS:
                continue
            data = np.stack(rows)                     # (W, n_joints, n_features)
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            p5, p50, p95 = np.nanpercentile(data, (5.0, 50.0, 95.0), axis=0)
            baseline.stats[mode] = {
                "mean": mean, "std": std,
                "std_effective": baseline._floored_std(mean, std),
                "p5": p5, "p50": p50, "p95": p95,
            }
            baseline.window_counts[mode] = len(rows)
        return baseline

    def _floored_std(self, mean, std):
        floors = np.array([config.BASELINE_STD_ABSOLUTE_FLOOR.get(signal_of(name), 0.0)
                           for name in self.feature_names])
        return np.maximum(np.maximum(std, config.BASELINE_STD_RELATIVE_FLOOR * np.abs(mean)),
                          floors[None, :])

    def for_mode(self, mode):
        if mode in self.stats:
            return mode, self.stats[mode]
        if ALL_MODES in self.stats:
            return ALL_MODES, self.stats[ALL_MODES]
        return None, None

    def to_dict(self):
        return {
            "feature_names": self.feature_names,
            "joint_names": self.joint_names,
            "window_counts": dict(self.window_counts),
            "stats": dict((mode, dict((k, v.tolist()) for k, v in stats.items()))
                          for mode, stats in self.stats.items()),
        }

    @classmethod
    def from_dict(cls, data):
        baseline = cls(data["feature_names"], data["joint_names"])
        baseline.window_counts = dict(data.get("window_counts", {}))
        for mode, stats in data.get("stats", {}).items():
            baseline.stats[mode] = dict((k, np.array(v)) for k, v in stats.items())
        return baseline
