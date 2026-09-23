"""
Shared logic for run-to-failure bearing datasets (XJTU-SY, PHM 2012).

Each run is a sequence of short vibration snapshots. Time comes from the file
number and the snapshot interval, so a partially downloaded run still has
correct timing. Labels:

    life_fraction = t / t_failure       (0 at the start, 1 at the last snapshot)
    RUL_fraction  = 1 - life_fraction
"""

import numpy as np

from datasets.base_adapter import DatasetAdapter, file_number
from datasets.signal_features import vibration_feature_names, vibration_features


class RunToFailureAdapter(DatasetAdapter):

    snapshot_channels = ("horizontal", "vertical")

    def read_snapshot(self, path):
        """(samples, 2) horizontal and vertical acceleration."""
        raise NotImplementedError

    def feature_names(self):
        names = []
        for channel in self.snapshot_channels:
            names.extend(vibration_feature_names(channel))
        return tuple(names)

    def snapshot_times(self, recording):
        first = file_number(recording.files[0])
        return np.array([(file_number(f) - first) * recording.snapshot_interval
                         for f in recording.files], dtype=np.float64)

    def extract_run(self, recording, progress=None, cancel=None):
        """
        Returns dict(times, features, names, life_fraction, rul_fraction).
        Cached in data/processed/<dataset>/.
        """
        names = self.feature_names()
        cache = self.cache_path(recording, "run")
        signature = recording.signature()
        cached = self.load_cache(cache, signature)
        if cached is None or tuple(cached["names"]) != names:
            rows = []
            for index, path in enumerate(recording.files):
                snapshot = self.read_snapshot(path)
                row = []
                for c in range(len(self.snapshot_channels)):
                    row.extend(vibration_features(snapshot[:, c], recording.sample_rate))
                rows.append(row)
                if progress is not None and index % 50 == 0:
                    progress(index, len(recording.files))
                if cancel is not None and cancel():
                    raise KeyboardInterrupt("cancelled")
            features = np.array(rows, dtype=np.float64)
            self.save_cache(cache, signature, features=features, names=np.array(names))
        else:
            features = cached["features"]
        times = self.snapshot_times(recording)
        life = times / times[-1] if times[-1] > 0 else np.zeros_like(times)
        return {
            "run_id": recording.run_id, "dataset": self.name, "condition": recording.condition,
            "times": times, "features": features, "names": names,
            "life_fraction": life, "rul_fraction": 1.0 - life,
            "total_life_s": float(times[-1]),
        }
