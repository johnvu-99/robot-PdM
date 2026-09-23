"""
Rolling window features.

Pure numpy, no Qt, no PyBullet. Works on any sample rate, so the same code
turns live 240 Hz telemetry and 100 Hz dataset CSVs into identical features.

For every signal and window: mean, std, var, min, max, median, rms, p2p, p95.
Trends (least squares slope, units per second) for torque, tracking error and
power. Engineered aliases use the detection window, for example
torque_RMS = torque_<DETECTION_WINDOW_S>s_rms.
"""

import numpy as np

import config
from monitoring.telemetry import CHANNEL_INDEX

# Feature signal name -> telemetry channel.
SIGNALS = (
    ("torque", "motor_torque"),
    ("velocity", "velocity"),
    ("acceleration", "acceleration"),
    ("tracking_error", "tracking_error"),
    ("power", "power"),
    ("force", "reaction_force_magnitude"),
)
SIGNAL_NAMES = tuple(name for name, _ in SIGNALS)
SIGNAL_CHANNELS = tuple(CHANNEL_INDEX[channel] for _, channel in SIGNALS)

STATISTICS = ("mean", "std", "var", "min", "max", "median", "rms", "p2p", "p95")
TREND_SIGNALS = ("torque", "tracking_error", "power")

ENGINEERED = (
    ("torque_RMS", "torque"),
    ("tracking_error_RMS", "tracking_error"),
    ("acceleration_RMS", "acceleration"),
    ("power_RMS", "power"),
    ("force_RMS", "force"),
)


def feature_names(windows=None):
    windows = windows or config.FEATURE_WINDOWS_S
    names = []
    for window in windows:
        for signal in SIGNAL_NAMES:
            for stat in STATISTICS:
                names.append("%s_%ds_%s" % (signal, window, stat))
        for signal in TREND_SIGNALS:
            names.append("%s_%ds_trend" % (signal, window))
    names.extend(alias for alias, _ in ENGINEERED)
    names.extend("%s_trend" % signal for signal in TREND_SIGNALS)
    return names


def signal_of(feature):
    """Which signal a feature is computed from (used for std floors)."""
    for signal in sorted(SIGNAL_NAMES, key=len, reverse=True):
        if feature.startswith(signal):
            return signal
    return None


class FeatureExtractor(object):

    def __init__(self, windows=None, detection_window=None):
        self.windows = tuple(windows or config.FEATURE_WINDOWS_S)
        self.detection_window = int(detection_window or config.DETECTION_WINDOW_S)
        self.names = feature_names(self.windows)
        self.index = dict((name, i) for i, name in enumerate(self.names))

    @property
    def longest_window(self):
        return max(self.windows)

    def compute(self, times, signals):
        """
        times: (N,) seconds, increasing. signals: (N, n_joints, len(SIGNALS)).
        Returns (n_joints, n_features) or None if the shortest window is not
        yet filled. Windows longer than the data are computed on what exists
        and reported through available_windows().
        """
        if times.shape[0] < 3:
            return None
        n = signals.shape[1]
        out = np.full((n, len(self.names)), np.nan)
        end = times[-1]
        column = 0
        engineered = {}
        trends = {}
        for window in self.windows:
            start = np.searchsorted(times, end - window + 1e-9)
            x = signals[start:]
            t = times[start:]
            if x.shape[0] < 3:
                column += len(SIGNAL_NAMES) * len(STATISTICS) + len(TREND_SIGNALS)
                continue
            mean = x.mean(axis=0)
            std = x.std(axis=0)
            minimum = x.min(axis=0)
            maximum = x.max(axis=0)
            rms = np.sqrt(np.mean(x * x, axis=0))
            median, p95 = np.percentile(x, (50.0, 95.0), axis=0)
            stats = (mean, std, std * std, minimum, maximum, median, rms,
                     maximum - minimum, p95)
            for s_index in range(len(SIGNAL_NAMES)):
                for stat in stats:
                    out[:, column] = stat[:, s_index]
                    column += 1

            tc = t - t.mean()
            denominator = float(np.sum(tc * tc))
            for signal in TREND_SIGNALS:
                s_index = SIGNAL_NAMES.index(signal)
                values = x[:, :, s_index]
                slope = (tc[:, None] * (values - values.mean(axis=0))).sum(axis=0) / max(denominator, 1e-12)
                out[:, column] = slope
                if window == self.detection_window:
                    trends[signal] = slope
                column += 1
            if window == self.detection_window:
                for alias, signal in ENGINEERED:
                    engineered[alias] = rms[:, SIGNAL_NAMES.index(signal)]

        for alias, _ in ENGINEERED:
            if alias in engineered:
                out[:, self.index[alias]] = engineered[alias]
        for signal in TREND_SIGNALS:
            if signal in trends:
                out[:, self.index["%s_trend" % signal]] = trends[signal]
        return out

    def select(self, features, names):
        return features[..., [self.index[name] for name in names]]


def signals_from_matrix(matrices):
    """(N, n_joints, telemetry channels) -> (N, n_joints, len(SIGNALS))."""
    return matrices[:, :, list(SIGNAL_CHANNELS)]
