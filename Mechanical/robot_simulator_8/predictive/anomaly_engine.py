"""
Shared anomaly engine for real machine data (Mechanical-datasets and partner data).

Method, per operating setup (machine + condition):

    healthy windows (training block)
        |-- fit part (70%) --> RobustScaler + IsolationForest
        |                       + per feature median / MAD (for explanations)
        '-- calibration part (30%, later in time, never seen by the forest)
                            --> threshold = 95th percentile of anomaly score

    new window --> score = -IsolationForest.score_samples
               --> anomalous if score > threshold
               --> explanation = features with the largest robust z = (x - median) / (1.4826 MAD)

Why this shape:
  * Healthy behaviour differs between machines and speeds/loads, so one model
    per setup. A pooled model makes "healthy" broader than the faults (this
    detected nothing in Phase 6 testing).
  * The threshold comes from held-out healthy data, because scores on the
    training windows themselves are optimistic.
  * RobustScaler and MAD keep a few noisy healthy windows from stretching the
    notion of normal.
  * Temperature is checked separately (thermal features). A cooling fault moves
    a handful of slow features moderately, which a forest over a hundred
    vibration features averages away, so those features are compared directly
    against their healthy level: ratio = feature / healthy median, flagged at
    config.THERMAL_RATIO_WARN.

No Qt, no PyBullet. scikit-learn and joblib only.
"""

import datetime
import os

import joblib
import numpy as np

import config

MAD_TO_STD = 1.4826


def persistent_alarms(flags, windows=None):
    """
    flags: per window booleans in time order, one recording. A window is an
    alarm only if it and the (windows - 1) before it are all flagged, so the
    first (windows - 1) windows of a recording can never be alarms.
    """
    windows = config.ANOMALY_PERSISTENCE_WINDOWS if windows is None else windows
    flags = np.asarray(flags, dtype=bool)
    out = flags.copy()
    for shift in range(1, max(1, int(windows))):
        earlier = np.zeros_like(flags)
        earlier[shift:] = flags[:-shift]
        out &= earlier
    return out


def _blocks(x):
    """One array, or a list of arrays (one per recording, so persistence does not cross recordings)."""
    return [np.asarray(b, dtype=np.float64) for b in x] if isinstance(x, (list, tuple)) else [np.asarray(x, dtype=np.float64)]


class AnomalyModel(object):

    THERMAL_MARKERS = ("_rise_over_ambient", "_rise_per_watt", "_rise_since_start")

    def __init__(self, setup, feature_names, threshold_percentile=None, n_estimators=300,
                 random_state=None):
        self.setup = setup
        self.feature_names = list(feature_names)
        self.threshold_percentile = float(threshold_percentile or config.PARTNER_ANOMALY_THRESHOLD_PERCENTILE)
        self.n_estimators = int(n_estimators)
        self.random_state = config.RUL_RANDOM_STATE if random_state is None else random_state
        self.scaler = None
        self.forest = None
        self.median = None
        self.mad = None
        self.threshold = None
        self.thermal_columns = [i for i, name in enumerate(self.feature_names)
                                if any(name.endswith(m) for m in self.THERMAL_MARKERS)]
        self.thermal_reference = None
        self.feature_threshold = None     # None = single feature check off
        self.training = {}

    # -- training -----------------------------------------------------------

    def fit(self, healthy, calibration_fraction=None):
        """
        healthy: one (windows, features) array in time order, or a LIST of such
        arrays, one per recording/stream. Every stream gives its later part to
        calibration, so the threshold sees all healthy recordings, not just the
        last one.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        parts = healthy if isinstance(healthy, (list, tuple)) else [healthy]
        fraction = config.PARTNER_ANOMALY_CALIBRATION_FRACTION if calibration_fraction is None else calibration_fraction
        fit_parts, calibration_parts = [], []
        for part in parts:
            x = np.asarray(part, dtype=np.float64)
            x = x[np.all(np.isfinite(x), axis=1)]
            cut = int(round(x.shape[0] * (1.0 - fraction)))
            if x.shape[0] >= 4:
                fit_parts.append(x[:cut])
                calibration_parts.append(x[cut:])
        fit_part = np.vstack(fit_parts) if fit_parts else np.zeros((0, len(self.feature_names)))
        calibration = np.vstack(calibration_parts) if calibration_parts else fit_part[:0]
        if fit_part.shape[0] < 10 or calibration.shape[0] < 3:
            raise ValueError("setup %s: need at least ~15 healthy windows, have %d"
                             % (self.setup, fit_part.shape[0] + calibration.shape[0]))

        self.scaler = RobustScaler().fit(fit_part)
        self.forest = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state)
        self.forest.fit(self.scaler.transform(fit_part))
        self.median = np.median(fit_part, axis=0)
        mad = np.median(np.abs(fit_part - self.median), axis=0) * MAD_TO_STD
        # Features that are almost constant when healthy (a dominant frequency
        # locked to shaft speed) have MAD near zero, which would turn any change
        # into z in the thousands. The scale is floored by half the standard
        # deviation and by 5% of the typical magnitude.
        std = np.std(fit_part, axis=0)
        floor = np.maximum(0.5 * std, 0.05 * np.abs(self.median)) + 1e-9
        self.mad = np.maximum(mad, floor)
        self.threshold = float(np.percentile(self.raw_scores(calibration), self.threshold_percentile))
        self.feature_threshold = None
        if config.ANOMALY_FEATURE_CHECK:
            largest = np.max(np.abs((calibration - self.median) / self.mad), axis=1)
            self.feature_threshold = max(float(np.percentile(largest, config.ANOMALY_FEATURE_Z_PERCENTILE)),
                                         config.ANOMALY_FEATURE_Z_MIN)
        if self.thermal_columns:
            reference = np.median(np.vstack([fit_part, calibration])[:, self.thermal_columns], axis=0)
            # Only usable where the healthy level is clearly positive (a warm
            # motor). A near zero reference would make the ratio meaningless.
            self.thermal_reference = np.where(np.abs(reference) > 1e-3, reference, np.nan)
        self.training = {"fit_windows": int(fit_part.shape[0]),
                         "calibration_windows": int(calibration.shape[0]),
                         "created": datetime.datetime.now().isoformat()}
        return self

    # -- scoring ------------------------------------------------------------

    def raw_scores(self, x):
        return -self.forest.score_samples(self.scaler.transform(np.asarray(x, dtype=np.float64)))

    def thermal_ratios(self, x):
        """Largest ratio of a thermal feature to its healthy median (NaN if not applicable)."""
        values = np.asarray(x, dtype=np.float64)
        if not self.thermal_columns or self.thermal_reference is None:
            return np.full(values.shape[0], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = values[:, self.thermal_columns] / self.thermal_reference
        return np.nanmax(ratios, axis=1) if ratios.size else np.full(values.shape[0], np.nan)

    def score(self, x, top=3):
        """
        Per window dict: score, threshold ratio, top deviating features, and
        two flags. "anomalous": this window alone looks abnormal. "alarm": it
        and the windows just before it do (ANOMALY_PERSISTENCE_WINDOWS); x must
        be one recording in time order for that to mean something.
        """
        x = np.asarray(x, dtype=np.float64)
        scores = self.raw_scores(x)
        z = (x - self.median) / self.mad
        thermal = self.thermal_ratios(x)
        feature_threshold = getattr(self, "feature_threshold", None)   # absent in older model files
        results = []
        for i in range(x.shape[0]):
            order = np.argsort(-np.abs(z[i]))[:top]
            hot = bool(np.isfinite(thermal[i]) and thermal[i] >= config.THERMAL_RATIO_WARN)
            far = bool(feature_threshold is not None and abs(z[i, order[0]]) > feature_threshold)
            results.append({
                "score": float(scores[i]),
                "ratio": float(scores[i] / self.threshold) if self.threshold > 0 else 0.0,
                "anomalous": bool(scores[i] > self.threshold) or hot or far,
                "feature_alarm": far,
                "thermal_ratio": float(thermal[i]) if np.isfinite(thermal[i]) else None,
                "thermal_alarm": hot,
                "top_features": [(self.feature_names[k], float(z[i, k])) for k in order],
            })
        for result, alarm in zip(results, persistent_alarms([r["anomalous"] for r in results])):
            result["alarm"] = bool(alarm)
        return results

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, healthy_test, faulty_by_label):
        """
        False alarm rate on unseen healthy windows; detection rate and ROC AUC
        per fault label. Rates count ALARMS (with persistence); the per window
        rates without persistence are reported as window_*. Each input is one
        array or a list of arrays, one per recording.
        """
        from sklearn.metrics import roc_auc_score

        healthy = [r for block in _blocks(healthy_test) for r in self.score(block)]
        healthy_scores = np.array([h["score"] for h in healthy])
        report = {"setup": self.setup, "threshold": self.threshold,
                  "healthy_test_windows": int(healthy_scores.shape[0]),
                  "false_alarm_rate": float(np.mean([h["alarm"] for h in healthy])),
                  "window_false_alarm_rate": float(np.mean([h["anomalous"] for h in healthy])),
                  "persistence_windows": int(config.ANOMALY_PERSISTENCE_WINDOWS),
                  "thermal_check": bool(self.thermal_columns and self.thermal_reference is not None),
                  "labels": {}}
        for label, x in sorted(faulty_by_label.items()):
            blocks = [b for b in _blocks(x) if b.shape[0]]
            if not blocks:
                continue
            results = [r for block in blocks for r in self.score(block)]
            scores = np.array([r["score"] for r in results])
            y = np.r_[np.zeros(healthy_scores.shape[0]), np.ones(scores.shape[0])]
            report["labels"][label] = {
                "windows": int(scores.shape[0]),
                "detection_rate": float(np.mean([r["alarm"] for r in results])),
                "window_detection_rate": float(np.mean([r["anomalous"] for r in results])),
                "thermal_alarm_rate": float(np.mean([bool(r["thermal_alarm"]) for r in results])),
                "roc_auc": float(roc_auc_score(y, np.r_[healthy_scores, scores])),
                "median_ratio": float(np.median(scores / self.threshold)),
                "top_features": _top_features(self, np.vstack(blocks)),
            }
        return report


def _top_features(model, x, top=5):
    z = np.median(np.abs((x - model.median) / model.mad), axis=0)
    order = np.argsort(-z)[:top]
    return [(model.feature_names[k], float(z[k])) for k in order]


def time_blocks(n, test_fraction=None, gap=None):
    """Contiguous train block, gap, later test block. Never shuffled."""
    test_fraction = config.MECH_TEST_FRACTION if test_fraction is None else test_fraction
    gap = config.MECH_GAP_SEGMENTS if gap is None else gap
    test = max(1, int(round(n * test_fraction)))
    train_end = max(0, n - test - gap)
    return np.arange(0, train_end), np.arange(n - test, n)


def save_models(path, models, metadata):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump({"kind": "anomaly_models", "version": 1, "models": models,
                 "metadata": dict(metadata)}, path, compress=3)
    return path


def load_models(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("kind") != "anomaly_models":
        raise ValueError("%s is not an anomaly model bundle" % path)
    return bundle
