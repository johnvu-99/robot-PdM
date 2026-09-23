"""
Remaining useful life estimation for run-to-failure bearing data.

Target: RUL_fraction = 1 - t / t_failure, which is dimensionless and so
comparable between XJTU-SY (1 snapshot per minute) and PHM 2012 (every 10 s).

Features are causal and dimensionless: each run is normalised by its own first
RUL_BASELINE_SNAPSHOTS snapshots (information available at inference time),
then smoothed with fast and slow EMAs and a running maximum. No future sample
of a run ever influences a feature at an earlier time.

Leakage rules enforced here:
  * splits are by complete run (bearing), never by snapshot
  * the scaler lives inside the sklearn Pipeline, fitted on training runs only
  * train and validation run IDs are asserted disjoint
"""

import datetime
import os

import joblib
import numpy as np

import config

BASE_QUANTITIES = ("rms", "p2p", "kurtosis", "crest_factor", "spectral_energy",
                   "spectral_centroid", "band_energy_0", "band_energy_1", "band_energy_2",
                   "band_energy_3")
LOG_RATIO = ("rms", "p2p", "spectral_energy")
SMOOTHED = ("rms", "kurtosis")


def _ema(values, alpha):
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _ema_time(values, times, tau_s):
    """
    EMA with a time constant in seconds. Each step uses its own gap, so a
    dataset recorded every 10 s and one recorded every 60 s are smoothed over
    the same amount of real time.
    """
    out = np.empty_like(values)
    acc = values[0]
    previous = times[0]
    for i, v in enumerate(values):
        dt = max(float(times[i] - previous), 0.0)
        previous = times[i]
        alpha = 1.0 - np.exp(-dt / max(tau_s, 1e-6))
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _baseline_count(run):
    """How many snapshots make up the healthy reference at the start of a run."""
    total = run["features"].shape[0]
    if config.RUL_TIME_AWARE_SMOOTHING:
        count = int(np.sum(run["times"] - run["times"][0] <= config.RUL_BASELINE_SECONDS))
    else:
        count = config.RUL_BASELINE_SNAPSHOTS
    return max(1, min(count, max(1, total // 4)))


def rul_features(run):
    """Returns (matrix (T, F), names, health_indicator (T,))."""
    names_in = list(run["names"])
    features = run["features"]
    base_n = _baseline_count(run)
    times = np.asarray(run["times"], dtype=np.float64)
    columns = []
    names = []
    degradation = []
    for channel in ("horizontal", "vertical"):
        for quantity in BASE_QUANTITIES:
            x = features[:, names_in.index("%s_%s" % (channel, quantity))]
            base = np.mean(x[:base_n])
            if quantity in LOG_RATIO:
                value = np.log(np.maximum(x, 1e-12) / max(base, 1e-12))
            elif quantity.startswith("band_energy"):
                value = x - base
            else:
                value = x / base if abs(base) > 1e-12 else x
            columns.append(value)
            names.append("%s_%s_rel" % (channel, quantity))
            if quantity in SMOOTHED:
                if config.RUL_TIME_AWARE_SMOOTHING:
                    fast = _ema_time(value, times, config.RUL_EMA_FAST_TAU_S)
                    slow = _ema_time(value, times, config.RUL_EMA_SLOW_TAU_S)
                else:
                    fast = _ema(value, config.RUL_EMA_FAST)
                    slow = _ema(value, config.RUL_EMA_SLOW)
                columns.extend([fast, slow, np.maximum.accumulate(slow)])
                names.extend(["%s_%s_ema_fast" % (channel, quantity),
                              "%s_%s_ema_slow" % (channel, quantity),
                              "%s_%s_runmax" % (channel, quantity)])
                if quantity == "rms":
                    degradation.append(np.maximum.accumulate(slow))
    matrix = np.column_stack(columns)
    level = np.maximum(0.0, np.mean(degradation, axis=0))
    health = np.exp(-config.HEALTH_INDICATOR_SENSITIVITY * level)
    return matrix, names, health


def _make_estimator(kind):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "LINEAR_REGRESSION":
        model = LinearRegression()
    elif kind == "RANDOM_FOREST":
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=1,
                                      random_state=config.RUL_RANDOM_STATE)
    elif kind == "GRADIENT_BOOSTING":
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=config.RUL_RANDOM_STATE)
    else:
        raise ValueError("unknown RUL model %s" % kind)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _stack(runs):
    xs, ys, groups = [], [], []
    names = None
    for run in runs:
        matrix, names, _ = rul_features(run)
        xs.append(matrix)
        ys.append(run["rul_fraction"])
        groups.extend([run["run_id"]] * matrix.shape[0])
    return np.vstack(xs), np.concatenate(ys), np.array(groups), names


def assert_disjoint(train_runs, test_runs):
    overlap = set(r["run_id"] for r in train_runs) & set(r["run_id"] for r in test_runs)
    if overlap:
        raise ValueError("data leakage: runs in both train and test: %s" % sorted(overlap))


class RULModel(object):

    def __init__(self, kind=None):
        self.kind = kind or config.RUL_DEFAULT_MODEL
        self.pipeline = None
        self.feature_names = None
        self.training_runs = []
        self.training_dataset = ""
        self.created = ""

    def fit(self, runs):
        x, y, _, names = _stack(runs)
        self.pipeline = _make_estimator(self.kind)
        self.pipeline.fit(x, y)
        self.feature_names = names
        self.training_runs = [r["run_id"] for r in runs]
        self.training_dataset = ",".join(sorted(set(r["dataset"] for r in runs)))
        self.created = datetime.datetime.now().isoformat()
        return self

    def predict_run(self, run):
        matrix, names, health = rul_features(run)
        if names != self.feature_names:
            raise ValueError("feature layout differs from the trained model")
        prediction = np.clip(self.pipeline.predict(matrix), 0.0, 1.0)
        return prediction, health

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump({"version": 1, "kind": self.kind, "pipeline": self.pipeline,
                     "feature_names": self.feature_names, "training_runs": self.training_runs,
                     "training_dataset": self.training_dataset, "created": self.created,
                     "target": "RUL_fraction"}, path, compress=3)
        return path

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        if not isinstance(data, dict) or data.get("target") != "RUL_fraction":
            raise ValueError("not an RUL model file")
        model = cls(data["kind"])
        model.pipeline = data["pipeline"]
        model.feature_names = data["feature_names"]
        model.training_runs = data["training_runs"]
        model.training_dataset = data["training_dataset"]
        model.created = data["created"]
        return model


def run_metrics(run, prediction):
    """Fraction metrics on the whole run; time metrics past RUL_TIME_METRICS_FROM_LIFE."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    truth = run["rul_fraction"]
    metrics = {
        "run_id": run["run_id"], "dataset": run["dataset"], "condition": run["condition"],
        "snapshots": int(truth.shape[0]), "total_life_s": run["total_life_s"],
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)) if np.var(truth) > 0 else float("nan"),
    }
    # Percent of total life: the fraction error is already relative to life.
    metrics["mae_percent_life"] = 100.0 * metrics["mae"]
    # Rank correlation: does the prediction FALL as the machine ages, even if
    # the absolute level is off? 1.0 = perfect ordering, 0 = none. A model can
    # be useless by R2 yet still rank degradation correctly, which is what an
    # alarm threshold needs.
    if np.std(prediction) > 0 and np.std(truth) > 0:
        metrics["rank_correlation"] = float(spearmanr(truth, prediction).correlation)
    else:
        metrics["rank_correlation"] = float("nan")

    times = run["times"]
    late = run["life_fraction"] >= config.RUL_TIME_METRICS_FROM_LIFE
    f = np.clip(prediction[late], 0.0, 0.99)
    elapsed = times[late]
    predicted_time = elapsed * f / (1.0 - f)
    true_time = run["total_life_s"] - elapsed
    if late.any():
        error = predicted_time - true_time
        metrics["late_life_time_mae_s"] = float(np.mean(np.abs(error)))
        metrics["late_life_time_error_percent_life"] = float(
            100.0 * np.mean(np.abs(error)) / max(run["total_life_s"], 1e-9))
    return metrics


def aggregate(metrics_list):
    if not metrics_list:
        return {}
    keys = ("mae", "rmse", "r2", "rank_correlation", "mae_percent_life",
            "late_life_time_error_percent_life")
    out = {"runs": len(metrics_list)}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        if values:
            out[key] = float(np.mean(values))
    return out


def leave_one_run_out(runs, kind):
    """Cross validation by complete run. Returns per run metrics."""
    results = []
    for held_out in runs:
        train = [r for r in runs if r["run_id"] != held_out["run_id"]]
        assert_disjoint(train, [held_out])
        model = RULModel(kind).fit(train)
        prediction, _ = model.predict_run(held_out)
        results.append(run_metrics(held_out, prediction))
    return results
