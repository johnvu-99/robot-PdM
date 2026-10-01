"""
Experiments: raise the XJTU -> PHM rank correlation without tuning on the test set.

Train on all 15 XJTU-SY bearings (as the XJTU_TO_PHM protocol does), then score:
  dev  = PHM Learning_set (6 bearings)   <- used to CHOOSE between ideas
  test = PHM Full_Test_Set (11 bearings) <- only reported, never used to choose
  all  = all 17 PHM bearings             <- same set as the baseline table

Each variant changes one thing. Run from the project root:

    python results/experiments/xjtu_to_phm_rank.py [variant ...]
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from predictive import rul_model  # noqa: E402
from predictive.dataset_jobs import JobContext, _protocol_runs  # noqa: E402
from predictive.rul_model import aggregate, run_metrics  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runs_xjtu_to_phm.pkl")


def load_runs():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as handle:
            return pickle.load(handle)
    _, train, test = _protocol_runs(JobContext(None, None), {"protocol": "XJTU_TO_PHM"})
    with open(CACHE, "wb") as handle:
        pickle.dump((train, test), handle)
    return train, test


def subset(run):
    learning = ("Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2")
    return "dev" if run["run_id"].replace("PHM_", "") in learning else "test"


# -- post-processing of the predicted RUL curve (causal: uses only the past) --

def post_none(prediction, run):
    return prediction


def post_runmin(prediction, run):
    """RUL can only go down: damage does not heal."""
    return np.minimum.accumulate(prediction)


def post_ema_runmin(tau_s):
    def post(prediction, run):
        smooth = rul_model._ema_time(prediction, np.asarray(run["times"], float), tau_s)
        return np.minimum.accumulate(smooth)
    return post


def post_ema(tau_s):
    def post(prediction, run):
        return rul_model._ema_time(prediction, np.asarray(run["times"], float), tau_s)
    return post


def post_two_speed(tau_before, tau_after):
    """A1: slow smoothing until degradation is detected, faster afterwards."""
    def post(prediction, run):
        matrix, names, _ = rul_model.rul_features(run)
        fpt = rul_model.detect_fpt(run, matrix, names)
        times = np.asarray(run["times"], float)
        out = np.empty_like(prediction)
        acc, previous = prediction[0], times[0]
        for i, v in enumerate(prediction):
            tau = tau_before if i < fpt else tau_after
            alpha = 1.0 - np.exp(-max(times[i] - previous, 0.0) / tau)
            previous = times[i]
            acc = alpha * v + (1.0 - alpha) * acc
            out[i] = acc
        return out
    return post


def post_proportional(fraction, cap_s=900.0, floor_s=60.0):
    """A2: smooth over a fraction of the time lived so far (known live), capped."""
    def post(prediction, run):
        times = np.asarray(run["times"], float)
        out = np.empty_like(prediction)
        acc, previous = prediction[0], times[0]
        for i, v in enumerate(prediction):
            tau = min(cap_s, max(floor_s, fraction * (times[i] - times[0])))
            alpha = 1.0 - np.exp(-max(times[i] - previous, 0.0) / tau)
            previous = times[i]
            acc = alpha * v + (1.0 - alpha) * acc
            out[i] = acc
        return out
    return post


VARIANTS = {
    "baseline":          dict(post=post_none),
    "runmin":            dict(post=post_runmin),
    "ema300":            dict(post=post_ema(300.0)),
    "ema900":            dict(post=post_ema(900.0)),
    "ema300_runmin":     dict(post=post_ema_runmin(300.0)),
    "ema900_runmin":     dict(post=post_ema_runmin(900.0)),
    "ema1800_runmin":    dict(post=post_ema_runmin(1800.0)),
    # Fix 1: loudness measured from the quietest point so far (current system = "ema900").
    "quiet60":           dict(post=post_ema(900.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_QUIETEST_TAU_S=60.0)),
    "quiet300":          dict(post=post_ema(900.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_QUIETEST_TAU_S=300.0)),
    "quiet900":          dict(post=post_ema(900.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_QUIETEST_TAU_S=900.0)),
    "both300":           dict(post=post_ema(900.0), config=dict(RUL_LOUDNESS_REFERENCE="BOTH", RUL_QUIETEST_TAU_S=300.0)),
    # Fix 2, on top of Fix 1 (quiet300): shape-only features and/or trend features.
    "f2_trend":          dict(post=post_ema(900.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_TREND_FEATURES=True)),
    "f2_shape":          dict(post=post_ema(900.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_DROP_LOUDNESS_FEATURES=True)),
    # Fix 4, on top of Fix 1 (quiet300): two stages. k = how many std above normal.
    "f4_k3":             dict(post=post_ema(900.0), two_stage=dict(k=3.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "f4_k5":             dict(post=post_ema(900.0), two_stage=dict(k=5.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "f4_k8":             dict(post=post_ema(900.0), two_stage=dict(k=8.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "f4_k5_gate_only":   dict(post=post_ema(900.0), two_stage=dict(k=5.0, gate_only=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    # Weak bearings, on top of the current system ("f4_k5_gate_only").
    # A: Bearing2_7, smoothing too slow for a short life.
    "a1_two_speed_300":  dict(post=post_two_speed(900.0, 300.0), two_stage=dict(k=5.0, gate_only=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "a1_two_speed_120":  dict(post=post_two_speed(900.0, 120.0), two_stage=dict(k=5.0, gate_only=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "a2_prop_10":        dict(post=post_proportional(0.10), two_stage=dict(k=5.0, gate_only=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "a2_prop_25":        dict(post=post_proportional(0.25), two_stage=dict(k=5.0, gate_only=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    # B: Bearing2_3, kurtosis spikes without RMS growth.
    "b1_fpt_rms_only":   dict(post=post_ema(900.0), two_stage=dict(k=5.0, gate_only=True, signals=("rms_ema_fast",)), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "b2_fpt_both":       dict(post=post_ema(900.0), two_stage=dict(k=5.0, gate_only=True, need_all=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST")),
    "b3_kurt_conf_02":   dict(post=post_ema(900.0), two_stage=dict(k=5.0, gate_only=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_KURTOSIS_NEEDS_GROWTH=0.2)),
    "b3_kurt_conf_05":   dict(post=post_ema(900.0), two_stage=dict(k=5.0, gate_only=True), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_KURTOSIS_NEEDS_GROWTH=0.5)),
    "f2_shape_trend":    dict(post=post_ema(900.0), config=dict(RUL_LOUDNESS_REFERENCE="QUIETEST", RUL_DROP_LOUDNESS_FEATURES=True, RUL_TREND_FEATURES=True)),
}


# -- Fix 4: two stages. Detect the first predicting time (FPT), then predict. --

FPT_SIGNALS = ("rms_ema_fast", "kurtosis_ema_fast")


def detect_fpt(run, matrix, names, k, sustain_s=120.0, healthy_s=600.0, signals=FPT_SIGNALS, need_all=False):
    """
    Index of the first snapshot where degradation has started, causal. Normal
    range = the first healthy_s seconds of this run; degradation = any signal
    above mean + k * std for at least sustain_s seconds. len(run) if never.
    """
    times = np.asarray(run["times"], float) - run["times"][0]
    healthy = times <= healthy_s
    per_signal = []
    for signal in signals:
        hit = np.zeros(len(times), bool)
        for channel in ("horizontal", "vertical"):
            x = matrix[:, names.index("%s_%s" % (channel, signal))]
            mean, std = np.mean(x[healthy]), max(np.std(x[healthy]), 1e-6)
            hit |= x > mean + k * std
        per_signal.append(hit)
    above = np.all(per_signal, axis=0) if need_all else np.any(per_signal, axis=0)
    start = None
    for i in range(len(times)):
        if not above[i]:
            start = None
            continue
        if start is None:
            start = i
        if times[i] - times[start] >= sustain_s and times[start] > healthy_s:
            return start
    return len(times)


class TwoStage(object):
    """Stage 2 trained on the degradation phase only; constant before the FPT."""

    def __init__(self, k, gate_only=False, signals=FPT_SIGNALS, need_all=False):
        self.k, self.gate_only = k, gate_only
        self.fpt_options = dict(signals=tuple(signals), need_all=need_all)

    def fit(self, runs):
        xs, ys, at_fpt = [], [], []
        for run in runs:
            matrix, self.names, _ = rul_model.rul_features(run)
            fpt = detect_fpt(run, matrix, self.names, self.k, **self.fpt_options)
            rows = slice(0, None) if self.gate_only else slice(min(fpt, len(matrix) - 1), None)
            xs.append(matrix[rows])
            ys.append(run["rul_fraction"][rows])
            at_fpt.append(run["rul_fraction"][min(fpt, len(matrix) - 1)])
        self.before_fpt = float(np.median(at_fpt))
        self.pipeline = rul_model._make_estimator("RANDOM_FOREST").fit(np.vstack(xs), np.concatenate(ys))
        self.fpt_positions = at_fpt
        return self

    def predict_run(self, run):
        matrix, names, health = rul_model.rul_features(run)
        fpt = detect_fpt(run, matrix, names, self.k, **self.fpt_options)
        prediction = np.clip(self.pipeline.predict(matrix), 0.0, 1.0)
        prediction[:fpt] = np.maximum(prediction[:fpt], self.before_fpt)
        run.setdefault("_fpt_life", {})[self.k] = float(run["life_fraction"][min(fpt, len(matrix) - 1)])
        return prediction, health


def evaluate(model, runs, post):
    rows = []
    for run in runs:
        prediction, _ = model.predict_run(run)
        rows.append((subset(run), run_metrics(run, np.clip(post(prediction, run), 0.0, 1.0))))
    out = {}
    for name in ("dev", "test", "all"):
        chosen = [m for s, m in rows if name == "all" or s == name]
        out[name] = aggregate(chosen)
    return out, rows


def main(names):
    # Variants apply their own post-processing to the RAW model output.
    config.RUL_CROSS_DATASET_SMOOTHING_TAU_S = 0.0
    config.RUL_MONOTONE_PREDICTION = False
    train, test = load_runs()
    defaults = dict(RUL_LOUDNESS_REFERENCE="START", RUL_QUIETEST_TAU_S=300.0,
                    RUL_DROP_LOUDNESS_FEATURES=False, RUL_TREND_FEATURES=False,
                    RUL_KURTOSIS_NEEDS_GROWTH=0.0)
    models = {}
    print("%-18s | %-26s | %-26s | %-26s" % ("variant", "dev (6) MAE R2 rank",
                                              "test (11) MAE R2 rank", "all (17) MAE R2 rank"))
    for name in names or VARIANTS:
        settings = dict(defaults, **VARIANTS[name].get("config", {}))
        for key, value in settings.items():
            setattr(config, key, value)
        two_stage = VARIANTS[name].get("two_stage")
        key = tuple(sorted(settings.items())) + tuple(sorted((k, str(v)) for k, v in (two_stage or {}).items()))
        if key not in models:
            if two_stage:
                models[key] = TwoStage(**two_stage).fit(train)
                print("   %s: training FPT at RUL %.2f (median), before-FPT output %.2f"
                      % (name, np.median(models[key].fpt_positions), models[key].before_fpt))
            else:
                models[key] = rul_model.RULModel("RANDOM_FOREST").fit(train)
        summary, rows = evaluate(models[key], test, VARIANTS[name]["post"])
        cells = ["%.3f %5.2f %5.2f" % (summary[k]["mae"], summary[k]["r2"], summary[k]["rank_correlation"])
                 for k in ("dev", "test", "all")]
        print("%-18s | %-26s | %-26s | %-26s" % (name, cells[0], cells[1], cells[2]))
        if two_stage:
            lives = [r.get("_fpt_life", {}).get(two_stage["k"], 1.0) for r in test]
            print("   PHM degradation detected at life %s (1.00 = never)"
                  % " ".join("%.2f" % v for v in lives))
        if os.environ.get("PER_BEARING"):
            for s, m in rows:
                print("     %-4s %-16s MAE %.3f R2 %5.2f rank %5.2f"
                      % (s, m["run_id"], m["mae"], m["r2"], m["rank_correlation"]))


if __name__ == "__main__":
    main(sys.argv[1:])
