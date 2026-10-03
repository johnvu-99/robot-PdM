"""
A stronger test for RUL ideas: 32 bearings instead of 6 + 11.

View A, "new machine": train on one lab, test on the other, both directions
    (XJTU-SY -> PHM 2012: 17 bearings; PHM 2012 -> XJTU-SY: 15 bearings).
    Every bearing is predicted by a model that never saw its lab.
View B, "new bearing": leave one bearing out of all 32; the model has seen
    other bearings from the same lab.

Each variant is compared with the current system bearing by bearing (paired).
The 95% interval of the mean MAE difference comes from resampling the 32
bearings 10000 times.

Decision rule, fixed before the first run: adopt a change only if in View A the
mean MAE improves AND the interval excludes zero, and View B is not worse
(its interval must not lie entirely above zero).

    python results/experiments/eval32.py [A|B|AB] [variant ...]
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from predictive import rul_model  # noqa: E402
from predictive.rul_model import run_metrics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Experiments only: use every CPU core. Results are identical (fixed seed).
_make = rul_model._make_estimator


def _fast(kind):
    pipeline = _make(kind)
    if kind == "RANDOM_FOREST":
        pipeline.set_params(model__n_jobs=-1)
    return pipeline


rul_model._make_estimator = _fast

SETTINGS = ("RUL_KURTOSIS_NEEDS_GROWTH", "RUL_CROSS_DATASET_SMOOTHING_TAU_S", "RUL_FPT_STD_FLOOR")
DEFAULTS = dict((name, getattr(config, name)) for name in SETTINGS)

def _ema_variable(prediction, times, tau_of):
    """Causal EMA whose time constant may change at every step: tau_of(i, value, smoothed so far)."""
    out = np.empty_like(prediction)
    acc, previous = prediction[0], times[0]
    for i, v in enumerate(prediction):
        alpha = 1.0 - np.exp(-max(times[i] - previous, 0.0) / tau_of(i, v, acc))
        previous = times[i]
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def post_relative(fraction, cap_s, floor_s=60.0):
    """Smooth over a fraction of the time the bearing has lived so far (known live)."""
    def post(prediction, run):
        times = np.asarray(run["times"], float)
        return _ema_variable(prediction, times, lambda i, v, acc: min(cap_s, max(floor_s, fraction * (times[i] - times[0]))))
    return post


def post_asymmetric(tau_down, tau_up):
    """React quickly when the prediction falls, slowly when it rises."""
    def post(prediction, run):
        times = np.asarray(run["times"], float)
        return _ema_variable(prediction, times, lambda i, v, acc: tau_down if v < acc else tau_up)
    return post


NO_SMOOTH = {"RUL_CROSS_DATASET_SMOOTHING_TAU_S": 0.0}

# name: dict(reference, gate, config overrides, post = replaces the built in smoothing, kinds = models averaged)
VARIANTS = {
    "current":          dict(reference="QUIETEST", gate=True),
    "original_28sep":   dict(reference="START", gate=False, config=NO_SMOOTH),
    "b3_kurtosis":      dict(reference="QUIETEST", gate=True, config={"RUL_KURTOSIS_NEEDS_GROWTH": 0.2}),
    # Which fix does what? One at a time, on top of the original.
    "only_smoothing":   dict(reference="START", gate=False),
    "only_quietest":    dict(reference="QUIETEST", gate=False, config=NO_SMOOTH),
    "only_gate":        dict(reference="START", gate=True, config=NO_SMOOTH),
    "quietest_smooth":  dict(reference="QUIETEST", gate=False),
    "gate_smooth":      dict(reference="START", gate=True),
    # Step 3 ideas, on top of the current system.
    "rel_smooth_05":    dict(reference="QUIETEST", gate=True, config=NO_SMOOTH, post=post_relative(0.05, 3600.0)),
    "rel_smooth_10":    dict(reference="QUIETEST", gate=True, config=NO_SMOOTH, post=post_relative(0.10, 3600.0)),
    "rel_smooth_20":    dict(reference="QUIETEST", gate=True, config=NO_SMOOTH, post=post_relative(0.20, 7200.0)),
    "asym_300_900":     dict(reference="QUIETEST", gate=True, config=NO_SMOOTH, post=post_asymmetric(300.0, 900.0)),
    "asym_120_900":     dict(reference="QUIETEST", gate=True, config=NO_SMOOTH, post=post_asymmetric(120.0, 900.0)),
    "avg_3_models":     dict(reference="QUIETEST", gate=True, kinds=config.RUL_MODELS),
    "avg_rf_gb":        dict(reference="QUIETEST", gate=True, kinds=("RANDOM_FOREST", "GRADIENT_BOOSTING")),
}


class Predictor(object):
    """One or several model types (averaged), then an optional post-processing step."""

    def __init__(self, spec, train):
        self.post = spec.get("post")
        self.models = [rul_model.RULModel(kind).fit(train, spec["reference"], spec["gate"])
                       for kind in spec.get("kinds", ("RANDOM_FOREST",))]

    def predict(self, run):
        prediction = np.mean([m.predict_run(run)[0] for m in self.models], axis=0)
        if self.post is not None:
            prediction = np.clip(self.post(prediction, run), 0.0, 1.0)
        return prediction


def load():
    with open(os.path.join(HERE, ".runs_xjtu_to_phm.pkl"), "rb") as handle:
        xjtu, phm = pickle.load(handle)
    return xjtu, phm


def evaluate(name, xjtu, phm, view):
    spec = VARIANTS[name]
    for key, value in dict(DEFAULTS, **spec.get("config", {})).items():
        setattr(config, key, value)
    rows = {}
    if view == "A":
        for train, test in ((xjtu, phm), (phm, xjtu)):
            predictor = Predictor(spec, train)
            for run in test:
                rows[run["run_id"]] = run_metrics(run, predictor.predict(run))
    else:
        everything = xjtu + phm
        for held_out in everything:
            predictor = Predictor(spec, [r for r in everything if r["run_id"] != held_out["run_id"]])
            rows[held_out["run_id"]] = run_metrics(held_out, predictor.predict(held_out))
    return rows


def paired(rows, base, key="mae"):
    ids = sorted(rows)
    diff = np.array([rows[i][key] - base[i][key] for i in ids])
    rng = np.random.RandomState(0)
    means = diff[rng.randint(0, len(diff), (10000, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return diff.mean(), low, high, int((diff < -0.005).sum()), int((diff > 0.005).sum())


def mean(rows, key, only=None):
    values = [m[key] for i, m in rows.items() if (only is None or i.startswith(only)) and not np.isnan(m[key])]
    return float(np.mean(values))


def main(names):
    views = "AB"
    if names and names[0] in ("A", "B", "AB"):          # first argument may limit the views
        views, names = names[0], names[1:]
    names = names or list(VARIANTS)
    if "current" not in names:
        names = ["current"] + names
    xjtu, phm = load()
    cache_path = os.path.join(HERE, ".eval32_cache.pkl")
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    for name in names:
        for view in views:
            if view not in cache.setdefault(name, {}):
                cache[name][view] = evaluate(name, xjtu, phm, view)
                pickle.dump(cache, open(cache_path, "wb"))
    titles = {"A": "View A, new machine (train on one lab, test on the other; 32 bearings)",
              "B": "View B, new bearing (leave one out of all 32)"}
    for view in views:
        print("\n%s" % titles[view])
        print("  %-16s %6s %6s %6s | %-13s %-13s | %s" % (
            "variant", "MAE", "R2", "rank", "PHM bearings", "XJTU bearings", "MAE vs current: mean [95% interval], better / worse"))
        for name in names:
            rows = cache[name][view]
            line = "  %-16s %6.3f %6.2f %6.2f | MAE %.3f     MAE %.3f     |" % (
                name, mean(rows, "mae"), mean(rows, "r2"), mean(rows, "rank_correlation"),
                mean(rows, "mae", "PHM"), mean(rows, "mae", "XJTU"))
            if name != "current":
                d, low, high, better, worse = paired(rows, cache["current"][view])
                line += " %+.3f [%+.3f, %+.3f], %d / %d" % (d, low, high, better, worse)
                r, r_low, r_high, _, _ = paired(rows, cache["current"][view], "rank_correlation")
                line += " | rank %+.2f [%+.2f, %+.2f]" % (r, r_low, r_high)
            print(line)
    if os.environ.get("PER_BEARING"):
        for view in views:
            print("\nPer bearing, view %s (MAE / rank): %s" % (view, ", ".join(names)))
            for run_id in sorted(cache["current"][view]):
                print("  %-18s %s" % (run_id, "   ".join("%.3f / %5.2f" % (
                    cache[n][view][run_id]["mae"], cache[n][view][run_id]["rank_correlation"]) for n in names)))


if __name__ == "__main__":
    main(sys.argv[1:])
