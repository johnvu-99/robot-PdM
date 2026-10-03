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

    python results/experiments/eval32.py [variant ...]
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

VARIANTS = {
    # name: (loudness reference, gate, config overrides)
    "current":        ("QUIETEST", True, {}),
    "original_28sep": ("START", False, {"RUL_CROSS_DATASET_SMOOTHING_TAU_S": 0.0}),
    "b3_kurtosis":    ("QUIETEST", True, {"RUL_KURTOSIS_NEEDS_GROWTH": 0.2}),
}


def load():
    with open(os.path.join(HERE, ".runs_xjtu_to_phm.pkl"), "rb") as handle:
        xjtu, phm = pickle.load(handle)
    return xjtu, phm


def evaluate(name, xjtu, phm, views):
    reference, gate, overrides = VARIANTS[name]
    for key, value in dict(DEFAULTS, **overrides).items():
        setattr(config, key, value)
    out = {}
    if "A" in views:
        rows = {}
        for train, test in ((xjtu, phm), (phm, xjtu)):
            model = rul_model.RULModel("RANDOM_FOREST").fit(train, reference, gate)
            for run in test:
                rows[run["run_id"]] = run_metrics(run, model.predict_run(run)[0])
        out["A"] = rows
    if "B" in views:
        rows = {}
        everything = xjtu + phm
        for held_out in everything:
            others = [r for r in everything if r["run_id"] != held_out["run_id"]]
            model = rul_model.RULModel("RANDOM_FOREST").fit(others, reference, gate)
            rows[held_out["run_id"]] = run_metrics(held_out, model.predict_run(held_out)[0])
        out["B"] = rows
    return out


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
    names = names or list(VARIANTS)
    if "current" not in names:
        names = ["current"] + names
    xjtu, phm = load()
    cache_path = os.path.join(HERE, ".eval32_cache.pkl")
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    for name in names:
        if name not in cache:
            cache[name] = evaluate(name, xjtu, phm, "AB")
            pickle.dump(cache, open(cache_path, "wb"))
    titles = {"A": "View A, new machine (train on one lab, test on the other; 32 bearings)",
              "B": "View B, new bearing (leave one out of all 32)"}
    for view in "AB":
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
            print(line)
    if os.environ.get("PER_BEARING"):
        for view in "AB":
            print("\nPer bearing, view %s (MAE / rank): %s" % (view, ", ".join(names)))
            for run_id in sorted(cache["current"][view]):
                print("  %-18s %s" % (run_id, "   ".join("%.3f / %5.2f" % (
                    cache[n][view][run_id]["mae"], cache[n][view][run_id]["rank_correlation"]) for n in names)))


if __name__ == "__main__":
    main(sys.argv[1:])
