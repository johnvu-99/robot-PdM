"""
Experiment: uncertainty ranges for the XJTU -> PHM RUL model.

A range is only useful if it is honest: a "90%" range must contain the true RUL
about 90% of the time. This script builds ranges from the model's own errors on
bearings it did not train on (conformal prediction) and measures, on the PHM
bearings, how often the truth really falls inside ("coverage") and how wide the
ranges are.

    python results/experiments/uncertainty.py

Error sources:
  XJTU-CV  errors from leave-one-bearing-out cross validation on XJTU-SY.
           The model has still never seen a PHM bearing.
  PHM-dev  errors on the 6 PHM Learning_set bearings ("calibrate on a few
           bearings of the target machine"). Only the 11 test bearings count.

Range shapes:
  fixed    one +/- width for every prediction
  binned   lower and upper error quantiles per band of predicted RUL
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from predictive import rul_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = 0.90
POINTS_PER_BEARING = 200          # every bearing counts the same, long or short
BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])


def subset(run):
    learning = ("Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2")
    return "dev" if run["run_id"].replace("PHM_", "") in learning else "test"


def resample(prediction, truth):
    index = np.linspace(0, len(truth) - 1, POINTS_PER_BEARING).round().astype(int)
    return prediction[index], truth[index]


def fit_fixed(prediction, truth):
    width = float(np.quantile(np.abs(truth - prediction), LEVEL))
    return lambda p: (np.clip(p - width, 0, 1), np.clip(p + width, 0, 1))


def fit_binned(prediction, truth):
    error = truth - prediction
    low_all, high_all = np.quantile(error, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    lows, highs = [], []
    for a, b in zip(BINS[:-1], BINS[1:]):
        inside = (prediction >= a) & (prediction < b)
        if inside.sum() >= 100:
            low, high = np.quantile(error[inside], [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
        else:
            low, high = low_all, high_all
        lows.append(low)
        highs.append(high)
    lows, highs = np.array(lows), np.array(highs)

    def interval(p):
        band = np.clip(np.digitize(p, BINS) - 1, 0, len(lows) - 1)
        return np.clip(p + lows[band], 0, 1), np.clip(p + highs[band], 0, 1)
    interval.table = list(zip(BINS[:-1], lows, highs))
    return interval


def report(name, interval, predictions, only):
    coverages, widths = [], []
    for run, prediction in predictions:
        if only != "all" and subset(run) != only:
            continue
        low, high = interval(prediction)
        truth = run["rul_fraction"]
        coverages.append(np.mean((truth >= low) & (truth <= high)))
        widths.append(np.mean(high - low))
    print("  %-22s on %-4s (%2d bearings): coverage %.0f%% (worst bearing %.0f%%), average width %.2f"
          % (name, only, len(coverages), 100 * np.mean(coverages), 100 * np.min(coverages), np.mean(widths)))


def main():
    with open(os.path.join(HERE, ".runs_xjtu_to_phm.pkl"), "rb") as handle:
        train, test = pickle.load(handle)
    reference, gate = "QUIETEST", True              # the current XJTU_TO_PHM settings

    cv_prediction, cv_truth = [], []
    for held_out in train:
        others = [r for r in train if r["run_id"] != held_out["run_id"]]
        model = rul_model.RULModel("RANDOM_FOREST").fit(others, reference, gate)
        p, t = resample(model.predict_run(held_out)[0], held_out["rul_fraction"])
        cv_prediction.append(p)
        cv_truth.append(t)
    cv_prediction, cv_truth = np.concatenate(cv_prediction), np.concatenate(cv_truth)

    model = rul_model.RULModel("RANDOM_FOREST").fit(train, reference, gate)
    predictions = [(run, model.predict_run(run)[0]) for run in test]
    dev = [resample(p, run["rul_fraction"]) for run, p in predictions if subset(run) == "dev"]
    dev_prediction = np.concatenate([d[0] for d in dev])
    dev_truth = np.concatenate([d[1] for d in dev])

    print("Target: the true RUL inside the range %.0f%% of the time.\n" % (100 * LEVEL))
    print("Errors from XJTU cross validation (model never saw PHM):")
    for name, fit in (("fixed width", fit_fixed), ("binned by prediction", fit_binned)):
        interval = fit(cv_prediction, cv_truth)
        for only in ("dev", "test", "all"):
            report(name, interval, predictions, only)
    print("\nErrors from the 6 PHM dev bearings (calibrated on the target machine):")
    for name, fit in (("fixed width", fit_fixed), ("binned by prediction", fit_binned)):
        interval = fit(dev_prediction, dev_truth)
        report(name, interval, predictions, "test")
        if hasattr(interval, "table"):
            print("     predicted RUL band -> range added to the prediction")
            for start, low, high in interval.table:
                print("       %.1f - %.1f:  %+.2f to %+.2f" % (start, min(start + 0.2, 1.0), low, high))
    interval = fit_binned(cv_prediction, cv_truth)
    print("\nXJTU-CV binned table:")
    for start, low, high in interval.table:
        print("       %.1f - %.1f:  %+.2f to %+.2f" % (start, min(start + 0.2, 1.0), low, high))


if __name__ == "__main__":
    main()
