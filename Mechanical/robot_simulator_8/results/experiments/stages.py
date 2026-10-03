"""
Experiment: how good is the three stage answer (early / late / near failure)?

Stage boundaries are in config.RUL_STAGES and were fixed before this was run.
View A of eval32.py: train on one lab, test on the other, both directions, so
every bearing is judged by a model that never saw its lab.

    python results/experiments/stages.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from predictive.rul_model import run_metrics  # noqa: E402
from results.experiments.eval32 import VARIANTS, Predictor, load  # noqa: E402

KEYS = ("stage_accuracy", "near_failure_caught", "near_failure_too_early", "first_warning_life")


def line(name, rows):
    def mean(key):
        return np.nanmean([r[key] for r in rows])
    never = sum(1 for r in rows if np.isnan(r["first_warning_life"]))
    early = sum(1 for r in rows if r["first_warning_life"] < 0.5)
    print("  %-34s right stage %3.0f%% | near failure caught %3.0f%% | called near failure while early %3.0f%% | "
          "first warning at %3.0f%% of life | never warned %2d, warned before half of life %2d (of %d)"
          % (name, 100 * mean("stage_accuracy"), 100 * mean("near_failure_caught"),
             100 * mean("near_failure_too_early"), 100 * mean("first_warning_life"), never, early, len(rows)))


def calibrated_cuts(spec, train):
    """
    Stage cut-offs on the PREDICTED scale, from held out predictions on the
    training lab: the cut for a stage that truly starts at RUL b is the value
    below which a share b of the held out predictions fall (quantile matching).
    """
    held_out = []
    for run in train:
        predictor = Predictor(spec, [r for r in train if r["run_id"] != run["run_id"]])
        prediction = predictor.predict(run)
        index = np.linspace(0, len(prediction) - 1, 200).round().astype(int)
        held_out.append(prediction[index])
    held_out = np.concatenate(held_out)
    return dict((b, float(np.quantile(held_out, b))) for b in (0.2, 0.5))


def remap(prediction, cuts):
    """Move a prediction onto the true scale piecewise, so the fixed stage boundaries apply."""
    return np.interp(prediction, [0.0, cuts[0.2], cuts[0.5], 1.0], [0.0, 0.2, 0.5, 1.0])


def main():
    xjtu, phm = load()
    for label, train, test in (("XJTU-SY -> PHM 2012", xjtu, phm), ("PHM 2012 -> XJTU-SY", phm, xjtu)):
        print("\n%s (%d bearings)" % (label, len(test)))
        spec = VARIANTS["current"]
        predictor = Predictor(spec, train)
        predictions = [(run, predictor.predict(run)) for run in test]
        line("model: current", [run_metrics(run, p) for run, p in predictions])
        cuts = calibrated_cuts(spec, train)
        print("     calibrated cut-offs on the predicted scale: near failure below %.2f, late below %.2f"
              % (cuts[0.2], cuts[0.5]))
        rows = []
        for run, p in predictions:
            metrics = run_metrics(run, remap(p, cuts))
            rows.append(metrics)
        line("model: current, calibrated cuts", rows)
        original = Predictor(VARIANTS["original_28sep"], train)
        line("model: original_28sep", [run_metrics(run, original.predict(run)) for run in test])
        for guess, text in ((0.9, "always EARLY"), (0.1, "always NEAR FAILURE")):
            line("guess: %s" % text, [run_metrics(run, np.full(run["rul_fraction"].shape, guess)) for run in test])


if __name__ == "__main__":
    main()
