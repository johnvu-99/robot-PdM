"""
"Dumb" RUL baseline: always predict the same RUL fraction (0.5 = half of life left).

A real model has to beat this to be worth anything. Uses the same runs and the
same metric code as tools.rul, so the numbers are directly comparable.

Run from the project root:

    python results/baseline_2026-09-28/constant_guess.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from predictive.dataset_jobs import JobContext, _protocol_runs  # noqa: E402
from predictive.rul_model import aggregate, run_metrics  # noqa: E402

GUESS = 0.5


def main():
    ctx = JobContext(None, None)
    for protocol in ("XJTU_TO_PHM", "PHM_LEARNING_TO_FULL_TEST"):
        _, _, test = _protocol_runs(ctx, {"protocol": protocol})
        metrics = [run_metrics(run, np.full(run["rul_fraction"].shape, GUESS)) for run in test]
        summary = aggregate(metrics)
        print("%-28s constant %.1f on %d validation bearings: MAE %.3f  RMSE %.3f  R2 %.2f  rank n/a"
              % (protocol, GUESS, len(test), summary["mae"], summary["rmse"], summary["r2"]))


if __name__ == "__main__":
    main()
