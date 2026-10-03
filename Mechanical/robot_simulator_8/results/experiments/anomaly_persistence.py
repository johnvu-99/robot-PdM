"""
Experiment: fewer false alarms on healthy data (CWRU load 3 had 22%).

Rule fixed BEFORE running (not tuned on these results):
  PERSISTENCE  a window raises an alarm only if it AND the window just before
               it, in the same recording, are both flagged.

Separate measurement, same data:
  NO_DOMINANT  train without the dominant frequency features (a peak-picking
               feature that jumps between neighbouring frequency bins).

    python results/experiments/anomaly_persistence.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from predictive.anomaly_engine import AnomalyModel, time_blocks  # noqa: E402
from tools.mechanical_anomaly import HEALTHY, load_features  # noqa: E402


def persistent(flags):
    flags = np.asarray(flags, dtype=bool)
    out = np.zeros_like(flags)
    out[1:] = flags[1:] & flags[:-1]
    return out


def rates(model, blocks):
    """blocks: list of per recording test blocks. Returns (raw rate, persistent rate, windows)."""
    raw, kept = [], []
    for block in blocks:
        flags = np.array([r["anomalous"] for r in model.score(block)])
        raw.append(flags)
        kept.append(persistent(flags))
    raw, kept = np.concatenate(raw), np.concatenate(kept)
    return float(raw.mean()), float(kept.mean()), int(raw.shape[0])


def run(setups, drop_dominant):
    rows = []
    for key, items in sorted(setups.items()):
        healthy = [(r, f, n) for r, f, n in items if r.label == HEALTHY]
        faulty = [(r, f, n) for r, f, n in items if r.label != HEALTHY]
        if not healthy:
            continue
        names = list(healthy[0][2])
        keep = [i for i, n in enumerate(names) if not (drop_dominant and "dominant_frequency" in n)]
        names = [names[i] for i in keep]
        train, healthy_test = [], []
        for _, features, _ in healthy:
            train_idx, test_idx = time_blocks(features.shape[0])
            train.append(features[train_idx][:, keep])
            healthy_test.append(features[test_idx][:, keep])
        model = AnomalyModel(key, names).fit(train)
        false_raw, false_kept, n_healthy = rates(model, healthy_test)
        by_label = {}
        for recording, features, _ in faulty:
            _, test_idx = time_blocks(features.shape[0])
            by_label.setdefault(recording.label, []).append(features[test_idx][:, keep])
        detections = dict((label, rates(model, blocks)) for label, blocks in sorted(by_label.items()))
        rows.append((key, n_healthy, false_raw, false_kept, detections))
    return rows


def show(title, rows):
    print("\n%s" % title)
    print("  %-34s %8s %14s %14s   %s" % ("setup", "healthy", "false alarms", "with rule", "weakest fault: detected -> with rule"))
    for key, n, raw, kept, detections in rows:
        label, (d_raw, d_kept, _) = min(detections.items(), key=lambda kv: kv[1][1])
        print("  %-34s %8d %13.1f%% %13.1f%%   %s: %.1f%% -> %.1f%%"
              % (key[-34:], n, 100 * raw, 100 * kept, label, 100 * d_raw, 100 * d_kept))
    total = sum(r[1] for r in rows)
    print("  all setups, weighted by healthy windows: false alarms %.1f%% -> %.1f%%"
          % (100 * sum(r[1] * r[2] for r in rows) / total, 100 * sum(r[1] * r[3] for r in rows) / total))
    worst = min(d[1] for r in rows for d in r[4].values())
    print("  lowest detection with the rule, any fault in any setup: %.1f%%" % (100 * worst))


def main():
    _, setups = load_features(config.MECHANICAL_DATASET_DIR, verbose=False)
    show("Current features", run(setups, drop_dominant=False))
    show("Without dominant frequency features", run(setups, drop_dominant=True))


if __name__ == "__main__":
    main()
