"""
Anomaly detection on Mechanical-datasets
(https://github.com/cathysiyu/Mechanical-datasets).

    python -m tools.mechanical_anomaly train    --root data/external/Mechanical-datasets
    python -m tools.mechanical_anomaly evaluate --model models/mechanical_anomaly.joblib
    python -m tools.mechanical_anomaly score    --model models/mechanical_anomaly.joblib FILE [FILE ...]

train     one healthy model per setup (rig + condition), threshold on held-out
          healthy data, report on the later test blocks, save a model bundle
evaluate  re-run the test block report for a saved bundle
score     score any SEU CSV or CWRU .mat file window by window

Data: gearbox/gearset and gearbox/bearingset CSVs (SEU drivetrain simulator,
8 channels incl. motor torque) and dataset/*.mat (CWRU bearings). This is real
test rig data, not robot telemetry.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from datasets.mechanical_dataset import MechanicalDatasetAdapter  # noqa: E402
from predictive.anomaly_engine import (  # noqa: E402
    AnomalyModel, load_models, save_models, time_blocks,
)

HEALTHY = "HEALTHY"
DEFAULT_MODEL = os.path.join(config.MODEL_DIR, "mechanical_anomaly.joblib")


def setup_key(recording):
    """Healthy behaviour is only comparable on the same rig at the same speed/load."""
    return "%s | %s | %s" % (recording.meta.get("source", ""), recording.meta.get("subset", recording.system),
                             recording.condition)


def load_features(root, verbose=True):
    adapter = MechanicalDatasetAdapter(root)
    if not adapter.available():
        raise SystemExit("Mechanical-datasets not found at %s" % root)
    setups = {}
    for recording in adapter.discover():
        features, names = adapter.extract_features(recording)
        if verbose:
            print("  %-40s %-30s %4d windows" % (recording.recording_id, recording.label, features.shape[0]))
        setups.setdefault(setup_key(recording), []).append((recording, features, names))
    return adapter, setups


def build(setups):
    models = {}
    reports = []
    for key, items in sorted(setups.items()):
        healthy = [(r, f, n) for r, f, n in items if r.label == HEALTHY]
        faulty = [(r, f, n) for r, f, n in items if r.label != HEALTHY]
        if not healthy:
            reports.append({"setup": key, "skipped": "no healthy recording for this setup"})
            continue
        names = healthy[0][2]
        train, healthy_test = [], []
        for _, features, _ in healthy:
            train_idx, test_idx = time_blocks(features.shape[0])
            train.append(features[train_idx])
            healthy_test.append(features[test_idx])
        try:
            model = AnomalyModel(key, names).fit(train)
        except ValueError as exc:
            reports.append({"setup": key, "skipped": str(exc)})
            continue
        fault_test = {}
        for recording, features, _ in faulty:
            _, test_idx = time_blocks(features.shape[0])
            fault_test.setdefault(recording.label, []).append(features[test_idx])
        report = model.evaluate(np.vstack(healthy_test),
                                dict((k, np.vstack(v)) for k, v in fault_test.items()))
        report["healthy_recordings"] = [r.recording_id for r, _, _ in healthy]
        models[key] = model
        reports.append(report)
    return models, reports


def print_reports(reports):
    for report in reports:
        print("\n== %s" % report["setup"])
        if "skipped" in report:
            print("   skipped: %s" % report["skipped"])
            continue
        print("   healthy test windows %d, false alarm rate %.1f%%"
              % (report["healthy_test_windows"], 100 * report["false_alarm_rate"]))
        if not report["labels"]:
            print("   no fault recordings for this setup (model still usable for scoring)")
        for label, r in report["labels"].items():
            print("   %-30s detected %5.1f%%   ROC AUC %.3f   score/threshold %.2f   top: %s"
                  % (label, 100 * r["detection_rate"], r["roc_auc"], r["median_ratio"],
                     ", ".join("%s z=%.0f" % f for f in r["top_features"][:2])))


def command_train(args):
    print("Extracting features (cached in %s)" % config.PROCESSED_DIR)
    _, setups = load_features(args.root)
    models, reports = build(setups)
    if not models:
        raise SystemExit("no setup had enough healthy data")
    save_models(args.model, models, {"root": os.path.abspath(args.root), "reports": reports})
    print_reports(reports)
    report_path = os.path.splitext(args.model)[0] + "_report.json"
    with open(report_path, "w") as handle:
        json.dump(reports, handle, indent=2)
    print("\nSaved %d setup models to %s\nReport: %s" % (len(models), args.model, report_path))


def command_evaluate(args):
    bundle = load_models(args.model)
    root = args.root or bundle["metadata"]["root"]
    _, setups = load_features(root, verbose=False)
    reports = []
    for key, items in sorted(setups.items()):
        model = bundle["models"].get(key)
        if model is None:
            continue
        healthy_test, fault_test = [], {}
        for recording, features, _ in items:
            _, test_idx = time_blocks(features.shape[0])
            if recording.label == HEALTHY:
                healthy_test.append(features[test_idx])
            else:
                fault_test.setdefault(recording.label, []).append(features[test_idx])
        if healthy_test:
            reports.append(model.evaluate(np.vstack(healthy_test),
                                          dict((k, np.vstack(v)) for k, v in fault_test.items())))
    print_reports(reports)


def command_score(args):
    bundle = load_models(args.model)
    root = args.root or bundle["metadata"]["root"]
    adapter = MechanicalDatasetAdapter(root)
    by_path = dict((os.path.abspath(r.files[0]), r) for r in adapter.discover())
    for path in args.files:
        recording = by_path.get(os.path.abspath(path))
        if recording is None:
            print("%s: not a recognised Mechanical-datasets file name (e.g. health_20_0.csv)" % path)
            continue
        key = setup_key(recording)
        model = bundle["models"].get(key)
        if model is None:
            print("%s: no model for setup %s" % (path, key))
            continue
        features, _ = adapter.extract_features(recording)
        results = model.score(features)
        flagged = sum(r["anomalous"] for r in results)
        print("\n%s  (setup %s, file label %s)" % (os.path.basename(path), key, recording.label))
        print("   %d / %d windows anomalous (%.0f%%)" % (flagged, len(results), 100.0 * flagged / len(results)))
        worst = max(results, key=lambda r: r["ratio"])
        print("   worst window score/threshold %.2f, top deviations: %s"
              % (worst["ratio"], ", ".join("%s z=%+.1f" % f for f in worst["top_features"])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    train = sub.add_parser("train")
    train.add_argument("--root", default=config.MECHANICAL_DATASET_DIR)
    train.add_argument("--model", default=DEFAULT_MODEL)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", default=DEFAULT_MODEL)
    evaluate.add_argument("--root")
    score = sub.add_parser("score")
    score.add_argument("--model", default=DEFAULT_MODEL)
    score.add_argument("--root")
    score.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "train":
        command_train(args)
    elif args.command == "evaluate":
        command_evaluate(args)
    elif args.command == "score":
        command_score(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
