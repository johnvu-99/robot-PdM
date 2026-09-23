"""
Remaining useful life (RUL) from the command line.

Same code as the RUL tab in the app (predictive.dataset_jobs), so results match.

    python -m tools.rul info
    python -m tools.rul train    [--protocol XJTU_TO_PHM] [--model RANDOM_FOREST]
    python -m tools.rul evaluate --model-file models/rul_....joblib [--curves out.csv]
    python -m tools.rul compare  [--protocol ...]
    python -m tools.rul predict  --model-file models/rul_....joblib RUN_FOLDER

info      what each dataset folder contains
train     leave-one-bearing-out cross validation on the training set, then fit
          on all training bearings and save to models/
evaluate  validate a saved model on bearings it never saw (per bearing metrics)
compare   train and evaluate all three model types, one summary table
predict   RUL along one bearing folder (PHM acc_*.csv or XJTU-SY 1.csv, 2.csv, ...)

Protocols:
    XJTU_TO_PHM                 train on XJTU-SY, validate on PHM 2012 (default)
    PHM_LEARNING_TO_FULL_TEST   train on PHM Learning_set, validate on Full_Test_Set
                                (use this until XJTU-SY is downloaded)

RUL is reported as a fraction of total life: 1.0 = new, 0.0 = failed.
These are rolling bearing test rigs, not robot joints.
"""

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from datasets.base_adapter import Recording, natural_key  # noqa: E402
from datasets.phm2012 import PHM2012DatasetAdapter  # noqa: E402
from datasets.xjtu_sy import XJTUSYDatasetAdapter  # noqa: E402
from predictive import dataset_jobs  # noqa: E402
from predictive.rul_model import RULModel  # noqa: E402


class _Printer(object):
    """Queue stand-in that prints job progress on one updating line."""

    # Only a real terminal can redraw a line; elsewhere (VS Code output panel,
    # redirected to a file) progress is printed once per new step instead.
    live = sys.stdout.isatty()

    def __init__(self):
        self._last = None

    def clear(self):
        if self.live:
            sys.stdout.write("\r\033[K")

    def put(self, message):
        kind = message.get("kind")
        if kind == "progress":
            done, total = message.get("done", 0), message.get("total", 0)
            prefix = "[%d/%d] " % (done + 1, total) if total else ""
            text = "  %s%s" % (prefix, message.get("text", "")[:110])
            if self.live:
                sys.stdout.write("\r\033[K" + text)
                sys.stdout.flush()
            elif text != self._last:
                print(text)
            self._last = text
        elif kind in ("error", "cancelled"):
            self.clear()
            print("ERROR: %s" % message.get("message", kind))


def _run(job, params):
    printer = _Printer()
    result = dataset_jobs.run_job(job, params, printer)
    printer.clear()
    if result is None:
        raise SystemExit(1)
    return result


def _roots(args):
    return {"XJTU_SY": args.xjtu, "PHM2012": args.phm, "MECHANICAL": config.MECHANICAL_DATASET_DIR}


def _fmt(value, pattern="%.3f"):
    return "-" if value is None or (isinstance(value, float) and np.isnan(value)) else pattern % value


def _print_metrics(rows, title):
    print("\n%s" % title)
    print("  %-18s %-16s %8s %7s %7s %6s %6s %11s" % ("bearing", "condition", "life h", "MAE", "RMSE",
                                                     "R2", "rank", "late err %"))
    for m in rows:
        print("  %-18s %-16s %8.2f %7s %7s %6s %6s %11s" % (
            m["run_id"], m["condition"][:16], m["total_life_s"] / 3600.0, _fmt(m["mae"]),
            _fmt(m["rmse"]), _fmt(m["r2"], "%.2f"), _fmt(m.get("rank_correlation"), "%.2f"),
            _fmt(m.get("late_life_time_error_percent_life"), "%.1f")))


def _print_summary(summary, label):
    if not summary:
        return
    print("  %s: MAE %s  RMSE %s  R2 %s  rank %s  (MAE = %s%% of life, %d bearings)" % (
        label, _fmt(summary.get("mae")), _fmt(summary.get("rmse")), _fmt(summary.get("r2"), "%.2f"),
        _fmt(summary.get("rank_correlation"), "%.2f"),
        _fmt(summary.get("mae_percent_life"), "%.1f"), summary.get("runs", 0)))


# -- commands --------------------------------------------------------------------

def command_info(args):
    summaries = _run("load", {"roots": _roots(args)})["summaries"]
    for name in ("XJTU_SY", "PHM2012"):
        s = summaries[name]
        print("\n%s  (%s)" % (name, s["role"]))
        if not s["available"]:
            print("  not found at %s" % s["root"])
            continue
        print("  %d bearings, %d snapshot files, conditions %s" % (s["recordings"], s["files"], s["conditions"]))
        for run in s["runs"]:
            print("    %-18s %-16s %5d snapshots" % (run["run_id"], run["condition"], run["files"]))


def command_train(args):
    print("Training %s, protocol %s" % (args.model, args.protocol))
    params = {"roots": _roots(args), "protocol": args.protocol, "model": args.model}
    if args.out:
        params["path"] = args.out
    result = _run("train_rul", params)
    print("Training bearings  : %s" % ", ".join(result["training_runs"]))
    print("Validation bearings: %s  (never used for training)" % ", ".join(result["validation_runs"]))
    if result["cv"]:
        _print_metrics(result["cv"], "Leave one bearing out cross validation (training set)")
        _print_summary(result["cv_summary"], "cross validation")
    print("\nSaved %s" % result["model_path"])
    print("Next: python -m tools.rul evaluate --model-file %s --protocol %s" % (result["model_path"], args.protocol))
    return result


def command_evaluate(args):
    result = _run("evaluate_rul", {"roots": _roots(args), "protocol": args.protocol,
                                   "model_path": args.model_file})
    print("Model %s trained on %s (%s)" % (result["kind"], result["training_dataset"], result["protocol"]))
    _print_metrics(result["metrics"], "Validation on unseen bearings")
    _print_summary(result["summary"], "validation")
    print("\n  rank = Spearman correlation between true and predicted RUL: is the order right?")
    print("  late err % = error of the predicted remaining TIME, from 50% of life onwards,")
    print("  as a percent of total life (early in life the time estimate is unstable).")
    if args.curves:
        with open(args.curves, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bearing", "time_s", "life_fraction", "true_rul", "predicted_rul",
                             "health_indicator"])
            for curve in result["curves"]:
                for i in range(len(curve["times"])):
                    writer.writerow([curve["run_id"], "%.0f" % curve["times"][i],
                                     "%.4f" % curve["life_fraction"][i], "%.4f" % curve["true_rul"][i],
                                     "%.4f" % curve["predicted_rul"][i],
                                     "%.4f" % curve["health_indicator"][i]])
        print("Curves written to %s" % args.curves)
    return result


def command_compare(args):
    rows = []
    for kind in config.RUL_MODELS:
        print("\n=== %s" % kind)
        args.model = kind
        args.out = None
        trained = command_train(args)
        args.model_file = trained["model_path"]
        args.curves = None
        evaluated = command_evaluate(args)
        rows.append((kind, trained["cv_summary"], evaluated["summary"]))
    print("\nSUMMARY (%s)" % args.protocol)
    print("  %-18s %22s %30s" % ("model", "cross val MAE / R2", "validation MAE / RMSE / R2"))
    for kind, cv, val in rows:
        print("  %-18s %10s / %-9s %11s / %-6s / %-6s" % (
            kind, _fmt(cv.get("mae")), _fmt(cv.get("r2"), "%.2f"), _fmt(val.get("mae")),
            _fmt(val.get("rmse")), _fmt(val.get("r2"), "%.2f")))
    best = min(rows, key=lambda r: r[2].get("mae", 1e9))
    print("  lowest validation MAE: %s" % best[0])


def _recording_for_folder(folder):
    folder = os.path.abspath(folder)
    name = os.path.basename(folder)
    phm = sorted(glob.glob(os.path.join(folder, "acc_*.csv")), key=natural_key)
    if phm:
        adapter = PHM2012DatasetAdapter(os.path.dirname(os.path.dirname(folder)))
        return adapter, Recording("PHM2012", "PHM_%s" % name, "PHM_%s" % name, adapter.system, "",
                                  "RUN_TO_FAILURE", phm, config.PHM_SAMPLE_RATE_HZ,
                                  adapter.snapshot_channels,
                                  snapshot_interval=config.PHM_SNAPSHOT_INTERVAL_S)
    xjtu = sorted([f for f in glob.glob(os.path.join(folder, "*.csv"))
                   if re.match(r"\d+\.csv$", os.path.basename(f))], key=natural_key)
    if xjtu:
        adapter = XJTUSYDatasetAdapter(os.path.dirname(os.path.dirname(folder)))
        return adapter, Recording("XJTU_SY", "XJTU_%s" % name, "XJTU_%s" % name, adapter.system,
                                  os.path.basename(os.path.dirname(folder)), "RUN_TO_FAILURE", xjtu,
                                  config.XJTU_SAMPLE_RATE_HZ, adapter.snapshot_channels,
                                  snapshot_interval=config.XJTU_SNAPSHOT_INTERVAL_S)
    raise SystemExit("%s: no acc_*.csv (PHM 2012) or 1.csv, 2.csv ... (XJTU-SY) files found" % folder)


def command_predict(args):
    model = RULModel.load(args.model_file)
    adapter, recording = _recording_for_folder(args.folder)
    run = adapter.extract_run(recording)
    prediction, health = model.predict_run(run)
    if recording.run_id in model.training_runs:
        print("NOTE: %s was used to train this model, so this is not a fair test." % recording.run_id)
    print("%s: %d snapshots, %.2f h recorded, model %s" % (
        recording.run_id, len(recording.files), run["total_life_s"] / 3600.0, model.kind))
    print("\n  %8s %8s %13s %12s %9s" % ("time h", "% life", "predicted RUL", "remaining h", "health"))
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        i = min(len(prediction) - 1, int(round(fraction * (len(prediction) - 1))))
        t = run["times"][i]
        f = float(prediction[i])
        remaining = t * f / (1.0 - f) / 3600.0 if 0.0 < f < 0.99 and t > 0 else None
        print("  %8.2f %7.0f%% %13.2f %12s %9.2f" % (t / 3600.0, 100 * run["life_fraction"][i], f,
                                                   _fmt(remaining, "%.2f"), health[i]))
    print("\n  '% life' assumes the folder ends at failure (true for PHM 2012 and XJTU-SY).")
    print("  For a machine still running, read the last row: predicted RUL is the share of life left.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xjtu", default=config.XJTU_SY_DIR, help="XJTU-SY folder")
    parser.add_argument("--phm", default=config.PHM2012_DIR, help="PHM 2012 folder")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info")
    protocols = list(config.RUL_PROTOCOLS)
    for name in ("train", "evaluate", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", default=config.RUL_DEFAULT_PROTOCOL, choices=protocols)
        if name == "train":
            p.add_argument("--model", default=config.RUL_DEFAULT_MODEL, choices=config.RUL_MODELS)
            p.add_argument("--out", help="model file path (default: models/rul_<model>_<protocol>_<time>.joblib)")
        if name == "evaluate":
            p.add_argument("--model-file", required=True)
            p.add_argument("--curves", help="write true/predicted RUL curves to this CSV")
    predict = sub.add_parser("predict")
    predict.add_argument("--model-file", required=True)
    predict.add_argument("folder", help="one bearing folder")
    args = parser.parse_args(argv)
    handlers = {"info": command_info, "train": command_train, "evaluate": command_evaluate,
                "compare": command_compare, "predict": command_predict}
    if args.command not in handlers:
        parser.print_help()
        return 0
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
