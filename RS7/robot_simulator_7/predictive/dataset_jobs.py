"""
Dataset and RUL jobs, run in a separate process.

Parsing thousands of vibration files and training forests holds the GIL for a
long time, so none of it runs in the simulator process. The UI starts a job
with multiprocessing (spawn), receives progress and a small result dictionary
on a queue, and loads saved models from disk.

No Qt imports.
"""

import datetime
import os
import time
import traceback

import numpy as np

import config
from datasets.mechanical_dataset import MechanicalDatasetAdapter
from datasets.phm2012 import PHM2012DatasetAdapter
from datasets.xjtu_sy import XJTUSYDatasetAdapter
from predictive import fault_validation
from predictive.rul_model import (
    RULModel, aggregate, assert_disjoint, leave_one_run_out, run_metrics,
)


class Cancelled(Exception):
    pass


def adapters(roots=None):
    roots = roots or {}
    return {
        "MECHANICAL": MechanicalDatasetAdapter(roots.get("MECHANICAL")),
        "XJTU_SY": XJTUSYDatasetAdapter(roots.get("XJTU_SY")),
        "PHM2012": PHM2012DatasetAdapter(roots.get("PHM2012")),
    }


class JobContext(object):

    def __init__(self, queue, cancel_event):
        self.queue = queue
        self.cancel_event = cancel_event

    def report(self, kind, **fields):
        if self.queue is not None:
            message = {"kind": kind}
            message.update(fields)
            self.queue.put(message)

    def progress(self, text, done=0, total=0):
        self.report("progress", text=text, done=done, total=total)

    def cancelled(self):
        return self.cancel_event is not None and self.cancel_event.is_set()

    def check(self):
        if self.cancelled():
            raise Cancelled()


# -- jobs ----------------------------------------------------------------------

def job_load(ctx, params):
    summaries = {}
    for name, adapter in adapters(params.get("roots")).items():
        recordings = adapter.discover() if adapter.available() else []
        summary = adapter.summary(recordings)
        summary["runs"] = [r.describe() for r in recordings][:200]
        summaries[name] = summary
    return {"summaries": summaries}


def job_preprocess(ctx, params):
    """Validate that every recording can be read and segmented; write an index."""
    import json
    report = {}
    for name, adapter in adapters(params.get("roots")).items():
        if not adapter.available():
            report[name] = {"available": False}
            continue
        recordings = adapter.discover()
        problems = []
        for index, recording in enumerate(recordings):
            ctx.check()
            ctx.progress("%s: checking %s" % (name, recording.recording_id), index, len(recordings))
            try:
                if name == "MECHANICAL":
                    segments = adapter.segment(recording, adapter.load_signals(recording))
                    recording.meta["segments"] = int(segments.shape[0])
                else:
                    for path in (recording.files[0], recording.files[-1]):
                        adapter.read_snapshot(path)
            except (IOError, OSError, ValueError) as exc:
                problems.append("%s: %s" % (recording.recording_id, exc))
        index_path = os.path.join(config.PROCESSED_DIR, name, "index.json")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "w") as handle:
            json.dump({"created": datetime.datetime.now().isoformat(),
                       "recordings": [dict(r.describe(), **{"meta": r.meta}) for r in recordings],
                       "problems": problems}, handle, indent=2)
        report[name] = {"available": True, "recordings": len(recordings), "problems": problems,
                        "index": index_path}
    return {"preprocess": report}


def _runs(ctx, adapter, only=None):
    runs = []
    recordings = adapter.discover()
    if only is not None:
        recordings = [r for r in recordings if only(r)]
    for index, recording in enumerate(recordings):
        ctx.check()
        ctx.progress("%s: features for %s (%d snapshots)"
                     % (adapter.name, recording.run_id, len(recording.files)), index, len(recordings))
        runs.append(adapter.extract_run(recording, cancel=ctx.cancelled))
    return runs


def job_extract(ctx, params):
    all_adapters = adapters(params.get("roots"))
    result = {}
    for name in ("XJTU_SY", "PHM2012"):
        adapter = all_adapters[name]
        if adapter.available():
            runs = _runs(ctx, adapter)
            result[name] = {"runs": len(runs), "snapshots": int(sum(r["features"].shape[0] for r in runs))}
    mech = all_adapters["MECHANICAL"]
    if mech.available():
        recordings = mech.discover()
        segments = 0
        for index, recording in enumerate(recordings):
            ctx.check()
            ctx.progress("MECHANICAL: features for %s" % recording.recording_id, index, len(recordings))
            features, _ = mech.extract_features(recording)
            segments += features.shape[0]
        result["MECHANICAL"] = {"recordings": len(recordings), "segments": int(segments)}
    return {"extract": result}


def _protocol_runs(ctx, params):
    all_adapters = adapters(params.get("roots"))
    protocol = params.get("protocol", config.RUL_DEFAULT_PROTOCOL)
    phm = all_adapters["PHM2012"]
    if protocol == "XJTU_TO_PHM":
        xjtu = all_adapters["XJTU_SY"]
        if not xjtu.available() or not xjtu.discover():
            raise ValueError("XJTU-SY not found at %s. Download it (see README) or choose the "
                             "PHM Learning_set protocol." % xjtu.root)
        train = _runs(ctx, xjtu)
        test = _runs(ctx, phm) if phm.available() else []
    elif protocol == "PHM_LEARNING_TO_FULL_TEST":
        train = _runs(ctx, phm, lambda r: r.meta.get("subset") == "Learning_set")
        test = _runs(ctx, phm, lambda r: r.meta.get("subset") == "Full_Test_Set")
    else:
        raise ValueError("unknown protocol %s" % protocol)
    if not train:
        raise ValueError("no training runs found")
    assert_disjoint(train, test)
    return protocol, train, test


def job_train_rul(ctx, params):
    kind = params.get("model", config.RUL_DEFAULT_MODEL)
    protocol, train, test = _protocol_runs(ctx, params)
    cv = []
    if len(train) >= 3:
        ctx.progress("Leave one run out cross validation (%d folds)" % len(train))
        cv = leave_one_run_out(train, kind)
    ctx.check()
    ctx.progress("Fitting %s on %d training runs" % (kind, len(train)))
    started = time.perf_counter()
    model = RULModel(kind).fit(train)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = params.get("path") or os.path.join(config.MODEL_DIR, "rul_%s_%s_%s.joblib"
                                              % (kind.lower(), protocol.lower(), stamp))
    model.save(path)
    return {"model_path": path, "kind": kind, "protocol": protocol,
            "training_runs": model.training_runs, "validation_runs": [r["run_id"] for r in test],
            "cv": cv, "cv_summary": aggregate(cv), "fit_seconds": time.perf_counter() - started}


def job_evaluate_rul(ctx, params):
    model = RULModel.load(params["model_path"])
    protocol, train, test = _protocol_runs(ctx, params)
    leaked = set(model.training_runs) & set(r["run_id"] for r in test)
    if leaked:
        raise ValueError("model was trained on validation runs %s" % sorted(leaked))
    metrics = []
    curves = []
    for run in test:
        ctx.check()
        prediction, health = model.predict_run(run)
        metrics.append(run_metrics(run, prediction))
        step = max(1, run["times"].shape[0] // 600)
        curves.append({"run_id": run["run_id"], "times": run["times"][::step].tolist(),
                       "life_fraction": run["life_fraction"][::step].tolist(),
                       "true_rul": run["rul_fraction"][::step].tolist(),
                       "predicted_rul": prediction[::step].tolist(),
                       "health_indicator": health[::step].tolist(),
                       "total_life_s": run["total_life_s"]})
    return {"model_path": params["model_path"], "kind": model.kind, "protocol": protocol,
            "training_dataset": model.training_dataset, "training_runs": model.training_runs,
            "metrics": metrics, "summary": aggregate(metrics), "curves": curves}


def job_mechanical(ctx, params):
    adapter = adapters(params.get("roots"))["MECHANICAL"]
    if not adapter.available():
        raise ValueError("Mechanical-datasets not found at %s" % adapter.root)
    items = []
    recordings = adapter.discover()
    for index, recording in enumerate(recordings):
        ctx.check()
        ctx.progress("Features for %s" % recording.recording_id, index, len(recordings))
        features, names = adapter.extract_features(recording)
        if features.shape[0] >= 10:
            items.append((recording.describe(), features, names))
    ctx.progress("Training classifier and anomaly detector")
    return {"mechanical": fault_validation.validate(items)}


JOBS = {
    "load": job_load,
    "preprocess": job_preprocess,
    "extract": job_extract,
    "train_rul": job_train_rul,
    "evaluate_rul": job_evaluate_rul,
    "mechanical": job_mechanical,
}


def run_job(name, params, queue=None, cancel_event=None):
    ctx = JobContext(queue, cancel_event)
    started = time.perf_counter()
    try:
        result = JOBS[name](ctx, params or {})
    except Cancelled:
        ctx.report("cancelled", job=name)
        return None
    except (ValueError, KeyError, IOError, OSError, KeyboardInterrupt) as exc:
        ctx.report("error", job=name, message="%s: %s" % (type(exc).__name__, exc))
        return None
    result["seconds"] = time.perf_counter() - started
    ctx.report("finished", job=name, result=result)
    return result


def process_main(name, params, queue, cancel_event):
    """multiprocessing entry point. Every outcome is reported on the queue."""
    try:
        run_job(name, params, queue, cancel_event)
    except Exception as exc:  # the UI must always hear back from the child
        queue.put({"kind": "error", "job": name, "message": "%s: %s\n%s" % (
            type(exc).__name__, exc, traceback.format_exc(limit=4))})
