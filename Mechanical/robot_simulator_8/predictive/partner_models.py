"""
Models for partner collected data. Chosen automatically from label_mode:

    fault_labels    anomaly detection (healthy data only) + fault classification
    unlabeled       anomaly detection, assuming training runs are mostly normal
    run_to_failure  anomaly detection on early life + RUL regression

Leakage rules (always):
  * windows are grouped by run_id; a run is entirely train or entirely test
  * scalers are fitted on training data only (inside models / pipelines)
  * with a single run per group, only later time blocks are tested, and the
    report says that this is weaker evidence
"""

import datetime

import numpy as np

import config
from predictive.anomaly_engine import AnomalyModel, time_blocks


def group_split(groups, test_fraction=None, seed=None, stratify=None):
    """Hold out whole groups. stratify: optional dict group -> class, to keep every class in train."""
    test_fraction = config.PARTNER_TEST_FRACTION if test_fraction is None else test_fraction
    rng = np.random.RandomState(config.PARTNER_SPLIT_SEED if seed is None else seed)
    unique = sorted(set(groups))
    if stratify:
        by_class = {}
        for g in unique:
            by_class.setdefault(stratify[g], []).append(g)
        test = []
        for members in by_class.values():
            members = list(members)
            rng.shuffle(members)
            k = int(round(len(members) * test_fraction))
            if len(members) >= 2:
                test.extend(members[:max(1, k)])
    else:
        members = list(unique)
        rng.shuffle(members)
        test = members[:max(1, int(round(len(members) * test_fraction)))] if len(members) >= 2 else []
    test = set(test)
    return [g for g in unique if g not in test], sorted(test)


def _setups(data):
    return sorted(set(data["conditions"]))


# -- anomaly ---------------------------------------------------------------------

def train_anomaly(data):
    """One AnomalyModel per condition. Returns (models, report)."""
    x, groups, labels = data["X"], data["groups"], data["labels"]
    mode, healthy_label = data["label_mode"], data["healthy_label"]
    models, reports = {}, []
    for setup in _setups(data):
        in_setup = data["conditions"] == setup
        if mode == "fault_labels":
            normal = in_setup & (labels == healthy_label)
        elif mode == "run_to_failure":
            normal = in_setup & (data["life_fraction"] <= 0.3)       # early life treated as healthy
        else:
            normal = in_setup
        normal_groups = sorted(set(groups[normal]))
        train_groups, test_groups = group_split(normal_groups)
        evidence = "held out runs %s" % ", ".join(test_groups)
        train_mask = normal & np.isin(groups, train_groups)
        healthy_test_mask = normal & np.isin(groups, test_groups)
        if not test_groups:
            # One normal run: train on its early time block, test on the later block.
            index = np.flatnonzero(normal)
            train_idx, test_idx = time_blocks(index.shape[0])
            train_mask = np.zeros_like(normal)
            train_mask[index[train_idx]] = True
            healthy_test_mask = np.zeros_like(normal)
            healthy_test_mask[index[test_idx]] = True
            evidence = "single normal run: later time block of the same run (weaker evidence)"
        parts = []
        for stream in sorted(set(data["streams"][train_mask])):
            idx = np.flatnonzero(train_mask & (data["streams"] == stream))
            parts.append(x[idx[np.argsort(data["times"][idx])]])
        try:
            model = AnomalyModel(setup, data["names"]).fit(parts)
        except ValueError as exc:
            reports.append({"setup": setup, "skipped": str(exc)})
            continue

        faulty = {}
        if mode == "fault_labels":
            for label in sorted(set(labels[in_setup])):
                if label != healthy_label:
                    faulty[label] = x[in_setup & (labels == label)]
        elif mode == "run_to_failure":
            late = in_setup & (data["life_fraction"] >= 0.8) & np.isin(groups, test_groups or normal_groups)
            faulty["LATE_LIFE (last 20%)"] = x[late]
        report = model.evaluate(x[healthy_test_mask], faulty) if healthy_test_mask.any() else \
            {"setup": setup, "labels": {}}
        report["evidence"] = evidence
        report["train_windows"] = int(train_mask.sum())
        if mode == "unlabeled":
            flagged = {}
            for group in sorted(set(groups[in_setup])):
                scores = model.score(x[in_setup & (groups == group)])
                flagged[group] = float(np.mean([s["anomalous"] for s in scores]))
            report["anomalous_fraction_per_run"] = flagged
        models[setup] = model
        reports.append(report)
    return models, reports


# -- classification ---------------------------------------------------------------

def train_classifier(data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x, y, groups = data["X"], data["labels"], data["groups"]
    classes = sorted(set(y))
    if len(classes) < 2:
        return None, {"skipped": "need at least two labels"}
    group_class = dict((g, y[groups == g][0]) for g in set(groups))
    train_groups, test_groups = group_split(sorted(group_class), stratify=group_class)
    train = np.isin(groups, train_groups)
    test = np.isin(groups, test_groups)
    missing = sorted(set(classes) - set(y[train]))
    if not test.any():
        return None, {"skipped": "every label needs at least two runs to test on an unseen run"}
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("model", RandomForestClassifier(n_estimators=300, random_state=config.PARTNER_SPLIT_SEED,
                                                          n_jobs=1))])
    pipeline.fit(x[train], y[train])
    predicted = pipeline.predict(x[test])
    importances = pipeline.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:10]
    report = {
        "train_runs": train_groups, "test_runs": test_groups,
        "accuracy": float(accuracy_score(y[test], predicted)),
        "macro_f1": float(f1_score(y[test], predicted, average="macro")),
        "labels": classes,
        "confusion": confusion_matrix(y[test], predicted, labels=classes).tolist(),
        "top_features": [(data["names"][i], float(importances[i])) for i in order],
    }
    if missing:
        report["warning"] = "labels only in test runs (cannot be predicted): %s" % ", ".join(missing)
    return pipeline, report


# -- remaining useful life ----------------------------------------------------------

def rul_features(data):
    """Per stream: features relative to the stream's first windows (causal, dimensionless)."""
    x = data["X"]
    out = np.empty_like(x)
    for stream in set(data["streams"]):
        idx = np.flatnonzero(data["streams"] == stream)
        idx = idx[np.argsort(data["times"][idx])]
        base_n = max(1, min(config.PARTNER_RUL_BASELINE_WINDOWS, idx.shape[0] // 4))
        base = x[idx[:base_n]].mean(axis=0)
        scale = np.abs(base) + x[idx[:base_n]].std(axis=0) + 1e-9
        out[idx] = (x[idx] - base) / scale
    return out


def train_rul(data, kind="RANDOM_FOREST"):
    from predictive.rul_model import _make_estimator
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    x = rul_features(data)
    y = data["rul_fraction"]
    groups = data["groups"]
    runs = sorted(set(groups))
    if len(runs) < 3:
        return None, {"skipped": "RUL needs at least 3 run-to-failure runs"}
    folds = []
    for held in runs:                         # leave one run out
        train, test = groups != held, groups == held
        model = _make_estimator(kind).fit(x[train], y[train])
        p = np.clip(model.predict(x[test]), 0, 1)
        folds.append({"run_id": held, "mae": float(mean_absolute_error(y[test], p)),
                      "rmse": float(np.sqrt(mean_squared_error(y[test], p))),
                      "r2": float(r2_score(y[test], p)) if np.var(y[test]) > 0 else float("nan")})
    final = _make_estimator(kind).fit(x, y)
    summary = dict((k, float(np.nanmean([f[k] for f in folds]))) for k in ("mae", "rmse", "r2"))
    return final, {"model": kind, "leave_one_run_out": folds, "summary": summary}


# -- orchestration --------------------------------------------------------------------

def train_all(data, rul_kind="RANDOM_FOREST"):
    bundle = {"kind": "partner_models", "version": 1, "created": datetime.datetime.now().isoformat(),
              "schema_hash": data["schema_hash"], "names": data["names"],
              "label_mode": data["label_mode"], "domain": data["domain"]}
    report = {"label_mode": data["label_mode"], "windows": int(data["X"].shape[0]),
              "features": len(data["names"]), "runs": len(set(data["groups"])),
              "skipped_windows": data["skipped_windows"]}
    bundle["anomaly"], report["anomaly"] = train_anomaly(data)
    if data["label_mode"] == "fault_labels":
        bundle["classifier"], report["classification"] = train_classifier(data)
    if data["label_mode"] == "run_to_failure":
        bundle["rul"], report["rul"] = train_rul(data, rul_kind)
    return bundle, report


def score(bundle, data):
    """Score new windows with a trained bundle. Returns per stream summaries."""
    if data["schema_hash"] != bundle["schema_hash"]:
        raise ValueError("channel layout differs from the one the models were trained on")
    results = []
    for stream in sorted(set(data["streams"])):
        mask = data["streams"] == stream
        setup = data["conditions"][mask][0]
        model = bundle["anomaly"].get(setup) or next(iter(bundle["anomaly"].values()))
        scores = model.score(data["X"][mask])
        worst = max(scores, key=lambda s: s["ratio"])
        entry = {"stream": stream, "setup_model": model.setup, "windows": len(scores),
                 "anomalous_fraction": float(np.mean([s["anomalous"] for s in scores])),
                 "worst_ratio": worst["ratio"], "worst_features": worst["top_features"]}
        if bundle.get("classifier") is not None:
            predicted = bundle["classifier"].predict(data["X"][mask])
            values, counts = np.unique(predicted, return_counts=True)
            entry["predicted_label"] = str(values[np.argmax(counts)])
            entry["label_vote"] = float(counts.max() / counts.sum())
        if bundle.get("rul") is not None:
            prediction = np.clip(bundle["rul"].predict(rul_features(data)[mask]), 0, 1)
            entry["rul_fraction_last"] = float(prediction[np.argmax(data["times"][mask])])
        results.append(entry)
    return results
