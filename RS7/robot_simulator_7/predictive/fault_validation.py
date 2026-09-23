"""
Fault feature validation on Mechanical-datasets.

Two experiments on real gearbox/bearing vibration and torque:
  1. Fault classification (RandomForestClassifier).
  2. Anomaly detection: IsolationForest trained on HEALTHY segments only, then
     the detection rate per fault label at the false alarm rate on held-out
     healthy segments.

Segments of one recording are correlated, so each recording is split into a
contiguous training block and a later test block separated by a gap. The
scaler is fitted on training segments only.
"""

import numpy as np

import config


def split_blocks(n_segments):
    test = max(1, int(round(n_segments * config.MECH_TEST_FRACTION)))
    train_end = max(1, n_segments - test - config.MECH_GAP_SEGMENTS)
    return np.arange(0, train_end), np.arange(n_segments - test, n_segments)


def _setup_key(info):
    """Rig and operating condition. Healthy behaviour is only comparable within one."""
    return "%s | %s" % (info.get("system", ""), info.get("condition", ""))


def _anomaly_by_setup(items, scaler):
    """
    One IsolationForest per rig and condition, fitted on the first 70% of that
    setup's healthy training block. Threshold: 95th percentile of scores on the
    remaining 30% (calibration), never on test segments.
    Reports detection rate per label and ROC AUC (threshold free).
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score

    setups = {}
    for info, features in items:
        setups.setdefault(_setup_key(info), []).append((info, features))
    results = []
    for key, members in sorted(setups.items()):
        healthy = [f for info, f in members if info["label"] == "HEALTHY"]
        faulty = [(info, f) for info, f in members if info["label"] != "HEALTHY"]
        if not healthy or not faulty:
            continue
        train_parts, healthy_test = [], []
        for features in healthy:
            train_idx, test_idx = split_blocks(features.shape[0])
            train_parts.append(features[train_idx])
            healthy_test.append(features[test_idx])
        fit_parts, calibration_parts = [], []
        for part in train_parts:
            cut = max(1, int(round(part.shape[0] * 0.7)))
            fit_parts.append(part[:cut])
            calibration_parts.append(part[cut:])
        train = scaler.transform(np.vstack(fit_parts))
        calibration = scaler.transform(np.vstack(calibration_parts))
        if train.shape[0] < 10 or calibration.shape[0] < 3:
            continue
        forest = IsolationForest(n_estimators=300, random_state=config.RUL_RANDOM_STATE).fit(train)
        # Threshold from healthy segments the forest did not see (later in time
        # than the fit block, earlier than the test block).
        threshold = float(np.percentile(-forest.score_samples(calibration), 95.0))
        healthy_scores = -forest.score_samples(scaler.transform(np.vstack(healthy_test)))
        entry = {"setup": key, "healthy_train_segments": int(train.shape[0]),
                 "false_alarm_rate": float(np.mean(healthy_scores > threshold)), "labels": {}}
        for info, features in faulty:
            _, test_idx = split_blocks(features.shape[0])
            scores = -forest.score_samples(scaler.transform(features[test_idx]))
            y = np.r_[np.zeros(healthy_scores.shape[0]), np.ones(scores.shape[0])]
            entry["labels"][info["label"]] = {
                "detection_rate": float(np.mean(scores > threshold)),
                "roc_auc": float(roc_auc_score(y, np.r_[healthy_scores, scores])),
            }
        results.append(entry)
    return results


def validate(recordings_features):
    """
    recordings_features: list of (recording_describe_dict, features, names).
    Only recordings with the same feature layout (SEU or CWRU) are combined.
    """
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler

    groups = {}
    for info, features, names in recordings_features:
        info = dict(info)
        groups.setdefault(tuple(names), []).append((info, features))
    reports = []
    for names, items in groups.items():
        x_train, y_train, x_test, y_test = [], [], [], []
        for info, features in items:
            train_idx, test_idx = split_blocks(features.shape[0])
            x_train.append(features[train_idx]); y_train += [info["label"]] * len(train_idx)
            x_test.append(features[test_idx]); y_test += [info["label"]] * len(test_idx)
        x_train = np.vstack(x_train); x_test = np.vstack(x_test)
        y_train = np.array(y_train); y_test = np.array(y_test)
        labels = sorted(set(y_train) | set(y_test))
        source = "CWRU" if len(names) < 20 else "SEU"
        report = {"source": source, "recordings": len(items), "features": len(names),
                  "train_segments": int(x_train.shape[0]), "test_segments": int(x_test.shape[0]),
                  "labels": labels}

        scaler = StandardScaler().fit(x_train)
        xs_train = scaler.transform(x_train)
        xs_test = scaler.transform(x_test)
        if len(labels) >= 2:
            classifier = RandomForestClassifier(n_estimators=300, random_state=config.RUL_RANDOM_STATE,
                                                n_jobs=1).fit(xs_train, y_train)
            predicted = classifier.predict(xs_test)
            report["accuracy"] = float(accuracy_score(y_test, predicted))
            report["macro_f1"] = float(f1_score(y_test, predicted, average="macro"))
            report["confusion"] = confusion_matrix(y_test, predicted, labels=labels).tolist()
            ranking = np.argsort(classifier.feature_importances_)[::-1][:8]
            report["top_features"] = [(names[i], float(classifier.feature_importances_[i]))
                                      for i in ranking]

        report["anomaly"] = _anomaly_by_setup(items, scaler)
        reports.append(report)
    return reports
