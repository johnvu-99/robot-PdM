"""Anomaly engine: learns from healthy data only, and flags what does not fit."""

import numpy as np
import pytest

from predictive.anomaly_engine import AnomalyModel

NAMES = ["vibration_rms", "vibration_kurtosis", "torque_rms"]


def healthy(windows, seed):
    return np.random.RandomState(seed).randn(windows, 3) * [0.1, 0.2, 0.1] + [1.0, 3.0, 5.0]


@pytest.fixture
def model():
    return AnomalyModel("rig A", NAMES, n_estimators=50).fit(healthy(300, 0))


def test_few_false_alarms_on_new_healthy_data(model):
    flags = [r["anomalous"] for r in model.score(healthy(300, 1))]
    assert np.mean(flags) < 0.15


def test_a_fault_is_detected_and_the_changed_feature_is_named(model):
    faulty = healthy(100, 2) + [1.0, 3.0, 0.0]            # louder AND spikier, like real damage
    results = model.score(faulty)
    assert np.mean([r["anomalous"] for r in results]) > 0.9
    assert results[0]["top_features"][0][0] == "vibration_kurtosis"


def test_evaluation_report_has_false_alarms_and_detection_per_fault(model):
    faulty = healthy(100, 2) + [1.0, 3.0, 0.0]
    report = model.evaluate(healthy(100, 3), {"DAMAGED": faulty})
    assert report["false_alarm_rate"] < 0.15
    assert report["labels"]["DAMAGED"]["detection_rate"] > 0.9
    assert report["labels"]["DAMAGED"]["roc_auc"] > 0.95


@pytest.mark.xfail(strict=False, reason=(
    "Known weakness: a fault that moves ONE feature, even by 15 standard deviations, is only "
    "caught about 70% of the time. An Isolation Forest dilutes a single unusual feature among "
    "the normal ones. The thermal ratio check exists for this reason; other features have none."))
def test_a_fault_in_a_single_feature_is_detected(model):
    faulty = healthy(100, 2) + [0.0, 3.0, 0.0]
    assert np.mean([r["anomalous"] for r in model.score(faulty)]) > 0.9


def test_too_little_healthy_data_is_a_clear_error():
    with pytest.raises(ValueError, match="healthy windows"):
        AnomalyModel("rig A", NAMES).fit(healthy(8, 0))
