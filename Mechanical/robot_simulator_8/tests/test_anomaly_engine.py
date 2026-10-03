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
    assert report["false_alarm_rate"] <= report["window_false_alarm_rate"] < 0.15
    assert report["labels"]["DAMAGED"]["detection_rate"] > 0.9
    assert report["labels"]["DAMAGED"]["roc_auc"] > 0.95


def test_a_fault_in_a_single_feature_is_detected(model):
    """One feature 15 std out. The forest alone caught ~70%; the single feature check closes it."""
    faulty = healthy(100, 2) + [0.0, 3.0, 0.0]
    results = model.score(faulty)
    assert np.mean([r["anomalous"] for r in results]) > 0.9
    assert np.mean([r["feature_alarm"] for r in results]) > 0.9


def test_single_feature_check_can_be_switched_off(monkeypatch):
    import config
    monkeypatch.setattr(config, "ANOMALY_FEATURE_CHECK", False)
    forest_only = AnomalyModel("rig A", NAMES, n_estimators=50).fit(healthy(300, 0))
    assert forest_only.feature_threshold is None
    assert not any(r["feature_alarm"] for r in forest_only.score(healthy(100, 2) + [0.0, 3.0, 0.0]))


def test_single_feature_threshold_is_never_below_the_minimum(model):
    import config
    assert model.feature_threshold >= config.ANOMALY_FEATURE_Z_MIN


def test_too_little_healthy_data_is_a_clear_error():
    with pytest.raises(ValueError, match="healthy windows"):
        AnomalyModel("rig A", NAMES).fit(healthy(8, 0))


# -- persistence ------------------------------------------------------------

def test_persistence_needs_two_flags_in_a_row():
    from predictive.anomaly_engine import persistent_alarms
    flags = [True, False, True, True, True, False, True]
    assert persistent_alarms(flags, 2).tolist() == [False, False, False, True, True, False, False]
    assert persistent_alarms(flags, 3).tolist() == [False, False, False, False, True, False, False]
    assert persistent_alarms(flags, 1).tolist() == flags


def test_an_isolated_odd_window_is_flagged_but_raises_no_alarm(model):
    windows = healthy(40, 5)
    windows[20] += [1.0, 3.0, 0.0]
    results = model.score(windows)
    assert results[20]["anomalous"] and not results[20]["alarm"]


def test_a_lasting_fault_raises_alarms_after_one_window(model):
    windows = healthy(40, 5)
    windows[20:] += [1.0, 3.0, 0.0]
    results = model.score(windows)
    assert not results[20]["alarm"] and all(r["alarm"] for r in results[21:])


def test_persistence_does_not_carry_over_between_recordings(model):
    faulty = healthy(10, 6) + [1.0, 3.0, 0.0]
    report = model.evaluate(healthy(50, 3), {"DAMAGED": [faulty, faulty]})
    assert report["labels"]["DAMAGED"]["window_detection_rate"] == 1.0
    assert report["labels"]["DAMAGED"]["detection_rate"] == pytest.approx(18 / 20)   # first window of each
