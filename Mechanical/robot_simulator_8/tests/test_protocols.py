"""Protocols: which bearings train and which test, and what each protocol switches on."""

import pytest

import config
from predictive.dataset_jobs import JobContext, _protocol_runs, run_job
from tests.conftest import write_phm_folder, write_xjtu_folder


@pytest.fixture
def roots(tmp_path):
    return {
        "PHM2012": write_phm_folder(tmp_path / "phm", {
            "Learning_set": ["Bearing1_1", "Bearing1_2", "Bearing2_1"],
            "Full_Test_Set": ["Bearing1_3", "Bearing2_3"]}),
        "XJTU_SY": write_xjtu_folder(tmp_path / "xjtu", {
            "35Hz12kN": ["Bearing1_1", "Bearing1_2", "Bearing1_3"]}),
    }


def split(protocol, roots):
    _, train, test = _protocol_runs(JobContext(None, None), {"protocol": protocol, "roots": roots})
    return sorted(r["run_id"] for r in train), sorted(r["run_id"] for r in test)


def test_xjtu_to_phm_trains_on_xjtu_and_tests_on_every_phm_bearing(roots):
    train, test = split("XJTU_TO_PHM", roots)
    assert train == ["XJTU_Bearing1_1", "XJTU_Bearing1_2", "XJTU_Bearing1_3"]
    assert test == ["PHM_Bearing1_1", "PHM_Bearing1_2", "PHM_Bearing1_3",
                    "PHM_Bearing2_1", "PHM_Bearing2_3"]


def test_phm_protocol_trains_on_learning_set_and_tests_on_full_test_set(roots):
    train, test = split("PHM_LEARNING_TO_FULL_TEST", roots)
    assert train == ["PHM_Bearing1_1", "PHM_Bearing1_2", "PHM_Bearing2_1"]
    assert test == ["PHM_Bearing1_3", "PHM_Bearing2_3"]


def test_combined_protocol_trains_on_both_and_keeps_the_test_bearings_out(roots):
    train, test = split("XJTU_PLUS_PHM_TO_FULL_TEST", roots)
    assert len(train) == 6 and test == ["PHM_Bearing1_3", "PHM_Bearing2_3"]
    assert not set(train) & set(test)


@pytest.mark.parametrize("protocol", list(config.RUL_PROTOCOLS))
def test_no_protocol_shares_a_bearing_between_train_and_test(protocol, roots):
    train, test = split(protocol, roots)
    assert train and test and not set(train) & set(test)


def test_every_protocol_has_its_settings_defined():
    for protocol in config.RUL_PROTOCOLS:
        assert protocol in config.RUL_LOUDNESS_REFERENCE_BY_PROTOCOL
        assert protocol in config.RUL_FPT_GATE_BY_PROTOCOL
    assert config.RUL_DEFAULT_PROTOCOL in config.RUL_PROTOCOLS


def test_unknown_protocol_and_missing_data_are_clear_errors(roots, tmp_path):
    with pytest.raises(ValueError, match="unknown protocol"):
        split("NOT_A_PROTOCOL", roots)
    with pytest.raises(ValueError, match="XJTU-SY not found"):
        split("XJTU_TO_PHM", dict(roots, XJTU_SY=str(tmp_path / "missing")))


def test_train_then_evaluate_runs_end_to_end(roots, tmp_path):
    path = str(tmp_path / "model.joblib")
    trained = run_job("train_rul", {"protocol": "XJTU_TO_PHM", "model": "LINEAR_REGRESSION",
                                    "roots": roots, "path": path})
    assert trained["model_path"] == path and len(trained["cv"]) == 3
    evaluated = run_job("evaluate_rul", {"protocol": "XJTU_TO_PHM", "model_path": path, "roots": roots})
    assert len(evaluated["metrics"]) == 5
    assert 0.0 <= evaluated["summary"]["mae"] <= 1.0
    assert trained["interval"]["bearings"] == 3 and trained["interval"]["half_width"] > 0.0
    assert 0.0 <= evaluated["summary"]["coverage"] <= 1.0
    assert set(evaluated["curves"][0]) >= {"rul_low", "rul_high", "predicted_rul"}


def test_evaluating_a_model_on_a_bearing_it_trained_on_is_refused(roots, tmp_path):
    path = str(tmp_path / "model.joblib")
    run_job("train_rul", {"protocol": "XJTU_PLUS_PHM_TO_FULL_TEST", "model": "LINEAR_REGRESSION",
                          "roots": roots, "path": path})
    # XJTU_TO_PHM validates on ALL PHM bearings, three of which this model trained on.
    assert run_job("evaluate_rul", {"protocol": "XJTU_TO_PHM", "model_path": path, "roots": roots}) is None
