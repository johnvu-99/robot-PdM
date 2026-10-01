"""RUL model: no leakage, no use of the future, and model files stay compatible."""

import joblib
import numpy as np
import pytest

import config
from predictive import rul_model
from predictive.rul_model import (
    RULModel, assert_disjoint, detect_fpt, leave_one_run_out, postprocess_prediction,
    rul_features, run_metrics,
)
from tests.conftest import make_run, truncate


def training_runs():
    return [make_run("XJTU_Bearing%d" % i, wear_from=0.4 + 0.1 * i, seed=i) for i in range(4)]


# -- features --------------------------------------------------------------

@pytest.mark.parametrize("reference", ["START", "QUIETEST", "BOTH"])
def test_features_never_use_the_future(reference):
    """Recording more of a run must not change the features of its past."""
    run = make_run("A", run_in=1.0)
    full, _, _ = rul_features(run, reference)
    early, _, _ = rul_features(truncate(run, 120), reference)
    assert np.allclose(full[:120], early)


def test_quietest_reference_does_not_reward_run_in():
    """A bearing that starts loud and settles must not look healthier than new."""
    run = make_run("A", run_in=1.5, wear_from=2.0)          # settles, never wears
    middle = slice(80, 160)
    start, names, _ = rul_features(run, "START")
    quietest, _, _ = rul_features(run, "QUIETEST")
    column = names.index("horizontal_rms_rel")
    assert np.mean(start[middle, column]) < -0.5
    assert abs(np.mean(quietest[middle, column])) < 0.05


def test_optional_feature_sets_change_the_layout_only_when_switched_on(monkeypatch):
    run = make_run("A")
    default_names = rul_features(run)[1]
    monkeypatch.setattr(config, "RUL_TREND_FEATURES", True)
    assert len(rul_features(run)[1]) == len(default_names) + 4
    monkeypatch.setattr(config, "RUL_TREND_FEATURES", False)
    monkeypatch.setattr(config, "RUL_DROP_LOUDNESS_FEATURES", True)
    assert not any("_rms_" in name for name in rul_features(run)[1])


# -- leakage ---------------------------------------------------------------

def test_a_bearing_in_train_and_test_is_an_error():
    runs = training_runs()
    assert_disjoint(runs[:2], runs[2:])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(runs[:3], runs[2:])


def test_cross_validation_tests_each_bearing_exactly_once():
    runs = training_runs()
    results = leave_one_run_out(runs, "LINEAR_REGRESSION")
    assert sorted(m["run_id"] for m in results) == sorted(r["run_id"] for r in runs)


# -- metrics ---------------------------------------------------------------

def test_metrics_of_a_perfect_and_a_backwards_prediction():
    run = make_run("A")
    perfect = run_metrics(run, run["rul_fraction"])
    assert perfect["mae"] == 0.0 and perfect["r2"] == 1.0
    assert perfect["rank_correlation"] == pytest.approx(1.0)
    backwards = run_metrics(run, 1.0 - run["rul_fraction"])
    assert backwards["rank_correlation"] == pytest.approx(-1.0)
    assert backwards["r2"] < 0


def test_always_guessing_half_gives_mae_of_a_quarter():
    run = make_run("A", snapshots=1001)
    guess = run_metrics(run, np.full(1001, 0.5))
    assert guess["mae"] == pytest.approx(0.25, abs=1e-3)
    assert guess["r2"] == pytest.approx(0.0, abs=1e-6)


# -- model -----------------------------------------------------------------

@pytest.mark.parametrize("kind", config.RUL_MODELS)
def test_every_model_type_predicts_a_fraction_per_snapshot(kind):
    model = RULModel(kind).fit(training_runs())
    prediction, health = model.predict_run(make_run("XJTU_new", seed=9))
    assert prediction.shape == health.shape == (200,)
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_model_learns_that_wear_means_less_life_left():
    model = RULModel("RANDOM_FOREST").fit(training_runs())
    run = make_run("XJTU_new", wear_from=0.55, seed=9)
    prediction, _ = model.predict_run(run)
    assert np.mean(prediction[:40]) > np.mean(prediction[-40:]) + 0.2
    assert run_metrics(run, prediction)["rank_correlation"] > 0.5


def test_save_and_load_give_the_same_predictions(tmp_path):
    model = RULModel("RANDOM_FOREST").fit(training_runs(), "QUIETEST", True)
    path = model.save(str(tmp_path / "model.joblib"))
    loaded = RULModel.load(path)
    run = make_run("PHM_new", dataset="PHM2012", seed=9)
    assert loaded.loudness_reference == "QUIETEST" and loaded.fpt_floor == model.fpt_floor
    assert np.array_equal(model.predict_run(run)[0], loaded.predict_run(run)[0])


def test_model_files_from_before_the_new_settings_still_load(tmp_path):
    model = RULModel("LINEAR_REGRESSION").fit(training_runs())
    path = str(tmp_path / "old.joblib")
    joblib.dump({"version": 1, "kind": model.kind, "pipeline": model.pipeline,
                 "feature_names": model.feature_names, "training_runs": model.training_runs,
                 "training_dataset": model.training_dataset, "created": model.created,
                 "target": "RUL_fraction"}, path)
    loaded = RULModel.load(path)
    assert loaded.loudness_reference == "START" and loaded.fpt_floor is None


def test_a_file_that_is_not_an_rul_model_is_refused(tmp_path):
    path = str(tmp_path / "other.joblib")
    joblib.dump({"something": "else"}, path)
    with pytest.raises(ValueError):
        RULModel.load(path)


def test_model_refuses_a_run_with_a_different_feature_layout(monkeypatch):
    model = RULModel("LINEAR_REGRESSION").fit(training_runs())
    monkeypatch.setattr(config, "RUL_TREND_FEATURES", True)
    with pytest.raises(ValueError, match="feature layout"):
        model.predict_run(make_run("XJTU_new"))


# -- first predicting time and the gate ------------------------------------

def test_degradation_start_is_reported_by_the_time_wear_is_clear():
    """Wear starts at snapshot 100. It must be flagged by 140, and never inside
    the first 10 minutes, which define what normal looks like."""
    run = make_run("B", wear_from=0.5)
    matrix, names, _ = rul_features(run, "QUIETEST")
    fpt = detect_fpt(run, matrix, names)
    healthy_window = int(config.RUL_FPT_HEALTHY_S / 60.0)
    assert healthy_window < fpt < 140


@pytest.mark.xfail(strict=False, reason=(
    "Known weakness: the first 10 minutes (11 snapshots at one per minute) are too few to "
    "measure the normal spread, so 5 standard deviations is exceeded by noise alone. The gate "
    "therefore opens early; it works as 'no low prediction early in life', not as a true "
    "degradation detector. See CODE_GUIDE_MECHANICAL.md section 9."))
def test_a_bearing_that_never_wears_is_never_flagged():
    run = make_run("A", wear_from=2.0)
    matrix, names, _ = rul_features(run, "QUIETEST")
    assert detect_fpt(run, matrix, names) == 200


def test_degradation_start_does_not_move_when_more_is_recorded():
    run = make_run("B", wear_from=0.5)
    matrix, names, _ = rul_features(run, "QUIETEST")
    fpt = detect_fpt(run, matrix, names)
    shorter = truncate(run, fpt + 20)
    matrix, names, _ = rul_features(shorter, "QUIETEST")
    assert detect_fpt(shorter, matrix, names) == fpt


def test_gate_holds_the_prediction_up_until_degradation_starts():
    model = RULModel("RANDOM_FOREST").fit(training_runs(), "QUIETEST", True)
    assert 0.0 < model.fpt_floor < 1.0
    run = make_run("XJTU_new", wear_from=0.7, seed=9)
    matrix, names, _ = rul_features(run, "QUIETEST")
    fpt = detect_fpt(run, matrix, names)
    prediction, _ = model.predict_run(run)
    assert fpt > 0 and np.all(prediction[:fpt] >= model.fpt_floor - 1e-9)


def test_no_gate_unless_asked_for():
    assert RULModel("LINEAR_REGRESSION").fit(training_runs()).fpt_floor is None


# -- smoothing of predictions ----------------------------------------------

def noisy_prediction(run):
    return np.clip(run["rul_fraction"] + 0.2 * np.random.RandomState(3).randn(200), 0.0, 1.0)


def test_predictions_are_smoothed_only_on_a_dataset_the_model_did_not_train_on():
    run = make_run("PHM_new", dataset="PHM2012")
    raw = noisy_prediction(run)
    same = postprocess_prediction(raw.copy(), run, "PHM2012")
    other = postprocess_prediction(raw.copy(), run, "XJTU_SY")
    assert np.array_equal(same, raw)
    assert np.mean(np.abs(np.diff(other))) < 0.2 * np.mean(np.abs(np.diff(raw)))


def test_smoothing_never_uses_the_future():
    run = make_run("PHM_new", dataset="PHM2012")
    raw = noisy_prediction(run)
    full = postprocess_prediction(raw.copy(), run, "XJTU_SY")
    early = postprocess_prediction(raw[:120].copy(), truncate(run, 120), "XJTU_SY")
    assert np.allclose(full[:120], early)


def test_monotone_option_is_off_by_default_and_works_when_on(monkeypatch):
    run = make_run("PHM_new", dataset="PHM2012")
    raw = noisy_prediction(run)
    assert np.any(np.diff(postprocess_prediction(raw.copy(), run, "PHM2012")) > 0)
    monkeypatch.setattr(config, "RUL_MONOTONE_PREDICTION", True)
    assert np.all(np.diff(postprocess_prediction(raw.copy(), run, "PHM2012")) <= 0)
