"""Dataset adapters: find the right files, in the right order, with the right labels."""

import os

import numpy as np

import config
from datasets.phm2012 import PHM2012DatasetAdapter
from datasets.xjtu_sy import XJTUSYDatasetAdapter
from tests.conftest import write_phm_folder, write_xjtu_folder


def test_phm_uses_learning_and_full_test_but_not_the_truncated_test_set(tmp_path):
    root = write_phm_folder(tmp_path / "phm", {
        "Learning_set": ["Bearing1_1"], "Full_Test_Set": ["Bearing1_3"], "Test_set": ["Bearing1_3"]})
    recordings = PHM2012DatasetAdapter(root).discover()
    assert sorted((r.run_id, r.meta["subset"]) for r in recordings) == [
        ("PHM_Bearing1_1", "Learning_set"), ("PHM_Bearing1_3", "Full_Test_Set")]


def test_phm_ignores_temperature_files_and_reads_the_two_acceleration_columns(tmp_path):
    root = write_phm_folder(tmp_path / "phm", {"Learning_set": ["Bearing2_1"]}, snapshots=5)
    adapter = PHM2012DatasetAdapter(root)
    recording = adapter.discover()[0]
    assert len(recording.files) == 5
    assert all(os.path.basename(f).startswith("acc_") for f in recording.files)
    assert adapter.read_snapshot(recording.files[0]).shape == (64, 2)
    assert recording.condition == "1650rpm_4200N"


def test_run_labels_go_from_new_to_failed(tmp_path):
    root = write_phm_folder(tmp_path / "phm", {"Learning_set": ["Bearing1_1"]}, snapshots=12)
    adapter = PHM2012DatasetAdapter(root)
    run = adapter.extract_run(adapter.discover()[0])
    assert run["features"].shape == (12, len(adapter.feature_names()))
    assert np.allclose(np.diff(run["times"]), config.PHM_SNAPSHOT_INTERVAL_S)
    assert run["rul_fraction"][0] == 1.0 and run["rul_fraction"][-1] == 0.0
    assert np.all(np.diff(run["rul_fraction"]) < 0)
    assert np.allclose(run["life_fraction"] + run["rul_fraction"], 1.0)


def test_xjtu_sorts_files_by_number_and_skips_the_header(tmp_path):
    root = write_xjtu_folder(tmp_path / "xjtu", {"35Hz12kN": ["Bearing1_1"]}, snapshots=12)
    adapter = XJTUSYDatasetAdapter(root)
    recording = adapter.discover()[0]
    assert [os.path.basename(f) for f in recording.files] == ["%d.csv" % i for i in range(1, 13)]
    assert recording.condition == "35Hz12kN"
    assert adapter.read_snapshot(recording.files[0]).shape == (64, 2)
    run = adapter.extract_run(recording)
    assert np.allclose(np.diff(run["times"]), config.XJTU_SNAPSHOT_INTERVAL_S)


def test_features_are_cached_in_the_processed_folder_and_reused(tmp_path):
    root = write_phm_folder(tmp_path / "phm", {"Learning_set": ["Bearing1_1"]}, snapshots=6)
    adapter = PHM2012DatasetAdapter(root)
    recording = adapter.discover()[0]
    first = adapter.extract_run(recording)
    cache = adapter.cache_path(recording, "run")
    assert cache.startswith(str(tmp_path)) and os.path.isfile(cache)
    assert np.array_equal(first["features"], adapter.extract_run(recording)["features"])


def test_missing_dataset_folder_is_reported_not_crashed(tmp_path):
    adapter = XJTUSYDatasetAdapter(str(tmp_path / "nothing_here"))
    assert not adapter.available()
    assert adapter.discover() == []
