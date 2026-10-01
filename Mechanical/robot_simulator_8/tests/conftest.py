"""
Shared test setup.

Every test writes its caches and model files to a temporary folder, never to
the real data/processed or models folders. No test needs the downloaded
datasets: small synthetic ones are built on the fly.
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from datasets.signal_features import vibration_feature_names  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setattr(config, "MODEL_DIR", str(tmp_path / "models"))


def make_run(run_id, dataset="XJTU_SY", snapshots=200, interval_s=60.0, wear_from=0.6,
             run_in=0.0, seed=0):
    """
    A synthetic run to failure in the format the adapters produce.

    Loudness (rms, p2p, spectral energy) is flat, then grows from wear_from of
    life onwards. run_in > 0 makes the bearing start loud and settle, like the
    PHM 2012 bearings that get quieter after the first minutes.
    """
    rng = np.random.RandomState(seed)
    names = vibration_feature_names("horizontal") + vibration_feature_names("vertical")
    life = np.linspace(0.0, 1.0, snapshots)
    wear = np.exp(3.0 * np.clip(life - wear_from, 0.0, None))
    settle = 1.0 + run_in * np.exp(-life / 0.05)
    loudness = wear * settle
    columns = []
    for name in names:
        noise = 1.0 + 0.01 * rng.randn(snapshots)
        if name.endswith(("_rms", "_std", "_p2p")):
            columns.append(loudness * noise)
        elif name.endswith(("_energy", "_spectral_energy")):
            columns.append(loudness ** 2 * noise)
        elif name.endswith("_kurtosis"):
            columns.append(3.0 * (1.0 + 0.5 * np.clip(life - wear_from, 0.0, None)) * noise)
        elif "band_energy" in name:
            columns.append(0.25 * noise)
        else:
            columns.append(noise)
    times = np.arange(snapshots) * interval_s
    return {
        "run_id": run_id, "dataset": dataset, "condition": "test",
        "times": times, "features": np.column_stack(columns), "names": names,
        "life_fraction": life, "rul_fraction": 1.0 - life, "total_life_s": float(times[-1]),
    }


def truncate(run, snapshots):
    """The same run as it would look if it had only been recorded this far."""
    out = dict(run)
    for key in ("times", "features", "life_fraction", "rul_fraction"):
        out[key] = run[key][:snapshots]
    return out


def write_phm_folder(root, bearings, snapshots=12, rows=64):
    """bearings: {"Learning_set": ["Bearing1_1", ...], ...}. PHM acc_*.csv layout."""
    rng = np.random.RandomState(1)
    for subset, names in bearings.items():
        for name in names:
            folder = os.path.join(str(root), subset, name)
            os.makedirs(folder)
            for index in range(1, snapshots + 1):
                data = np.column_stack([np.full(rows, 9), np.full(rows, 39), np.full(rows, 39),
                                        np.arange(rows), rng.randn(rows) * index, rng.randn(rows)])
                np.savetxt(os.path.join(folder, "acc_%05d.csv" % index), data, delimiter=",", fmt="%.6f")
            with open(os.path.join(folder, "temp_00001.csv"), "w") as handle:
                handle.write("9,39,39,0,25.0\n")
    return str(root)


def write_xjtu_folder(root, bearings, snapshots=12, rows=64):
    """bearings: {"35Hz12kN": ["Bearing1_1", ...]}. XJTU-SY 1.csv, 2.csv, ... with a header row."""
    rng = np.random.RandomState(2)
    for condition, names in bearings.items():
        for name in names:
            folder = os.path.join(str(root), condition, name)
            os.makedirs(folder)
            for index in range(1, snapshots + 1):
                data = np.column_stack([rng.randn(rows) * index, rng.randn(rows)])
                np.savetxt(os.path.join(folder, "%d.csv" % index), data, delimiter=",", fmt="%.6f",
                           header="Horizontal_vibration_signals,Vertical_vibration_signals", comments="")
    return str(root)
