"""Vibration features: known signals must give known numbers."""

import numpy as np

from datasets.signal_features import (
    band_names, vibration_feature_names, vibration_features,
)

RATE = 25600.0


def features_of(signal):
    return dict(zip(vibration_feature_names("x"), vibration_features(signal, RATE)))


def sine(frequency, amplitude=2.0, seconds=0.5):
    t = np.arange(int(RATE * seconds)) / RATE
    return amplitude * np.sin(2 * np.pi * frequency * t)


def test_names_and_values_line_up():
    assert len(vibration_feature_names("x")) == len(vibration_features(sine(100.0), RATE))


def test_sine_wave_has_textbook_values():
    f = features_of(sine(1000.0, amplitude=2.0))
    assert abs(f["x_rms"] - 2.0 / np.sqrt(2)) < 0.01
    assert abs(f["x_p2p"] - 4.0) < 0.01
    assert abs(f["x_crest_factor"] - np.sqrt(2)) < 0.01
    assert abs(f["x_kurtosis"] - 1.5) < 0.01          # a pure sine: 1.5, noise: 3
    assert abs(f["x_dominant_frequency"] - 1000.0) < 5.0


def test_band_energies_are_shares_that_sum_to_one():
    f = features_of(sine(1000.0) + 0.5 * sine(8000.0))
    shares = [f["x_%s" % name] for name in band_names()]
    assert abs(sum(shares) - 1.0) < 1e-6
    assert all(0.0 <= share <= 1.0 for share in shares)


def test_impacts_raise_kurtosis_but_barely_change_rms():
    rng = np.random.RandomState(0)
    smooth = rng.randn(8192)
    knocking = smooth.copy()
    knocking[::512] += 15.0                              # short knocks, like a cracked race
    assert features_of(knocking)["x_kurtosis"] > 2 * features_of(smooth)["x_kurtosis"]
    assert features_of(knocking)["x_rms"] < 1.5 * features_of(smooth)["x_rms"]


def test_constant_signal_gives_no_nan():
    assert np.all(np.isfinite(vibration_features(np.ones(1024), RATE)))
