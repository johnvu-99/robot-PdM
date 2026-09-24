"""
Condition monitoring features for raw vibration and torque segments.

Pure numpy. Time domain: RMS, std, peak to peak, crest factor, kurtosis,
skewness, energy. Frequency domain: dominant frequency, spectral centroid,
band energies. Torque: mean, RMS, std, peak, trend.
"""

import numpy as np

import config

VIBRATION_STATS = ("rms", "std", "p2p", "crest_factor", "kurtosis", "skewness", "energy",
                   "dominant_frequency", "spectral_centroid", "spectral_energy")
TORQUE_STATS = ("mean", "rms", "std", "peak", "trend")


def band_names():
    return tuple("band_energy_%d" % i for i in range(len(config.SPECTRAL_BANDS)))


def vibration_feature_names(prefix):
    return tuple("%s_%s" % (prefix, s) for s in VIBRATION_STATS + band_names())


def torque_feature_names(prefix):
    return tuple("%s_%s" % (prefix, s) for s in TORQUE_STATS)


def vibration_features(signal, sample_rate):
    """signal: 1 D array. Returns a tuple aligned with vibration_feature_names()."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    std = x.std()
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x))) if n else 0.0
    if std > 1e-12:
        z = x / std
        kurtosis = float(np.mean(z ** 4))
        skewness = float(np.mean(z ** 3))
    else:
        kurtosis = 0.0
        skewness = 0.0
    energy = float(np.sum(x * x))

    spectrum = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spectrum[0] = 0.0
    total = float(spectrum.sum())
    if total > 0.0:
        dominant = float(freqs[int(np.argmax(spectrum))])
        centroid = float(np.sum(freqs * spectrum) / total)
    else:
        dominant = 0.0
        centroid = 0.0
    nyquist = 0.5 * sample_rate
    bands = []
    for low, high in config.SPECTRAL_BANDS:
        mask = (freqs >= low * nyquist) & (freqs < high * nyquist if high < 1.0 else freqs <= nyquist)
        bands.append(float(spectrum[mask].sum() / total) if total > 0.0 else 0.0)
    return (rms, float(std), float(x.max() - x.min()) if n else 0.0,
            peak / rms if rms > 1e-12 else 0.0, kurtosis, skewness, energy,
            dominant, centroid, total) + tuple(bands)


def torque_features(signal, sample_rate):
    x = np.asarray(signal, dtype=np.float64)
    n = x.shape[0]
    t = np.arange(n) / sample_rate
    tc = t - t.mean()
    denominator = float(np.sum(tc * tc))
    trend = float(np.sum(tc * (x - x.mean())) / denominator) if denominator > 0 else 0.0
    return (float(x.mean()), float(np.sqrt(np.mean(x * x))), float(x.std()),
            float(np.max(np.abs(x))), trend)


def parse_numeric_text(text, columns):
    """
    Fast parse of delimited numbers (comma, tab or semicolon, optional trailing
    separator). Returns (rows, columns). Partial trailing rows are dropped.
    """
    cleaned = text.replace("\t", ",").replace(";", ",").replace("\r", "")
    lines = cleaned.split("\n")
    if len(lines) > 1 and not cleaned.endswith("\n"):
        # The final line has no terminator: it may be cut mid number.
        lines = lines[:-1]
    rows = []
    for line in lines:
        line = line.strip().rstrip(",")
        if line:
            rows.append(line)
    if not rows:
        return np.zeros((0, columns))
    flat = np.array(",".join(rows).split(","), dtype=np.float64)
    usable = (flat.shape[0] // columns) * columns
    return flat[:usable].reshape(-1, columns)
