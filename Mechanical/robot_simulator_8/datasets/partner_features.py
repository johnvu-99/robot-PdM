"""
Window features for partner channels, chosen by channel type.

    every channel        mean, std, rms, min, max, p2p, kurtosis, skewness, trend
    vibration, current,  plus dominant frequency, spectral centroid, band energies
    torque, force        (only when the sample rate is high enough)
    temperature          mean, max, C per minute, rise since start, rise over
                         ambient, rise per watt, over a longer trailing window
    ambient_temperature  mean only (used as the reference for temperature)
    position + target    tracking error channel derived as target - position

Pure numpy. Features are only computed for channels that exist: nothing is
invented for a sensor the partner did not install.
"""

import numpy as np

import config

GENERIC = ("mean", "std", "rms", "min", "max", "p2p", "kurtosis", "skewness", "trend")

# Temperature is judged by level and by how fast and how far it has moved,
# always over the longer temperature window.
THERMAL = ("mean", "max", "rate_c_per_min", "rise_since_start")
THERMAL_AMBIENT = "rise_over_ambient"     # added when an ambient channel exists
THERMAL_PER_POWER = "rise_per_watt"       # added when a power channel exists
SPECTRAL = ("dominant_frequency", "spectral_centroid") + tuple(
    "band_energy_%d" % i for i in range(len(config.SPECTRAL_BANDS)))
SPECTRAL_TYPES = ("vibration", "current", "torque", "force")


def thermal_feature_names(channel_name, has_ambient, has_power):
    names = ["%s_%s" % (channel_name, s) for s in THERMAL]
    if has_ambient:
        names.append("%s_%s" % (channel_name, THERMAL_AMBIENT))
    if has_power:
        names.append("%s_%s" % (channel_name, THERMAL_PER_POWER))
    return names


def thermal_features(values, times, start_value, ambient=None, power=None):
    """
    values/times: the trailing temperature window. start_value: the channel's
    value at the beginning of this recording (causal: known at run time).
    Returns values aligned with thermal_feature_names().
    """
    mean = float(np.mean(values))
    tc = times - times.mean()
    denominator = float(np.sum(tc * tc))
    rate = float(np.sum(tc * (values - mean)) / denominator) if denominator > 0 else 0.0
    out = [mean, float(np.max(values)), 60.0 * rate, mean - float(start_value)]
    if ambient is not None:
        out.append(mean - float(np.mean(ambient)))
    if power is not None:
        # Temperature rise per watt of mechanical power: rises when cooling
        # degrades, and is not fooled by a heavier duty cycle.
        watts = float(np.mean(np.abs(power)))
        reference = out[4] if ambient is not None else out[3]
        out.append(reference / watts if watts > 1e-6 else 0.0)
    return out


def channel_feature_names(channel_name, channel_type, sample_rate):
    if channel_type == "ambient_temperature":
        return ["%s_mean" % channel_name]
    names = ["%s_%s" % (channel_name, s) for s in GENERIC]
    if channel_type in SPECTRAL_TYPES and sample_rate and sample_rate >= config.PARTNER_SPECTRAL_MIN_RATE_HZ:
        names.extend("%s_%s" % (channel_name, s) for s in SPECTRAL)
    return names


def _generic(x, t):
    mean = float(np.mean(x))
    centered = x - mean
    std = float(np.std(x))
    if std > 1e-12:
        z = centered / std
        kurtosis = float(np.mean(z ** 4))
        skewness = float(np.mean(z ** 3))
    else:
        kurtosis = skewness = 0.0
    tc = t - t.mean()
    denominator = float(np.sum(tc * tc))
    trend = float(np.sum(tc * centered) / denominator) if denominator > 0 else 0.0
    return {"mean": mean, "std": std, "rms": float(np.sqrt(np.mean(x * x))),
            "min": float(np.min(x)), "max": float(np.max(x)), "p2p": float(np.ptp(x)),
            "kurtosis": kurtosis, "skewness": skewness, "trend": trend}


def _spectral(x, sample_rate):
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.shape[0]))) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], 1.0 / sample_rate)
    spectrum[0] = 0.0
    total = float(spectrum.sum())
    out = {"dominant_frequency": 0.0, "spectral_centroid": 0.0}
    if total > 0.0:
        out["dominant_frequency"] = float(freqs[int(np.argmax(spectrum))])
        out["spectral_centroid"] = float(np.sum(freqs * spectrum) / total)
    nyquist = 0.5 * sample_rate
    for i, (low, high) in enumerate(config.SPECTRAL_BANDS):
        upper = freqs <= nyquist if high >= 1.0 else freqs < high * nyquist
        mask = (freqs >= low * nyquist) & upper
        out["band_energy_%d" % i] = float(spectrum[mask].sum() / total) if total > 0 else 0.0
    return out


def window_features(signals, times, channels, sample_rate):
    """
    signals: dict name -> 1 D array for this window; channels: list of (name, type).
    Returns an ordered list of (feature_name, value).
    """
    values = []
    for name, channel_type in channels:
        x = signals[name]
        stats = _generic(x, times)
        names = channel_feature_names(name, channel_type, sample_rate)
        spectral = None
        for full in names:
            key = full[len(name) + 1:]
            if key in stats:
                values.append((full, stats[key]))
            else:
                if spectral is None:
                    spectral = _spectral(x, sample_rate)
                values.append((full, spectral[key]))
    return values
