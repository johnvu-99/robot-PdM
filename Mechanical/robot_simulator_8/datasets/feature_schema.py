"""
Unified feature format across domains.

Simulation telemetry and real vibration datasets measure different things, so
their raw columns are never merged. Each domain keeps its own feature names,
and ConditionFeatures maps them into abstract categories:

    LOAD, MOTION_ERROR, VIBRATION, ENERGY, DEGRADATION, ANOMALY

A feature that a domain does not measure is simply absent (None). Nothing is
imputed: PyBullet has no vibration sensor, and bearing rigs have no tracking
error, and this layer never pretends otherwise.
"""

import numpy as np

DOMAIN_SIMULATION = "SIMULATION"
DOMAIN_REAL_MACHINE = "REAL_MACHINE"

CATEGORIES = ("LOAD", "MOTION_ERROR", "VIBRATION", "ENERGY", "DEGRADATION", "ANOMALY")

SIMULATION_FEATURES = (
    "torque_RMS", "tracking_error_RMS", "velocity_RMS", "acceleration_RMS",
    "power_RMS", "reaction_force_RMS", "torque_trend", "tracking_error_trend",
)

REAL_MACHINE_FEATURES = (
    "vibration_RMS", "vibration_std", "vibration_kurtosis", "vibration_crest_factor",
    "vibration_peak_to_peak", "spectral_energy", "dominant_frequency",
    "torque_RMS", "temperature",
)

# Abstract feature -> category.
CATEGORY_OF = {
    "torque_RMS": "LOAD",
    "reaction_force_RMS": "LOAD",
    "torque_trend": "DEGRADATION",
    "tracking_error_RMS": "MOTION_ERROR",
    "tracking_error_trend": "DEGRADATION",
    "velocity_RMS": "MOTION_ERROR",
    "acceleration_RMS": "MOTION_ERROR",
    "power_RMS": "ENERGY",
    "vibration_RMS": "VIBRATION",
    "vibration_std": "VIBRATION",
    "vibration_kurtosis": "VIBRATION",
    "vibration_crest_factor": "VIBRATION",
    "vibration_peak_to_peak": "VIBRATION",
    "dominant_frequency": "VIBRATION",
    "spectral_energy": "ENERGY",
    "temperature": "LOAD",
    "health_indicator": "DEGRADATION",
    "rul_fraction": "DEGRADATION",
    "anomaly_score": "ANOMALY",
}

# Simulation monitoring feature names (Phase 5) -> abstract names.
SIMULATION_SOURCE = {
    "torque_RMS": "torque_RMS",
    "tracking_error_RMS": "tracking_error_RMS",
    "velocity_RMS": "velocity_5s_rms",
    "acceleration_RMS": "acceleration_RMS",
    "power_RMS": "power_RMS",
    "reaction_force_RMS": "force_RMS",
    "torque_trend": "torque_trend",
    "tracking_error_trend": "tracking_error_trend",
}


class ConditionFeatures(object):
    """Features of one asset at one moment, grouped by category, gaps allowed."""

    def __init__(self, domain, source, values=None):
        if domain not in (DOMAIN_SIMULATION, DOMAIN_REAL_MACHINE):
            raise ValueError("unknown domain %s" % domain)
        self.domain = domain
        self.source = source
        self.values = {}
        for name, value in (values or {}).items():
            self.set(name, value)

    def set(self, name, value):
        if name not in CATEGORY_OF:
            raise KeyError("feature %s has no category" % name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            self.values.pop(name, None)
        else:
            self.values[name] = float(value)

    def get(self, name):
        return self.values.get(name)

    def category(self, category):
        return dict((k, v) for k, v in self.values.items() if CATEGORY_OF[k] == category)

    @property
    def available_categories(self):
        return tuple(c for c in CATEGORIES if self.category(c))

    def vector(self, names):
        """(values, mask). Missing features are NaN with mask False, never filled."""
        values = np.array([self.values.get(n, np.nan) for n in names], dtype=np.float64)
        return values, ~np.isnan(values)

    @classmethod
    def from_simulation(cls, source, monitoring_features):
        values = {}
        for abstract, name in SIMULATION_SOURCE.items():
            if name in monitoring_features:
                values[abstract] = monitoring_features[name]
        return cls(DOMAIN_SIMULATION, source, values)

    @classmethod
    def from_vibration(cls, source, features, channel, torque_prefix=None):
        """features: dict from datasets.signal_features names."""
        mapping = {
            "vibration_RMS": "%s_rms" % channel,
            "vibration_std": "%s_std" % channel,
            "vibration_kurtosis": "%s_kurtosis" % channel,
            "vibration_crest_factor": "%s_crest_factor" % channel,
            "vibration_peak_to_peak": "%s_p2p" % channel,
            "spectral_energy": "%s_spectral_energy" % channel,
            "dominant_frequency": "%s_dominant_frequency" % channel,
        }
        if torque_prefix:
            mapping["torque_RMS"] = "%s_rms" % torque_prefix
        values = dict((k, features[v]) for k, v in mapping.items() if v in features)
        return cls(DOMAIN_REAL_MACHINE, source, values)

    def to_dict(self):
        return {"domain": self.domain, "source": self.source, "values": dict(self.values),
                "categories": list(self.available_categories)}
