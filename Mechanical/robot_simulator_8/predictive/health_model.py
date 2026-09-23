"""
Joint health score (0 - 100) from monitoring evidence.

Evidence per joint, each turned into a penalty in [0, 1]:
    tracking error, torque, force, power, acceleration   baseline z scores
    isolation forest                                     anomaly score
    degradation trend                                    z score of trend features
    RUL                                                  health trend extrapolation

Combination (independent reductions, configurable weights):
    health = 100 * product(1 - weight * penalty) over AVAILABLE evidence

Why not a weighted average: one severe fault must be able to pull health down
on its own, and an average dilutes it with healthy signals. Missing evidence is
excluded rather than counted as healthy. The per item contributions are
returned so the score can always be explained.

No Qt, no PyBullet, no scikit-learn.
"""

import collections
import json
import math
import os

import numpy as np

import config

EVIDENCE = ("tracking_error", "torque", "force", "power", "acceleration",
            "temperature", "isolation_forest", "degradation_trend", "rul")


def ramp(value, start, full):
    if value is None or math.isnan(value):
        return None
    if full <= start:
        return 1.0 if value >= start else 0.0
    return float(min(1.0, max(0.0, (value - start) / (full - start))))


def band(health):
    if health is None:
        return "NO_DATA"
    for threshold, name in config.HEALTH_BANDS:
        if health >= threshold:
            return name
    return config.HEALTH_BANDS[-1][1]


def load_weights(path=None):
    weights = dict(config.HEALTH_WEIGHTS)
    path = path or config.HEALTH_WEIGHTS_FILE
    if os.path.isfile(path):
        with open(path) as handle:
            stored = json.load(handle)
        for key, value in stored.items():
            if key in weights:
                weights[key] = float(min(1.0, max(0.0, value)))
    return weights


def save_weights(weights, path=None):
    path = path or config.HEALTH_WEIGHTS_FILE
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(dict(weights), handle, indent=2)
    return path


class HealthModel(object):

    def __init__(self, n_joints, weights=None):
        self.n = int(n_joints)
        self.weights = dict(weights or config.HEALTH_WEIGHTS)
        self.reset()

    def set_weights(self, weights):
        for key, value in weights.items():
            if key in self.weights:
                self.weights[key] = float(min(1.0, max(0.0, value)))

    def reset(self):
        self._smoothed = [None] * self.n
        self._last_time = [None] * self.n
        self._history = [collections.deque() for _ in range(self.n)]

    def evaluate(self, sim_time, inputs):
        """
        inputs: per joint dict with optional keys
            z: {evidence: z score}   if_score: float   trend_z: float
        Returns per joint result dicts.
        """
        return [self._joint(j, sim_time, inputs[j]) for j in range(self.n)]

    def _joint(self, j, t, data):
        penalties = {}
        for name, z in (data.get("z") or {}).items():
            penalties[name] = ramp(z, config.HEALTH_Z_START, config.HEALTH_Z_FULL)
        # Temperature uses an absolute rise over ambient, not a baseline z score:
        # a motor warms up for minutes after starting, and that warm-up is
        # normal. Rise is what a maintenance engineer would judge too.
        if data.get("temperature_rise") is not None:
            absolute = ramp(data["temperature_rise"], config.THERMAL_WARN_RISE_C,
                            config.THERMAL_CRITICAL_RISE_C)
            relative = ramp(data.get("temperature_ratio"), config.THERMAL_RATIO_WARN,
                            config.THERMAL_RATIO_CRITICAL)
            penalties["temperature"] = max(absolute, relative or 0.0)
        if data.get("if_score") is not None:
            penalties["isolation_forest"] = ramp(data["if_score"], config.HEALTH_IF_START,
                                                 config.HEALTH_IF_FULL)
        if data.get("trend_z") is not None:
            penalties["degradation_trend"] = ramp(data["trend_z"], config.HEALTH_TREND_Z_START,
                                                  config.HEALTH_TREND_Z_FULL)
        penalties = dict((k, v) for k, v in penalties.items() if v is not None)
        if not penalties:
            return {"health": None, "severity": "NO_DATA", "rul_s": None, "trend": "no data",
                    "trend_slope": None, "contributions": [], "evidence_used": [],
                    "confidence": 0.0}

        remaining = 1.0
        contributions = []
        for name, penalty in penalties.items():
            reduction = self.weights.get(name, 0.0) * penalty
            remaining *= 1.0 - reduction
            contributions.append((name, round(penalty, 3), round(reduction, 3)))
        raw = 100.0 * remaining

        # Smooth in simulated time so one noisy window cannot flip a band.
        previous = self._smoothed[j]
        if previous is None or self._last_time[j] is None or t < self._last_time[j]:
            smoothed = raw
            self._history[j].clear()
        else:
            dt = t - self._last_time[j]
            alpha = 1.0 - math.exp(-dt / max(config.HEALTH_SMOOTHING_S, 1e-6))
            smoothed = previous + alpha * (raw - previous)
        self._smoothed[j] = smoothed
        self._last_time[j] = t

        rul_s, slope, trend = self._trend(j, t, smoothed)
        health = smoothed
        # The RUL penalty is predictive evidence: it only matters while the
        # joint is still above CRITICAL. Below that the score already says so,
        # and applying it again would count the same drop twice.
        if rul_s is not None and smoothed > config.RUL_TREND_CRITICAL_HEALTH:
            rul_penalty = ramp(config.RUL_PENALTY_HORIZON_S - rul_s, 0.0, config.RUL_PENALTY_HORIZON_S)
            reduction = self.weights.get("rul", 0.0) * rul_penalty
            health = smoothed * (1.0 - reduction)
            contributions.append(("rul", round(rul_penalty, 3), round(reduction, 3)))

        contributions.sort(key=lambda item: item[2], reverse=True)
        total_weight = sum(self.weights.values()) or 1.0
        used = [name for name, _, _ in contributions]
        return {
            "health": float(health),
            "severity": band(health),
            "rul_s": rul_s,
            "trend": trend,
            "trend_slope": slope,
            "contributions": contributions,
            "evidence_used": used,
            "confidence": float(sum(self.weights.get(n, 0.0) for n in used) / total_weight),
        }

    def _trend(self, j, t, health):
        """Linear fit of the smoothed score over RUL_TREND_WINDOW_S (before the RUL penalty)."""
        history = self._history[j]
        history.append((t, health))
        while history and history[0][0] < t - config.RUL_TREND_WINDOW_S:
            history.popleft()
        if len(history) < config.RUL_TREND_MIN_POINTS:
            return None, None, "collecting"
        times = np.array([h[0] for h in history])
        values = np.array([h[1] for h in history])
        tc = times - times.mean()
        denominator = float(np.sum(tc * tc))
        if denominator <= 0.0:
            return None, None, "collecting"
        slope = float(np.sum(tc * (values - values.mean())) / denominator)   # % per s
        if slope < -config.RUL_TREND_MIN_SLOPE:
            if health <= config.RUL_TREND_CRITICAL_HEALTH:
                return 0.0, slope, "falling"
            return (health - config.RUL_TREND_CRITICAL_HEALTH) / -slope, slope, "falling"
        if slope > config.RUL_TREND_MIN_SLOPE:
            return None, slope, "recovering"
        return None, slope, "stable"


def robot_health(joint_results):
    values = [r["health"] for r in joint_results if r["health"] is not None]
    if not values:
        return None
    mode = config.ROBOT_HEALTH_AGGREGATION
    if mode == "MIN":
        return float(min(values))
    if mode == "MEAN_MIN":
        return float(0.5 * (np.mean(values) + min(values)))
    return float(np.mean(values))
