"""
Anomaly detectors.

RuleBasedDetector   baseline mean + 2 / 3 std with persistence.
IsolationForestDetector   one scikit-learn IsolationForest per joint, with a
                    StandardScaler fitted on the training (normal) data only.

Both are stateful only in their persistence counters, which reset whenever the
simulation restarts. No Qt, no PyBullet.
"""

import numpy as np
import config

NORMAL = "NORMAL"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
NO_MODEL = "NO_MODEL"
SEVERITY_ORDER = {NO_MODEL: -1, NORMAL: 0, WARNING: 1, CRITICAL: 2}


def worst(*statuses):
    return max(statuses, key=lambda s: SEVERITY_ORDER.get(s, -1))


class RuleBasedDetector(object):

    def __init__(self, extractor, n_joints, features=None):
        self.extractor = extractor
        self.features = list(features or config.RULE_FEATURES)
        self.columns = [extractor.index[name] for name in self.features]
        self.n = n_joints
        self.reset()

    def reset(self):
        shape = (self.n, len(self.features))
        self.warning_count = np.zeros(shape, dtype=int)
        self.critical_count = np.zeros(shape, dtype=int)

    def evaluate(self, features, baseline, mode):
        """
        features: (n_joints, n_features). Returns per joint dicts with status,
        the most deviating feature and its z score.
        """
        used_mode, stats = baseline.for_mode(mode) if baseline is not None else (None, None)
        if stats is None:
            return [{"status": NO_MODEL, "z": 0.0, "feature": "", "baseline_mode": ""}
                    for _ in range(self.n)]
        values = features[:, self.columns]
        mean = stats["mean"][:, self.columns]
        std = stats["std_effective"][:, self.columns]
        z = (values - mean) / std
        z = np.where(np.isnan(z), 0.0, z)

        warn = z > config.RULE_WARNING_SIGMA
        crit = z > config.RULE_CRITICAL_SIGMA
        self.warning_count = np.where(warn, self.warning_count + 1, 0)
        self.critical_count = np.where(crit, self.critical_count + 1, 0)

        results = []
        for j in range(self.n):
            status = NORMAL
            if np.any(self.warning_count[j] >= config.RULE_WARNING_PERSISTENCE):
                status = WARNING
            if np.any(self.critical_count[j] >= config.RULE_CRITICAL_PERSISTENCE):
                status = CRITICAL
            k = int(np.argmax(z[j]))
            results.append({
                "status": status, "z": float(z[j, k]), "feature": self.features[k],
                "baseline_mode": used_mode,
                "z_all": dict(zip(self.features, (float(v) for v in z[j]))),
            })
        return results


class IsolationForestDetector(object):

    def __init__(self, feature_names, joint_names):
        self.feature_names = list(feature_names)
        self.joint_names = list(joint_names)
        self.models = []      # per joint (scaler, forest, margin_std)
        self.training_windows = 0
        self.reset()

    @property
    def trained(self):
        return bool(self.models)

    def reset(self):
        n = len(self.joint_names)
        self.warning_count = np.zeros(n, dtype=int)
        self.critical_count = np.zeros(n, dtype=int)

    def fit(self, samples):
        """samples: (W, n_joints, len(feature_names)), normal data only."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        data = np.asarray(samples, dtype=np.float64)
        if data.shape[0] < config.IF_MIN_TRAINING_WINDOWS:
            raise ValueError("need at least %d normal windows, have %d"
                             % (config.IF_MIN_TRAINING_WINDOWS, data.shape[0]))
        models = []
        for j in range(data.shape[1]):
            x = data[:, j, :]
            x = x[~np.any(np.isnan(x), axis=1)]
            scaler = StandardScaler().fit(x)
            xs = scaler.transform(x)
            forest = IsolationForest(n_estimators=config.IF_N_ESTIMATORS,
                                     contamination=config.IF_CONTAMINATION,
                                     random_state=config.IF_RANDOM_STATE)
            forest.fit(xs)
            margin = -forest.decision_function(xs)     # > 0 means anomalous
            models.append((scaler, forest, float(max(np.std(margin), 1e-3))))
        self.models = models
        self.training_windows = int(data.shape[0])
        self.reset()

    def evaluate(self, features):
        """features: (n_joints, len(feature_names)). Returns per joint dicts."""
        results = []
        for j, (scaler, forest, margin_std) in enumerate(self.models):
            x = features[j:j + 1]
            if np.any(np.isnan(x)):
                results.append({"status": NO_MODEL, "score": 0.0, "margin": 0.0})
                continue
            margin = float(-forest.decision_function(scaler.transform(x))[0])
            # Logistic squashing: 0.5 at the learned threshold.
            score = 1.0 / (1.0 + np.exp(-margin / margin_std))
            self.warning_count[j] = self.warning_count[j] + 1 if margin > 0.0 else 0
            critical = margin > config.IF_CRITICAL_MARGIN_SIGMA * margin_std
            self.critical_count[j] = self.critical_count[j] + 1 if critical else 0
            status = NORMAL
            if self.warning_count[j] >= config.IF_WARNING_PERSISTENCE:
                status = WARNING
            if self.critical_count[j] >= config.IF_CRITICAL_PERSISTENCE:
                status = CRITICAL
            results.append({"status": status, "score": float(score), "margin": margin})
        return results
