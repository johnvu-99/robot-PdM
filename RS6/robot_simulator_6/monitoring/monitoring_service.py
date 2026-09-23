"""
Monitoring service: feature extraction and anomaly detection off the physics
thread.

The simulation worker hands over raw 240 Hz records through submit() (a queue
put, never blocking). This thread keeps a numpy ring buffer, evaluates features
every FEATURE_HOP_S of simulated time, runs the rule based and Isolation Forest
detectors, and publishes a result snapshot that the UI polls. Training and
model I/O also run here, so the GUI thread never blocks on scikit-learn.

No Qt imports. ML never enters the physics code.
"""

import collections
import logging
import queue
import threading
import time

import numpy as np

import config
from monitoring import dataset_features
from monitoring.anomaly_detector import (
    CRITICAL, IsolationForestDetector, NO_MODEL, NORMAL, RuleBasedDetector, WARNING, worst,
)
from monitoring.baseline import Baseline
from monitoring.feature_extractor import FeatureExtractor, signals_from_matrix
from storage import model_store

log = logging.getLogger(__name__)

STABLE_SETTLE_S = 3.0


class MonitoringService(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, name="MonitoringService")
        self.daemon = True
        self.extractor = FeatureExtractor()
        self._inbox = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = {"joints": [], "status": "Waiting for robot."}
        self._messages = collections.deque(maxlen=200)

        self.robot_model = ""
        self.joint_names = []
        self.n = 0
        self.baseline = None
        self.forest = None
        self.rules = None
        self._training_features = []     # (mode, features) from clean windows
        self._collect_target = 0
        self._collect_rejected = 0
        self._collecting = False
        self._label_cache = {}
        self._reset_buffer()

    # -- producer API (simulation thread) -----------------------------------

    def submit(self, records, mode, transitioning, state):
        if records:
            self._inbox.put(("data", records, mode, transitioning, state))

    # -- UI API (GUI thread) ------------------------------------------------

    def command(self, name, payload=None):
        self._inbox.put(("command", name, payload or {}))

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def pop_messages(self):
        messages = []
        while self._messages:
            messages.append(self._messages.popleft())
        return messages

    def stop(self):
        self._stop_event.set()
        self._inbox.put(("stop",))

    # -- thread -------------------------------------------------------------

    def run(self):
        while not self._stop_event.is_set():
            try:
                item = self._inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if item[0] == "stop":
                    break
                if item[0] == "data":
                    self._on_data(*item[1:])
                elif item[0] == "command":
                    self._on_command(item[1], item[2])
            except (ValueError, KeyError, IndexError, IOError, OSError, TypeError) as exc:
                self._message("Monitoring error: %s" % exc)
                log.warning("monitoring command failed: %s", exc)

    def _message(self, text):
        self._messages.append(text)

    # -- buffer -------------------------------------------------------------

    def _reset_buffer(self):
        capacity = int((self.extractor.longest_window + 1.0) * config.PHYSICS_HZ)
        self._capacity = capacity
        self._times = np.zeros(capacity)
        self._signals = np.zeros((capacity, self.n, 6))
        self._normal = np.ones((capacity, self.n), dtype=bool)
        self._stable = np.zeros(capacity, dtype=bool)
        self._head = 0
        self._count = 0
        self._last_eval = -1e9
        self._mode = None
        self._mode_since = 0.0
        self._last_truth = ["NORMAL"] * self.n
        self._stats = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self._joint_status = [NO_MODEL] * self.n
        if self.rules is not None:
            self.rules.reset()
        if self.forest is not None:
            self.forest.reset()

    def _labels_normal(self, labels):
        key = id(labels)
        cached = self._label_cache.get(key)
        if cached is None or cached[0] is not labels:
            flags = np.array([label[0] == "NORMAL" for label in labels], dtype=bool)
            names = [label[0] for label in labels]
            if len(self._label_cache) > 64:
                self._label_cache.clear()
            cached = (labels, flags, names)
            self._label_cache[key] = cached
        return cached[1], cached[2]

    def _on_data(self, records, mode, transitioning, state):
        if self.n == 0 or records[0][1].shape[0] != self.n:
            return
        times = np.array([r[0] for r in records])
        if self._count and times[0] < self._times[(self._head - 1) % self._capacity]:
            self._reset_buffer()
        if mode != self._mode:
            self._mode = mode
            self._mode_since = times[0]
        signals = signals_from_matrix(np.stack([r[1] for r in records]))
        k = times.shape[0]
        positions = (self._head + np.arange(k)) % self._capacity
        self._times[positions] = times
        self._signals[positions] = signals
        stable = (not transitioning) and state == "RUNNING" and times - self._mode_since >= STABLE_SETTLE_S
        self._stable[positions] = stable
        for i, record in enumerate(records):
            if record[2] is None:
                self._normal[positions[i]] = True
            else:
                flags, names = self._labels_normal(record[2])
                self._normal[positions[i]] = flags
                self._last_truth = names
        self._head = (self._head + k) % self._capacity
        self._count = min(self._count + k, self._capacity)

        latest = times[-1]
        if latest - self._last_eval >= config.FEATURE_HOP_S:
            self._last_eval = latest
            self._evaluate(mode)

    def _ordered(self, seconds):
        order = (self._head - self._count + np.arange(self._count)) % self._capacity
        times = self._times[order]
        keep = times >= times[-1] - seconds + 1e-9
        order = order[keep]
        return times[keep], order

    # -- evaluation ---------------------------------------------------------

    def _evaluate(self, mode):
        times, order = self._ordered(self.extractor.longest_window)
        if times[-1] - times[0] < config.DETECTION_WINDOW_S - 2 * config.FIXED_DT:
            self._publish(None, None, None, mode, "Filling %d s window..." % config.DETECTION_WINDOW_S)
            return
        full_window = times[-1] - times[0] >= self.extractor.longest_window - 2 * config.FIXED_DT
        signals = self._signals[order]
        features = self.extractor.compute(times, signals)

        _, detect_order = self._ordered(config.DETECTION_WINDOW_S)
        detection_stable = bool(np.all(self._stable[detect_order]))
        clean = full_window and bool(np.all(self._stable[order])) and bool(np.all(self._normal[order]))

        if self._collecting:
            if clean:
                self._training_features.append((mode, features))
            else:
                self._collect_rejected += 1
            if len(self._training_features) >= self._collect_target:
                self._finish_baseline("live collection")

        rule_results = None
        forest_results = None
        if detection_stable:
            if self.baseline is not None and self.baseline.trained:
                rule_results = self.rules.evaluate(features, self.baseline, mode)
            if self.forest is not None and self.forest.trained:
                forest_results = self.forest.evaluate(
                    self.extractor.select(features, config.IF_FEATURES))
            note = ""
        else:
            note = "Motion settling (mode change or not running): detectors on hold."
        self._publish(features, rule_results, forest_results, mode, note)

    def _publish(self, features, rule_results, forest_results, mode, note):
        joints = []
        for j in range(self.n):
            rule = rule_results[j] if rule_results else None
            forest = forest_results[j] if forest_results else None
            previous = self._joint_status[j]
            if rule is None and forest is None:
                status = previous
            else:
                status = worst(rule["status"] if rule else NO_MODEL,
                               forest["status"] if forest else NO_MODEL)
            truth = self._last_truth[j] if j < len(self._last_truth) else "NORMAL"
            if (rule or forest) and status != NO_MODEL:
                predicted = status != NORMAL
                actual = truth != "NORMAL"
                key = ("tp" if actual else "fp") if predicted else ("fn" if actual else "tn")
                self._stats[key] += 1
            if status != previous and status != NO_MODEL and (rule or forest):
                self._message("Joint %d: %s -> %s (%s)" % (
                    j + 1, previous, status,
                    ("rule %s z=%.1f" % (rule["feature"], rule["z"])) if rule else "isolation forest"))
            self._joint_status[j] = status
            joints.append({
                "status": status,
                "rule_status": rule["status"] if rule else NO_MODEL,
                "rule_feature": rule["feature"] if rule else "",
                "rule_z": rule["z"] if rule else 0.0,
                "baseline_mode": rule["baseline_mode"] if rule else "",
                "if_status": forest["status"] if forest else NO_MODEL,
                "score": forest["score"] if forest else None,
                "truth": truth,
                "features": ({name: float(features[j, self.extractor.index[name]])
                              for name in config.RULE_FEATURES + ("torque_trend", "tracking_error_trend", "power_trend")}
                             if features is not None else {}),
            })
        snapshot = {
            "joints": joints,
            "mode": mode,
            "note": note,
            "baseline": self._baseline_summary(),
            "forest": ("%d windows" % self.forest.training_windows) if self.forest and self.forest.trained else "",
            "collecting": self._collecting,
            "collected": len(self._training_features),
            "collect_target": self._collect_target,
            "rejected": self._collect_rejected,
            "stats": dict(self._stats),
            "robot_model": self.robot_model,
        }
        with self._lock:
            self._snapshot = snapshot

    def _baseline_summary(self):
        if self.baseline is None or not self.baseline.trained:
            return ""
        return ", ".join("%s %d" % (mode, count)
                         for mode, count in sorted(self.baseline.window_counts.items()))

    # -- commands -----------------------------------------------------------

    def _on_command(self, name, payload):
        handler = getattr(self, "_cmd_" + name, None)
        if handler is None:
            self._message("Unknown monitoring command %s" % name)
            return
        handler(payload)
        self._publish(None, None, None, self._mode, "")

    def _cmd_configure(self, payload):
        names = list(payload["names"])
        model = payload.get("urdf", "")
        changed = names != self.joint_names or model != self.robot_model
        self.joint_names = names
        self.robot_model = model
        self.n = len(names)
        if changed:
            self.baseline = None
            self.forest = None
            self._training_features = []
            self._collecting = False
            self.rules = RuleBasedDetector(self.extractor, self.n)
        self._reset_buffer()

    def _cmd_start_baseline(self, payload):
        seconds = float(payload.get("seconds", config.BASELINE_DEFAULT_SECONDS))
        self._training_features = []
        self._collect_rejected = 0
        self._collect_target = max(config.BASELINE_MIN_WINDOWS, int(seconds / config.FEATURE_HOP_S))
        self._collecting = True
        self._message("Baseline collection started: %d clean windows (%.0f s of normal motion). "
                      "Windows with faults, mode changes or pauses are rejected."
                      % (self._collect_target, seconds))

    def _cmd_cancel_baseline(self, payload):
        if self._collecting:
            self._collecting = False
            self._message("Baseline collection cancelled.")

    def _finish_baseline(self, source):
        self._collecting = False
        baseline = Baseline.fit(self.extractor.names, self.joint_names, self._training_features)
        if not baseline.trained:
            self._message("Baseline not trained: fewer than %d clean windows for any mode."
                          % config.BASELINE_MIN_WINDOWS)
            return False
        self.baseline = baseline
        self.rules = RuleBasedDetector(self.extractor, self.n)
        self._message("Baseline trained from %s: %d windows (%s)."
                      % (source, len(self._training_features), self._baseline_summary()))
        return True

    def _cmd_train_forest(self, payload):
        if self._collecting:
            self._message("Baseline collection still running (%d / %d windows). "
                          "Train the Isolation Forest when it finishes."
                          % (len(self._training_features), self._collect_target))
            return
        if not self._training_features:
            self._message("Train a baseline first: the Isolation Forest uses the same normal windows.")
            return
        data = np.stack([self.extractor.select(f, config.IF_FEATURES)
                         for _, f in self._training_features])
        started = time.perf_counter()
        forest = IsolationForestDetector(config.IF_FEATURES, self.joint_names)
        try:
            forest.fit(data)
        except ValueError as exc:
            self._message("Isolation Forest not trained: %s." % exc)
            return
        self.forest = forest
        self._message("Isolation Forest trained: %d joints x %d windows x %d features in %.1f s."
                      % (self.n, data.shape[0], data.shape[2], time.perf_counter() - started))

    def _cmd_train_from_dataset(self, payload):
        root = payload.get("root", config.SIMULATION_DATA_DIR)
        started = time.perf_counter()
        samples, names, files = dataset_features.normal_dataset_windows(
            root, self.extractor, config.FEATURE_HOP_S, STABLE_SETTLE_S + 1.0)
        if not samples:
            self._message("No normal dataset windows found in %s/normal." % root)
            return
        if names != self.joint_names:
            self._message("Dataset joints %s do not match the loaded robot." % names)
            return
        previous = self._training_features
        self._training_features = samples
        if not self._finish_baseline("%d dataset files" % files):
            self._training_features = previous
            self._message("Dataset gave %d usable windows; each run needs more than %.0f s "
                          "after settling. Generate the STANDARD preset (20 s runs)."
                          % (len(samples), self.extractor.longest_window))
        self._message("Dataset feature extraction took %.1f s." % (time.perf_counter() - started))

    def _cmd_reset_stats(self, payload):
        self._stats = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        if self.rules is not None:
            self.rules.reset()
        if self.forest is not None:
            self.forest.reset()

    def _cmd_save(self, payload):
        if self.baseline is None and self.forest is None:
            self._message("Nothing to save yet.")
            return
        path = payload.get("path") or model_store.default_path(self.robot_model)
        model_store.save_bundle(path, self.robot_model, self.joint_names, self.baseline,
                                self.forest, {"training_windows": len(self._training_features)})
        self._message("Model saved: %s" % path)

    def _cmd_load(self, payload):
        path = payload["path"]
        bundle = model_store.load_bundle(path)
        if list(bundle["joint_names"]) != self.joint_names:
            self._message("Model joints %s do not match the loaded robot." % bundle["joint_names"])
            return
        self.baseline = Baseline.from_dict(bundle["baseline"]) if bundle.get("baseline") else None
        self.rules = RuleBasedDetector(self.extractor, self.n)
        self.forest = None
        if bundle.get("forest"):
            forest = IsolationForestDetector(bundle["forest"]["feature_names"], self.joint_names)
            forest.models = bundle["forest"]["models"]
            forest.training_windows = bundle["forest"]["training_windows"]
            forest.reset()
            self.forest = forest
        self._message("Model loaded: %s (baseline %s, forest %s)."
                      % (path, "yes" if self.baseline else "no", "yes" if self.forest else "no"))
