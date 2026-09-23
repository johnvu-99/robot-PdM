"""
Anomaly tab: baseline and Isolation Forest controls, per joint results and a
running comparison against the fault injector's ground truth.

Polls the monitoring service snapshot at MONITOR_UI_HZ. Never computes.
"""

import os

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import config

COLUMNS = ("Joint", "Anomaly Score", "Isolation Forest", "Rules", "Status",
           "Largest deviation", "torque_RMS", "tracking_error_RMS", "Ground truth")

STATUS_COLORS = {
    "NORMAL": QColor("#3f9e67"),
    "WARNING": QColor("#d9a326"),
    "CRITICAL": QColor("#e0474c"),
    "NO_MODEL": QColor("#6c7480"),
}


class AnomalyPanel(QWidget):

    log = pyqtSignal(str)

    def __init__(self, service, parent=None):
        QWidget.__init__(self, parent)
        self.service = service
        self._names = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(10.0, 3600.0)
        self.seconds.setValue(config.BASELINE_DEFAULT_SECONDS)
        self.seconds.setSuffix(" s")
        self.seconds.setDecimals(0)
        controls.addWidget(self.seconds)
        buttons = (
            ("Train Baseline", lambda: service.command("start_baseline", {"seconds": self.seconds.value()})),
            ("Cancel", lambda: service.command("cancel_baseline")),
            ("Baseline from Dataset", lambda: service.command("train_from_dataset")),
            ("Train Isolation Forest", lambda: service.command("train_forest")),
            ("Save Model", self._on_save),
            ("Load Model", self._on_load),
            ("Reset Scores", lambda: service.command("reset_stats")),
        )
        for caption, handler in buttons:
            button = QPushButton(caption)
            button.clicked.connect(handler)
            controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status = QLabel("Train a baseline while the robot runs normal motion.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.validation = QLabel("")
        self.validation.setObjectName("metric")
        layout.addWidget(self.validation)

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000.0 / config.MONITOR_UI_HZ))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def build(self, description):
        self._names = description["names"]
        self.table.setRowCount(len(self._names))
        for row, name in enumerate(self._names):
            for column in range(len(COLUMNS)):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setText("%d  %s" % (row + 1, name))

    def _on_save(self):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save anomaly model", config.MODEL_DIR,
                                              "Model bundles (*.joblib)")
        if path:
            if not path.endswith(".joblib"):
                path += ".joblib"
            self.service.command("save", {"path": path})

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load anomaly model", config.MODEL_DIR,
                                              "Model bundles (*.joblib)")
        if path:
            self.service.command("load", {"path": path})

    def _refresh(self):
        for message in self.service.pop_messages():
            self.log.emit(message)
        snap = self.service.snapshot()
        parts = []
        if snap.get("collecting"):
            parts.append("Collecting baseline: %d / %d clean windows (%d rejected)."
                         % (snap["collected"], snap["collect_target"], snap["rejected"]))
        parts.append("Baseline: %s" % (snap.get("baseline") or "not trained"))
        parts.append("Isolation Forest: %s" % (snap.get("forest") or "not trained"))
        if snap.get("mode"):
            parts.append("Mode: %s" % snap["mode"])
        if snap.get("note"):
            parts.append(snap["note"])
        self.status.setText("   |   ".join(parts))

        for row, joint in enumerate(snap.get("joints", [])):
            if row >= self.table.rowCount():
                break
            score = joint.get("score")
            features = joint.get("features", {})
            values = (
                "-" if score is None else "%.2f" % score,
                joint["if_status"], joint["rule_status"], joint["status"],
                ("%s  z=%+.1f" % (joint["rule_feature"], joint["rule_z"])) if joint["rule_feature"] else "-",
                "%.2f" % features["torque_RMS"] if "torque_RMS" in features else "-",
                "%.4f" % features["tracking_error_RMS"] if "tracking_error_RMS" in features else "-",
                joint["truth"],
            )
            for column, text in enumerate(values, start=1):
                item = self.table.item(row, column)
                item.setText(text)
                if column in (2, 3, 4):
                    item.setForeground(STATUS_COLORS.get(text, STATUS_COLORS["NO_MODEL"]))
            self.table.item(row, 8).setForeground(
                STATUS_COLORS["NORMAL"] if joint["truth"] == "NORMAL" else STATUS_COLORS["CRITICAL"])

        stats = snap.get("stats", {})
        total = sum(stats.values())
        if total:
            tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
            precision = tp / float(tp + fp) if tp + fp else 0.0
            recall = tp / float(tp + fn) if tp + fn else 0.0
            self.validation.setText(
                "Validation against injected ground truth (per joint window): "
                "TP %d  FP %d  FN %d  TN %d   precision %.2f  recall %.2f   "
                "(persistence delays detection by design)" % (tp, fp, fn, tn, precision, recall))
        else:
            self.validation.setText("")
