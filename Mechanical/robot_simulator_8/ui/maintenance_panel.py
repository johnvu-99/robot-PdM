"""
Maintenance tab: recommendation, reasoning and health score breakdown per
joint, plus the configurable health weights.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

import config
from predictive.health_model import EVIDENCE
from predictive.maintenance_engine import DISCLAIMER
from ui.overview_panel import format_rul


class MaintenancePanel(QWidget):

    command = pyqtSignal(str, object)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._names = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)

        left = QVBoxLayout()
        banner = QLabel(DISCLAIMER)
        banner.setWordWrap(True)
        banner.setStyleSheet("color: #d9a326; font-weight: bold;")
        left.addWidget(banner)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setObjectName("eventLog")
        left.addWidget(self.text, 1)
        layout.addLayout(left, 3)

        group = QGroupBox("Health score weights")
        form = QFormLayout(group)
        hint = QLabel("Largest share of health each evidence item can remove on its own. "
                      "health = 100 x product(1 - weight x penalty) over available evidence.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        form.addRow(hint)
        self.spins = {}
        for name in EVIDENCE:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(config.HEALTH_WEIGHTS.get(name, 0.0))
            form.addRow(name.replace("_", " "), spin)
            self.spins[name] = spin
        row = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(lambda: self._send(False))
        save_button = QPushButton("Apply and save")
        save_button.clicked.connect(lambda: self._send(True))
        row.addWidget(apply_button)
        row.addWidget(save_button)
        form.addRow(row)
        layout.addWidget(group, 1)
        self._weights_loaded = False

    def build(self, description):
        self._names = description["names"]

    def _send(self, save):
        self.command.emit("set_health_weights", {
            "weights": dict((k, s.value()) for k, s in self.spins.items()), "save": save})

    def update_snapshot(self, snap):
        if not self._weights_loaded and snap.get("health_weights"):
            for name, value in snap["health_weights"].items():
                if name in self.spins:
                    self.spins[name].setValue(value)
            self._weights_loaded = True
        lines = []
        for j, joint in enumerate(snap.get("joints", [])):
            name = self._names[j] if j < len(self._names) else ""
            health = joint.get("health")
            recommendation = joint.get("recommendation") or {}
            if health is None or health["health"] is None:
                lines.append("Joint %d %s: no health data (%s)" % (
                    j + 1, name, recommendation.get("reason", "train a baseline")))
                continue
            lines.append("Joint %d %s: %.0f%% %s   ->  %s%s" % (
                j + 1, name, health["health"], health["severity"], recommendation.get("action", "-"),
                (" (also: %s)" % recommendation["secondary"].lower()) if recommendation.get("secondary") else ""))
            lines.append("    why: %s" % recommendation.get("reason", ""))
            parts = ["%s %.0f%%" % (item, 100 * reduction)
                     for item, _, reduction in health["contributions"] if reduction >= 0.005]
            lines.append("    health reduced by: %s" % (", ".join(parts) if parts else "nothing"))
            lines.append("    trend %s, RUL (health trend extrapolation to %d%%) %s, evidence confidence %.0f%%"
                         % (health["trend"], config.RUL_TREND_CRITICAL_HEALTH, format_rul(health["rul_s"]),
                            100 * health["confidence"]))
        text = "\n".join(lines)
        if text != self.text.toPlainText():
            scroll = self.text.verticalScrollBar().value()
            self.text.setPlainText(text)
            self.text.verticalScrollBar().setValue(scroll)
