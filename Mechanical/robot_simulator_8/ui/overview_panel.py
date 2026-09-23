"""
Overview dashboard: robot health, per joint health bars and the per joint
maintenance table. Draws the monitoring service snapshot; computes nothing.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from predictive.health_model import band
from predictive.maintenance_engine import DISCLAIMER

SEVERITY_COLORS = {
    "HEALTHY": "#3f9e67",
    "MINOR_DEGRADATION": "#8fb339",
    "WARNING": "#d9a326",
    "CRITICAL": "#e0474c",
    "NO_DATA": "#6c7480",
}

TABLE_COLUMNS = ("Joint", "Health Score", "Anomaly Score", "Degradation Trend",
                 "Estimated RUL", "Severity", "Maintenance Recommendation")


def format_rul(seconds):
    if seconds is None:
        return "-"
    if seconds <= 0.0:
        return "at CRITICAL"
    if seconds < 120.0:
        return "%.0f s" % seconds
    if seconds < 7200.0:
        return "%.1f min" % (seconds / 60.0)
    return "%.1f h" % (seconds / 3600.0)


class OverviewPanel(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._names = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)

        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        caption = QLabel("ROBOT HEALTH")
        caption.setObjectName("hint")
        left_layout.addWidget(caption)
        self.robot_value = QLabel("--")
        self.robot_value.setStyleSheet("font-size: 34px; font-weight: bold;")
        left_layout.addWidget(self.robot_value)
        self.robot_note = QLabel("Train a baseline on the Anomaly tab to score joints.")
        self.robot_note.setObjectName("hint")
        self.robot_note.setWordWrap(True)
        left_layout.addWidget(self.robot_note)
        self._bars_holder = QWidget()
        self._bars = QGridLayout(self._bars_holder)
        self._bars.setContentsMargins(0, 6, 0, 0)
        self._bars.setVerticalSpacing(3)
        left_layout.addWidget(self._bars_holder)
        left_layout.addStretch(1)
        layout.addWidget(left)

        right = QVBoxLayout()
        banner = QLabel(DISCLAIMER)
        banner.setWordWrap(True)
        banner.setStyleSheet("color: #d9a326; font-weight: bold;")
        right.addWidget(banner)
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(TABLE_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(TABLE_COLUMNS) - 1, QHeaderView.Stretch)
        right.addWidget(self.table, 1)
        layout.addLayout(right, 1)
        self._bar_widgets = []

    def build(self, description):
        self._names = description["names"]
        while self._bars.count():
            widget = self._bars.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._bar_widgets = []
        for j, name in enumerate(self._names):
            label = QLabel("Joint %d" % (j + 1))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("--")
            bar.setFixedHeight(16)
            self._bars.addWidget(label, j, 0)
            self._bars.addWidget(bar, j, 1)
            self._bar_widgets.append(bar)
        self.table.setRowCount(len(self._names))
        for row, name in enumerate(self._names):
            for column in range(len(TABLE_COLUMNS)):
                self.table.setItem(row, column, QTableWidgetItem(""))
            self.table.item(row, 0).setText("%d  %s" % (row + 1, name))

    def update_snapshot(self, snap):
        robot = snap.get("robot_health")
        joints = snap.get("joints", [])
        if robot is None:
            self.robot_value.setText("--")
            self.robot_value.setStyleSheet("font-size: 34px; font-weight: bold; color: #6c7480;")
        else:
            worst = min((j["health"] for j in joints if j.get("health") and j["health"]["health"] is not None),
                        key=lambda h: h["health"], default=None)
            color = SEVERITY_COLORS.get(band(robot), "#3f9e67")
            self.robot_value.setText("%.0f%%" % robot)
            self.robot_value.setStyleSheet("font-size: 34px; font-weight: bold; color: %s;" % color)
            self.robot_note.setText("Mean of joint scores. Lowest joint: %s" % (
                "%.0f%% %s" % (worst["health"], worst["severity"]) if worst else "-"))

        for j, joint in enumerate(joints):
            if j >= len(self._bar_widgets):
                break
            health = joint.get("health")
            bar = self._bar_widgets[j]
            recommendation = joint.get("recommendation") or {}
            if health is None or health["health"] is None:
                bar.setValue(0)
                bar.setFormat("no data")
                values = ("-", "-", "-", "-", "NO_DATA", recommendation.get("action", "-"))
                severity = "NO_DATA"
            else:
                severity = health["severity"]
                bar.setValue(int(round(health["health"])))
                bar.setFormat("%.0f%%  %s" % (health["health"], "" if severity == "HEALTHY" else severity))
                slope = health.get("trend_slope")
                trend = health["trend"] if slope is None else "%s (%+.2f %%/s)" % (health["trend"], slope)
                score = joint.get("score")
                action = recommendation.get("action", "-")
                if recommendation.get("secondary"):
                    action += " + %s" % recommendation["secondary"].lower()
                values = ("%.0f%%" % health["health"], "-" if score is None else "%.2f" % score,
                          trend, format_rul(health["rul_s"]), severity, action)
            color = SEVERITY_COLORS.get(severity, "#6c7480")
            bar.setStyleSheet("QProgressBar { background: #1a1d23; border: 1px solid #2b2f38; "
                              "border-radius: 2px; text-align: center; color: #e8ebf0; }"
                              "QProgressBar::chunk { background: %s; }" % color)
            for column, text in enumerate(values, start=1):
                item = self.table.item(j, column)
                if item is None:
                    continue
                item.setText(text)
                if column == 5:
                    item.setForeground(QColor(color))
