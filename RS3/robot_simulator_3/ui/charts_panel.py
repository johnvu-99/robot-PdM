"""
Realtime joint charts.

The worker sends decimated chunks (CHART_SAMPLE_HZ samples, CHART_UI_HZ
deliveries). They are copied into a fixed size numpy ring buffer here, and the
plots redraw on their own timer only while the tab is visible, so a hidden
chart costs nothing but a memory copy.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

import config
from monitoring.telemetry import CHANNEL_INDEX, NUM_CHANNELS

# (key, caption, unit)
CHART_METRICS = (
    ("position", "Position", "rad"),
    ("velocity", "Velocity", "rad/s"),
    ("acceleration", "Acceleration", "rad/s²"),
    ("motor_torque", "Torque", "N m"),
    ("tracking_error", "Tracking error", "rad"),
    ("power", "Mechanical power", "W"),
    ("reaction_force_magnitude", "Reaction force |F|", "N"),
)

pg.setConfigOptions(antialias=False, background="#14161b", foreground="#97a0ad")


class ChartsPanel(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._capacity = int(max(config.CHART_WINDOWS_S) * config.CHART_SAMPLE_HZ) + 8
        self._n = 0
        self._times = np.zeros(self._capacity)
        self._data = np.zeros((self._capacity, 0, NUM_CHANNELS))
        self._head = 0
        self._count = 0
        self._dirty = False
        self._window = float(config.CHART_DEFAULT_WINDOW_S)
        self._frozen = False
        self._joint_checks = []
        self._curves = [[], []]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.metric_combos = []
        for label, default in (("Top", "motor_torque"), ("Bottom", "tracking_error")):
            controls.addWidget(QLabel(label))
            combo = QComboBox()
            for key, caption, unit in CHART_METRICS:
                combo.addItem("%s (%s)" % (caption, unit), key)
            combo.setCurrentIndex([m[0] for m in CHART_METRICS].index(default))
            combo.currentIndexChanged.connect(self._on_metric_changed)
            controls.addWidget(combo)
            self.metric_combos.append(combo)

        controls.addSpacing(10)
        controls.addWidget(QLabel("Window"))
        self.window_combo = QComboBox()
        for seconds in config.CHART_WINDOWS_S:
            self.window_combo.addItem("%d s" % seconds, seconds)
        self.window_combo.setCurrentIndex(
            list(config.CHART_WINDOWS_S).index(config.CHART_DEFAULT_WINDOW_S))
        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        controls.addWidget(self.window_combo)

        self.freeze_check = QCheckBox("Freeze")
        self.freeze_check.toggled.connect(self._on_freeze)
        controls.addWidget(self.freeze_check)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear)
        controls.addWidget(clear_button)

        controls.addSpacing(10)
        self._joint_row = QHBoxLayout()
        self._joint_row.setSpacing(4)
        controls.addLayout(self._joint_row)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.plots = []
        for index in range(2):
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setClipToView(True)
            plot.setDownsampling(auto=True, mode="peak")
            plot.setMouseEnabled(x=False, y=True)
            plot.setXRange(-self._window, 0.0, padding=0.0)
            plot.setLabel("bottom", "time relative to now", units="s")
            if index == 1:
                plot.setXLink(self.plots[0])
            else:
                plot.addLegend(offset=(-10, 6))
            self.plots.append(plot)
            layout.addWidget(plot, 1)
        self._update_axis_labels()

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000.0 / config.CHART_UI_HZ))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # -- setup --------------------------------------------------------------

    def build(self, description):
        names = description["names"]
        self._n = len(names)
        self._data = np.zeros((self._capacity, self._n, NUM_CHANNELS))
        self.clear()

        while self._joint_row.count():
            widget = self._joint_row.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._joint_checks = []

        for plot_index, plot in enumerate(self.plots):
            for curve in self._curves[plot_index]:
                plot.removeItem(curve)
            self._curves[plot_index] = []
            if plot.plotItem.legend is not None:
                plot.plotItem.legend.clear()

        for joint in range(self._n):
            color = pg.intColor(joint, hues=max(self._n, 1), values=1,
                                maxValue=230, minValue=230, sat=200)
            check = QCheckBox("J%d" % (joint + 1))
            check.setChecked(True)
            check.setStyleSheet("QCheckBox { color: %s; }" % color.name())
            check.toggled.connect(self._on_joint_toggled)
            self._joint_row.addWidget(check)
            self._joint_checks.append(check)
            for plot_index, plot in enumerate(self.plots):
                curve = plot.plot(pen=pg.mkPen(color, width=1),
                                  name="J%d %s" % (joint + 1, names[joint]) if plot_index == 0 else None)
                curve.setClipToView(True)
                self._curves[plot_index].append(curve)

    def clear(self):
        self._head = 0
        self._count = 0
        self._dirty = True
        for curves in self._curves:
            for curve in curves:
                curve.setData([], [])

    # -- data ---------------------------------------------------------------

    def append_chunk(self, chunk):
        times, data = chunk
        if data.ndim != 3 or data.shape[1] != self._n:
            return
        if self._count and times[0] < self._times[(self._head - 1) % self._capacity]:
            # Simulated time restarted (stop, reset, robot swap).
            self.clear()
        k = times.shape[0]
        if k > self._capacity:
            times, data, k = times[-self._capacity:], data[-self._capacity:], self._capacity
        positions = (self._head + np.arange(k)) % self._capacity
        self._times[positions] = times
        self._data[positions] = data
        self._head = (self._head + k) % self._capacity
        self._count = min(self._count + k, self._capacity)
        self._dirty = True

    def _refresh(self):
        if self._frozen or not self._dirty or not self.isVisible() or self._count == 0:
            return
        self._dirty = False
        order = (self._head - self._count + np.arange(self._count)) % self._capacity
        times = self._times[order]
        latest = times[-1]
        keep = times >= latest - self._window
        order = order[keep]
        x = times[keep] - latest
        for plot_index, combo in enumerate(self.metric_combos):
            channel = CHANNEL_INDEX[combo.currentData()]
            values = self._data[order, :, channel]
            for joint, curve in enumerate(self._curves[plot_index]):
                if self._joint_checks[joint].isChecked():
                    curve.setData(x, values[:, joint])

    # -- events -------------------------------------------------------------

    def _update_axis_labels(self):
        for plot, combo in zip(self.plots, self.metric_combos):
            key = combo.currentData()
            for metric_key, caption, unit in CHART_METRICS:
                if metric_key == key:
                    plot.setLabel("left", caption, units=unit)
                    plot.enableAutoRange(axis="y")

    def _on_metric_changed(self, _index):
        self._update_axis_labels()
        self._dirty = True

    def _on_window_changed(self, _index):
        self._window = float(self.window_combo.currentData())
        self.plots[0].setXRange(-self._window, 0.0, padding=0.0)
        self._dirty = True

    def _on_freeze(self, frozen):
        self._frozen = bool(frozen)
        self._dirty = True

    def _on_joint_toggled(self, _checked):
        for joint, check in enumerate(self._joint_checks):
            for curves in self._curves:
                curves[joint].setVisible(check.isChecked())
        self._dirty = True
