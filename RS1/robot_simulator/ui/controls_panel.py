"""
Left hand control panel and the bottom telemetry panel.

Both are pure view code. They emit command name/payload pairs that the main
window forwards to the simulation thread, and they render whatever snapshots
come back. Nothing here blocks or computes.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

import config
from simulator.trajectory import MODES, MODE_LABELS, MODE_MANUAL


def _slider(minimum, maximum, value):
    widget = QSlider(Qt.Horizontal)
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)
    widget.setValue(value)
    widget.setTracking(True)
    return widget


class ControlsPanel(QWidget):

    command = pyqtSignal(str, object)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setFixedWidth(config.CONTROL_PANEL_WIDTH)

        self._joint_sliders = []
        self._joint_labels = []
        self._joint_ranges = []
        self._mode = MODES[0]
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_motion_group())
        layout.addWidget(self._build_joint_group(), 1)
        layout.addWidget(self._build_controller_group())
        layout.addWidget(self._build_view_group())

    # -- construction -------------------------------------------------------

    def _build_motion_group(self):
        group = QGroupBox("Motion")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignLeft)

        self.mode_combo = QComboBox()
        for mode in MODES:
            self.mode_combo.addItem(MODE_LABELS[mode], mode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Pattern", self.mode_combo)

        self.speed_slider = _slider(10, 200, 100)
        self.speed_value = QLabel("1.00x")
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        form.addRow("Speed", self._with_value(self.speed_slider, self.speed_value))

        self.amplitude_slider = _slider(5, 120, 60)
        self.amplitude_value = QLabel("0.60x")
        self.amplitude_slider.valueChanged.connect(self._on_amplitude_changed)
        form.addRow("Amplitude",
                    self._with_value(self.amplitude_slider, self.amplitude_value))

        return group

    def _build_joint_group(self):
        group = QGroupBox("Joint targets")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(6, 6, 6, 6)

        self.joint_hint = QLabel("Switch to manual joint sliders to command "
                                 "individual axes.")
        self.joint_hint.setWordWrap(True)
        self.joint_hint.setObjectName("hint")
        outer.addWidget(self.joint_hint)

        self._joint_container = QWidget()
        self._joint_layout = QGridLayout(self._joint_container)
        self._joint_layout.setContentsMargins(0, 4, 0, 0)
        self._joint_layout.setVerticalSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self._joint_container)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)
        return group

    def _build_controller_group(self):
        group = QGroupBox("Controller")
        form = QFormLayout(group)

        self.position_gain_slider = _slider(5, 100,
                                            int(config.DEFAULT_POSITION_GAIN * 100))
        self.position_gain_value = QLabel("%.2f" % config.DEFAULT_POSITION_GAIN)
        self.position_gain_slider.valueChanged.connect(self._on_gains_changed)
        form.addRow("Position gain",
                    self._with_value(self.position_gain_slider,
                                     self.position_gain_value))

        self.velocity_gain_slider = _slider(10, 300,
                                            int(config.DEFAULT_VELOCITY_GAIN * 100))
        self.velocity_gain_value = QLabel("%.2f" % config.DEFAULT_VELOCITY_GAIN)
        self.velocity_gain_slider.valueChanged.connect(self._on_gains_changed)
        form.addRow("Velocity gain",
                    self._with_value(self.velocity_gain_slider,
                                     self.velocity_gain_value))
        return group

    def _build_view_group(self):
        group = QGroupBox("View")
        outer = QVBoxLayout(group)

        grid = QGridLayout()
        grid.setSpacing(4)
        names = list(config.CAMERA_PRESETS.keys())
        for index, name in enumerate(names):
            button = QPushButton(name)
            button.setProperty("preset", name)
            button.clicked.connect(self._on_preset_clicked)
            grid.addWidget(button, index // 3, index % 3)
        outer.addLayout(grid)

        row = QHBoxLayout()
        row.addWidget(QLabel("Render quality"))
        self.quality_combo = QComboBox()
        for name in config.QUALITY_ORDER:
            self.quality_combo.addItem(name.title(), name)
        self.quality_combo.setCurrentIndex(
            config.QUALITY_ORDER.index(config.DEFAULT_QUALITY))
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)
        row.addWidget(self.quality_combo, 1)
        outer.addLayout(row)
        return group

    @staticmethod
    def _with_value(slider, label):
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        label.setMinimumWidth(64)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(slider, 1)
        row.addWidget(label)
        return holder

    # -- population ---------------------------------------------------------

    def build_joint_sliders(self, description):
        """Called once, after the worker reports what it actually loaded."""
        names = description["names"]
        lower = description["lower"]
        upper = description["upper"]
        home = description["home"]

        self._joint_sliders = []
        self._joint_labels = []
        self._joint_ranges = []

        for index, name in enumerate(names):
            low = lower[index] + config.JOINT_LIMIT_MARGIN
            high = upper[index] - config.JOINT_LIMIT_MARGIN
            self._joint_ranges.append((low, high))

            caption = QLabel("%d  %s" % (index + 1, name))
            caption.setObjectName("jointName")
            slider = _slider(0, config.SLIDER_TICKS,
                             self._to_ticks(home[index], low, high))
            slider.setEnabled(False)
            slider.valueChanged.connect(self._on_joint_changed)
            value = QLabel("%+.3f" % home[index])
            value.setMinimumWidth(56)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            row = index * 2
            self._joint_layout.addWidget(caption, row, 0, 1, 2)
            self._joint_layout.addWidget(slider, row + 1, 0)
            self._joint_layout.addWidget(value, row + 1, 1)

            self._joint_sliders.append(slider)
            self._joint_labels.append(value)

        self._joint_layout.setRowStretch(len(names) * 2, 1)

    def follow_measured_positions(self, positions):
        """
        Keep the sliders on the live pose while another pattern is running, so
        switching to manual control never produces a jump.
        """
        if self._mode == MODE_MANUAL or not self._joint_sliders:
            return
        self._updating = True
        for index, slider in enumerate(self._joint_sliders):
            if index >= len(positions):
                break
            low, high = self._joint_ranges[index]
            slider.setValue(self._to_ticks(positions[index], low, high))
            self._joint_labels[index].setText("%+.3f" % positions[index])
        self._updating = False

    @staticmethod
    def _to_ticks(value, low, high):
        span = high - low
        if span <= 0.0:
            return 0
        ratio = (value - low) / span
        return int(round(config.clamp(ratio, 0.0, 1.0) * config.SLIDER_TICKS))

    @staticmethod
    def _from_ticks(ticks, low, high):
        return low + (float(ticks) / config.SLIDER_TICKS) * (high - low)

    # -- events -------------------------------------------------------------

    def _on_mode_changed(self, index):
        mode = self.mode_combo.itemData(index)
        self._mode = mode
        manual = mode == MODE_MANUAL
        for slider in self._joint_sliders:
            slider.setEnabled(manual)
        self.joint_hint.setVisible(not manual)
        self.command.emit("set_mode", {"mode": mode})

    def _on_speed_changed(self, value):
        speed = value / 100.0
        self.speed_value.setText("%.2fx" % speed)
        self.command.emit("set_speed", {"value": speed})

    def _on_amplitude_changed(self, value):
        amplitude = value / 100.0
        self.amplitude_value.setText("%.2fx" % amplitude)
        self.command.emit("set_amplitude", {"value": amplitude})

    def _on_gains_changed(self, _value):
        position_gain = self.position_gain_slider.value() / 100.0
        velocity_gain = self.velocity_gain_slider.value() / 100.0
        self.position_gain_value.setText("%.2f" % position_gain)
        self.velocity_gain_value.setText("%.2f" % velocity_gain)
        self.command.emit("set_gains", {"position_gain": position_gain,
                                        "velocity_gain": velocity_gain})

    def _on_joint_changed(self, _value):
        if self._updating or self._mode != MODE_MANUAL:
            return
        targets = []
        for index, slider in enumerate(self._joint_sliders):
            low, high = self._joint_ranges[index]
            value = self._from_ticks(slider.value(), low, high)
            targets.append(value)
            self._joint_labels[index].setText("%+.3f" % value)
        self.command.emit("set_manual_targets", {"targets": targets})

    def _on_preset_clicked(self):
        button = self.sender()
        if button is None:
            return
        self.command.emit("camera_preset", {"name": button.property("preset")})

    def _on_quality_changed(self, index):
        self.command.emit("set_quality",
                          {"quality": self.quality_combo.itemData(index)})


class TelemetryPanel(QWidget):
    """Joint table, rolling performance figures and the event log."""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setMinimumHeight(190)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_table(), 4)
        layout.addWidget(self._build_performance(), 2)
        layout.addWidget(self._build_log(), 3)

    def _build_table(self):
        group = QGroupBox("Joint telemetry")
        box = QVBoxLayout(group)
        box.setContentsMargins(6, 6, 6, 6)

        self.table = QTableWidget(0, len(config.TELEMETRY_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(config.TELEMETRY_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        box.addWidget(self.table)
        return group

    def _build_performance(self):
        group = QGroupBox("Performance")
        form = QFormLayout(group)
        form.setContentsMargins(8, 6, 8, 6)
        form.setVerticalSpacing(2)

        self.perf_labels = {}
        rows = (
            ("physics_hz", "Physics"),
            ("render_fps", "Rendering"),
            ("real_time_factor", "Real time factor"),
            ("step_ms", "Physics step"),
            ("render_ms", "Render time"),
            ("sim_time", "Simulated time"),
            ("resolution", "Frame size"),
            ("renderer", "Renderer"),
        )
        for key, caption in rows:
            label = QLabel("-")
            label.setObjectName("metric")
            self.perf_labels[key] = label
            form.addRow(caption, label)
        return group

    def _build_log(self):
        group = QGroupBox("Events")
        box = QVBoxLayout(group)
        box.setContentsMargins(6, 6, 6, 6)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(config.EVENT_LOG_MAX_LINES)
        self.log.setObjectName("eventLog")
        box.addWidget(self.log)
        return group

    # -- updates ------------------------------------------------------------

    def build_rows(self, description):
        names = description["names"]
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            for column in range(len(config.TELEMETRY_COLUMNS)):
                item = QTableWidgetItem("")
                if column == 0:
                    item.setText("%d  %s" % (row + 1, name))
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, column, item)

    def update_telemetry(self, snapshot):
        """Existing items are rewritten in place; nothing is reallocated."""
        positions = snapshot["position"]
        targets = snapshot["target"]
        errors = snapshot["error"]
        velocities = snapshot["velocity"]
        torques = snapshot["torque"]
        powers = snapshot["power"]

        for row in range(self.table.rowCount()):
            if row >= len(positions):
                break
            self.table.item(row, 1).setText("%+.4f" % positions[row])
            self.table.item(row, 2).setText("%+.4f" % targets[row])
            self.table.item(row, 3).setText("%+.4f" % errors[row])
            self.table.item(row, 4).setText("%+.3f" % velocities[row])
            self.table.item(row, 5).setText("%+.2f" % torques[row])
            self.table.item(row, 6).setText("%+.2f" % powers[row])

    def update_stats(self, stats):
        labels = self.perf_labels
        labels["physics_hz"].setText("%.1f Hz" % stats["physics_hz"])
        labels["render_fps"].setText("%.1f FPS (target %.0f)"
                                     % (stats["render_fps"], stats["target_fps"]))
        labels["real_time_factor"].setText("%.3fx" % stats["real_time_factor"])
        labels["step_ms"].setText("%.3f ms" % stats["step_ms"])
        labels["render_ms"].setText("%.2f ms" % stats["render_ms"])
        labels["sim_time"].setText("%.2f s" % stats["sim_time"])
        labels["resolution"].setText(stats["resolution"])
        labels["renderer"].setText(stats["renderer"])

    def append_log(self, text):
        self.log.appendPlainText(text)
