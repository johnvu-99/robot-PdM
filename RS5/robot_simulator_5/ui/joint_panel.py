"""
Per joint controls and the static joint table.

Pure view code: sliders emit manual targets and enable flags as commands, and
the widgets repaint from the 10 Hz telemetry snapshot. Both widgets are rebuilt
from the description the worker sends after every robot load.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QGridLayout, QGroupBox, QHeaderView, QLabel,
    QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import config
from simulator.trajectory import MODE_MANUAL

JOINT_INFO_COLUMNS = ("ID", "Name", "Type", "Position", "Lower", "Upper",
                      "Max Force", "Max Velocity")


def _to_ticks(value, low, high):
    span = high - low
    if span <= 0.0:
        return 0
    ratio = (value - low) / span
    return int(round(config.clamp(ratio, 0.0, 1.0) * config.SLIDER_TICKS))


def _from_ticks(ticks, low, high):
    return low + (float(ticks) / config.SLIDER_TICKS) * (high - low)


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


class JointControlPanel(QGroupBox):
    """Slider, target, actual, velocity, torque and enable for each joint."""

    command = pyqtSignal(str, object)

    def __init__(self, parent=None):
        QGroupBox.__init__(self, "Joints", parent)
        self._rows = []
        self._mode = None
        self._updating = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        self.hint = QLabel("Select Manual Sliders to drag individual axes. "
                           "Unchecked joints hold their position.")
        self.hint.setWordWrap(True)
        self.hint.setObjectName("hint")
        outer.addWidget(self.hint)

        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setContentsMargins(0, 2, 0, 0)
        self._grid.setVerticalSpacing(1)
        self._grid.setHorizontalSpacing(6)
        outer.addWidget(self._container)

    # -- construction -------------------------------------------------------

    def build(self, description):
        _clear_layout(self._grid)
        self._rows = []

        names = description["names"]
        lower = description["lower"]
        upper = description["upper"]
        home = description["home"]
        joint_ids = description.get("joint_ids", list(range(len(names))))
        enabled = description.get("enabled", [True] * len(names))

        for index, name in enumerate(names):
            low = lower[index] + config.JOINT_LIMIT_MARGIN
            high = upper[index] - config.JOINT_LIMIT_MARGIN

            check = QCheckBox("%d  %s  (id %d)" % (index + 1, name, joint_ids[index]))
            check.setObjectName("jointName")
            check.setChecked(bool(enabled[index]))
            check.toggled.connect(
                lambda state, i=index: self._on_enable_toggled(i, state))

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, config.SLIDER_TICKS)
            slider.setValue(_to_ticks(home[index], low, high))
            slider.valueChanged.connect(self._on_slider_changed)

            target = QLabel("%+.3f" % home[index])
            target.setObjectName("metric")
            target.setMinimumWidth(56)
            target.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            target.setToolTip("Target position (rad or m)")

            readout = QLabel("act %+.3f   vel %+.2f   trq %+.1f" % (home[index], 0.0, 0.0))
            readout.setObjectName("metric")
            readout.setStyleSheet("font-size: 10px; color: #8f98a5;")

            base = index * 3
            self._grid.addWidget(check, base, 0, 1, 2)
            self._grid.addWidget(slider, base + 1, 0)
            self._grid.addWidget(target, base + 1, 1)
            self._grid.addWidget(readout, base + 2, 0, 1, 2)

            self._rows.append({
                "check": check, "slider": slider, "target": target,
                "readout": readout, "range": (low, high),
            })
        self._apply_enabled_state()

    # -- state --------------------------------------------------------------

    def set_mode(self, mode):
        if mode == self._mode:
            return
        self._mode = mode
        self._apply_enabled_state()

    def _apply_enabled_state(self):
        manual = self._mode == MODE_MANUAL
        for row in self._rows:
            row["slider"].setEnabled(manual and row["check"].isChecked())
        self.hint.setVisible(not manual)

    def update_telemetry(self, snapshot):
        positions = snapshot["position"]
        targets = snapshot["target"]
        velocities = snapshot["velocity"]
        torques = snapshot["torque"]
        enabled = snapshot.get("enabled")
        manual = self._mode == MODE_MANUAL

        self._updating = True
        for index, row in enumerate(self._rows):
            if index >= len(positions):
                break
            if enabled is not None and index < len(enabled):
                if row["check"].isChecked() != bool(enabled[index]):
                    row["check"].blockSignals(True)
                    row["check"].setChecked(bool(enabled[index]))
                    row["check"].blockSignals(False)
            row["readout"].setText("act %+.3f   vel %+.2f   trq %+.1f"
                                   % (positions[index], velocities[index], torques[index]))
            slider_in_use = manual and row["slider"].isEnabled()
            if not slider_in_use:
                # Follow the live reference so entering manual mode never jumps.
                low, high = row["range"]
                row["slider"].setValue(_to_ticks(targets[index], low, high))
                row["target"].setText("%+.3f" % targets[index])
        self._updating = False
        self._apply_enabled_state()

    # -- events -------------------------------------------------------------

    def _on_enable_toggled(self, index, state):
        self._apply_enabled_state()
        self.command.emit("set_joint_enabled", {"index": index, "enabled": bool(state)})

    def _on_slider_changed(self, _value):
        if self._updating or self._mode != MODE_MANUAL:
            return
        targets = []
        for row in self._rows:
            low, high = row["range"]
            value = _from_ticks(row["slider"].value(), low, high)
            targets.append(value)
            row["target"].setText("%+.3f" % value)
        self.command.emit("set_manual_targets", {"targets": targets})


class JointInfoTable(QWidget):
    """Every joint of the loaded body, including fixed ones."""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, len(JOINT_INFO_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(JOINT_INFO_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        self._position_items = []    # (movable_index, item)

    def build(self, description):
        joints = description.get("all_joints", [])
        self.table.setRowCount(len(joints))
        self._position_items = []
        for row, joint in enumerate(joints):
            movable = joint["movable"]
            values = [
                str(joint["id"]),
                joint["name"],
                joint["type"],
                "-",
                self._limit_text(joint["lower"], joint["upper"], joint["lower"],
                                 movable and joint["limits_fallback"]),
                self._limit_text(joint["lower"], joint["upper"], joint["upper"],
                                 movable and joint["limits_fallback"]),
                self._value_text(joint["max_force"], movable and joint["force_fallback"],
                                 config.FALLBACK_MAX_FORCE),
                self._value_text(joint["max_velocity"],
                                 movable and joint["velocity_fallback"],
                                 config.FALLBACK_MAX_VELOCITY),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column in (1, 2):
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if not movable:
                    item.setForeground(Qt.gray)
                self.table.setItem(row, column, item)
            if movable:
                self._position_items.append((joint["movable_index"],
                                             self.table.item(row, 3)))

    @staticmethod
    def _limit_text(low, high, value, fallback):
        if high <= low:
            return "none" if not fallback else "none (+/-pi)"
        return "%+.3f" % value

    @staticmethod
    def _value_text(value, fallback, substitute):
        if fallback:
            return "%.1f (default %.1f)" % (value, substitute)
        return "%.2f" % value

    def update_positions(self, positions):
        for movable_index, item in self._position_items:
            if 0 <= movable_index < len(positions):
                item.setText("%+.4f" % positions[movable_index])
