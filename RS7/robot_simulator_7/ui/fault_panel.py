"""
Fault injection, degradation and dataset generation controls.

Pure view code plus the dataset generator process handle. Faults and scenarios
are commands to the simulation thread; the dataset generator runs in a
separate process so it never shares the GIL with physics.
"""

import multiprocessing
import os
import queue

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QProgressBar, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

import config
from simulator import dataset_generator
from simulator.fault_injector import (
    FAULT_CAPTIONS, FAULT_TYPES, MOTOR_WEAKNESS, describe_severity,
)


def _spin(value, low, high, step, suffix, decimals=1):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


class FaultPanel(QWidget):

    command = pyqtSignal(str, object)
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._joint_names = []
        self._process = None
        self._queue = None
        self._cancel = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_fault_group())
        layout.addWidget(self._build_degradation_group())
        layout.addWidget(self._build_dataset_group())

        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._poll_generator)

    # -- construction -------------------------------------------------------

    def _build_fault_group(self):
        group = QGroupBox("Fault injection")
        outer = QVBoxLayout(group)
        form = QFormLayout()

        self.joint_combo = QComboBox()
        form.addRow("Joint", self.joint_combo)

        self.type_combo = QComboBox()
        for fault_type in FAULT_TYPES:
            self.type_combo.addItem(FAULT_CAPTIONS[fault_type], fault_type)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("Fault type", self.type_combo)

        self.severity_slider = QSlider(Qt.Horizontal)
        self.severity_slider.setRange(0, 100)
        self.severity_slider.setValue(50)
        self.severity_slider.valueChanged.connect(self._update_severity_text)
        self.weakness_combo = QComboBox()
        for level in config.MOTOR_WEAKNESS_LEVELS:
            self.weakness_combo.addItem("%d%% torque available" % level, 1.0 - level / 100.0)
        self.weakness_combo.setCurrentIndex(2)
        self.weakness_combo.currentIndexChanged.connect(self._update_severity_text)
        severity_holder = QWidget()
        severity_row = QHBoxLayout(severity_holder)
        severity_row.setContentsMargins(0, 0, 0, 0)
        severity_row.addWidget(self.severity_slider, 1)
        severity_row.addWidget(self.weakness_combo, 1)
        form.addRow("Severity", severity_holder)
        self.severity_text = QLabel("")
        self.severity_text.setObjectName("hint")
        form.addRow("", self.severity_text)

        self.start_spin = _spin(0.0, 0.0, 3600.0, 1.0, " s from now")
        form.addRow("Start time", self.start_spin)
        self.duration_spin = _spin(0.0, 0.0, 3600.0, 5.0, " s")
        self.duration_spin.setSpecialValueText("permanent")
        form.addRow("Duration", self.duration_spin)
        outer.addLayout(form)

        row = QHBoxLayout()
        inject = QPushButton("Inject")
        inject.setObjectName("start")
        inject.clicked.connect(self._on_inject)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._on_remove)
        clear = QPushButton("Clear all")
        clear.clicked.connect(lambda: self.command.emit("clear_faults", {}))
        row.addWidget(inject)
        row.addWidget(remove)
        row.addWidget(clear)
        outer.addLayout(row)

        self.active_list = QListWidget()
        self.active_list.setMaximumHeight(110)
        self.active_list.setObjectName("eventLog")
        outer.addWidget(self.active_list)

        self._on_type_changed(0)
        return group

    def _build_degradation_group(self):
        group = QGroupBox("Degradation scenario")
        form = QFormLayout(group)
        self.scenario_combo = QComboBox()
        for name in sorted(config.DEGRADATION_SCENARIOS.keys()):
            mechanisms = config.DEGRADATION_SCENARIOS[name]["mechanisms"]
            self.scenario_combo.addItem(name.replace("_", " ").title(), name)
            self.scenario_combo.setItemData(self.scenario_combo.count() - 1,
                                            ", ".join(sorted(mechanisms)), Qt.ToolTipRole)
        form.addRow("Scenario", self.scenario_combo)
        self.degradation_joint_combo = QComboBox()
        form.addRow("Joint", self.degradation_joint_combo)
        self.degradation_duration = _spin(config.DEGRADATION_DEFAULT_DURATION_S,
                                          5.0, 7200.0, 10.0, " s", 0)
        form.addRow("Duration", self.degradation_duration)
        self.final_health = _spin(config.DEGRADATION_DEFAULT_FINAL_HEALTH,
                                  0.0, 95.0, 5.0, " % health", 0)
        form.addRow("Final health", self.final_health)
        start = QPushButton("Start degradation")
        start.clicked.connect(self._on_degradation)
        form.addRow(start)
        return group

    def _build_dataset_group(self):
        group = QGroupBox("Dataset generator")
        outer = QVBoxLayout(group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        for name in ("QUICK", "STANDARD"):
            runs = len(dataset_generator.build_plan(name))
            self.preset_combo.addItem("%s (%d runs)" % (name.title(), runs), name)
        row.addWidget(self.preset_combo, 1)
        outer.addLayout(row)

        buttons = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self._on_generate)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        buttons.addWidget(self.generate_button)
        buttons.addWidget(self.cancel_button)
        outer.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        outer.addWidget(self.progress)
        self.dataset_status = QLabel("Writes data/simulation/normal and faulty.")
        self.dataset_status.setObjectName("hint")
        self.dataset_status.setWordWrap(True)
        outer.addWidget(self.dataset_status)
        return group

    # -- population ---------------------------------------------------------

    def build(self, description):
        self._joint_names = description["names"]
        for combo in (self.joint_combo, self.degradation_joint_combo):
            current = combo.currentIndex()
            combo.clear()
            for index, name in enumerate(self._joint_names):
                combo.addItem("%d  %s" % (index + 1, name), index)
            combo.setCurrentIndex(current if 0 <= current < combo.count() else min(1, combo.count() - 1))

    def update_faults(self, faults, sim_time):
        selected = self._selected_id()
        self.active_list.clear()
        for info in faults:
            text = "#%d  J%d  %s  %s  [%s]" % (
                info["id"], info["joint"] + 1,
                FAULT_CAPTIONS.get(info["type"], info["type"]),
                info["meaning"], info["status"])
            if info["kind"] == "degradation":
                text += "  health %.0f%%" % info.get("health", 100.0)
            elif info["status"] == "scheduled":
                text += "  in %.1f s" % (info["start"] - sim_time)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, info["id"])
            self.active_list.addItem(item)
            if info["id"] == selected:
                item.setSelected(True)

    def _selected_id(self):
        items = self.active_list.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    # -- events -------------------------------------------------------------

    def _severity(self):
        if self.type_combo.currentData() == MOTOR_WEAKNESS:
            return float(self.weakness_combo.currentData())
        return self.severity_slider.value() / 100.0

    def _on_type_changed(self, _index):
        weakness = self.type_combo.currentData() == MOTOR_WEAKNESS
        self.weakness_combo.setVisible(weakness)
        self.severity_slider.setVisible(not weakness)
        self._update_severity_text()

    def _update_severity_text(self, _value=None):
        fault_type = self.type_combo.currentData()
        severity = self._severity()
        self.severity_text.setText("%.0f%%: %s" % (100.0 * severity,
                                                   describe_severity(fault_type, severity)))

    def _on_inject(self):
        if self.joint_combo.count() == 0:
            return
        self.command.emit("inject_fault", {
            "joint": self.joint_combo.currentData(),
            "type": self.type_combo.currentData(),
            "severity": self._severity(),
            "start_delay": self.start_spin.value(),
            "duration": self.duration_spin.value(),
        })

    def _on_remove(self):
        item_id = self._selected_id()
        if item_id is not None:
            self.command.emit("remove_fault", {"id": item_id})

    def _on_degradation(self):
        if self.degradation_joint_combo.count() == 0:
            return
        self.command.emit("start_degradation", {
            "scenario": self.scenario_combo.currentData(),
            "joint": self.degradation_joint_combo.currentData(),
            "duration": self.degradation_duration.value(),
            "final_health": self.final_health.value(),
            "start_delay": 0.0,
        })

    # -- dataset generator process -----------------------------------------

    def _on_generate(self):
        if self._process is not None:
            return
        preset = self.preset_combo.currentData()
        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue()
        self._cancel = context.Event()
        self._process = context.Process(
            target=dataset_generator.process_main,
            args=(preset, config.SIMULATION_DATA_DIR, self._queue, self._cancel,
                  config.DEFAULT_CONTROL_MODE),
            name="DatasetGenerator")
        self._process.daemon = True
        self._process.start()
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, len(dataset_generator.build_plan(preset)))
        self.progress.setValue(0)
        self.dataset_status.setText("Starting generator process...")
        self.log.emit("Dataset generation started (%s) in a separate process." % preset)
        self._poll.start()

    def _on_cancel(self):
        if self._cancel is not None:
            self._cancel.set()
            self.dataset_status.setText("Cancelling after the current episode step...")

    def _poll_generator(self):
        if self._queue is None:
            return
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            kind = message.get("kind")
            if kind == "progress":
                self.progress.setValue(message["index"])
                self.dataset_status.setText("Run %d / %d: %s" % (
                    message["index"] + 1, message["total"], message["name"]))
            elif kind == "finished":
                self.progress.setValue(message["total"])
                text = "Done: %d runs in %.0f s. Manifest %s" % (
                    message["total"], message["seconds"], os.path.basename(message["manifest"]))
                self.dataset_status.setText(text)
                self.log.emit("Dataset " + text)
                self._finish_process()
                return
            elif kind == "cancelled":
                self.dataset_status.setText("Cancelled after %d runs. No manifest written." % message["done"])
                self.log.emit("Dataset generation cancelled.")
                self._finish_process()
                return
            elif kind == "error":
                self.dataset_status.setText("Failed: %s" % message["message"])
                self.log.emit("Dataset generation failed: %s" % message["message"])
                self._finish_process()
                return
        if self._process is not None and not self._process.is_alive():
            self.dataset_status.setText("Generator process exited unexpectedly (code %s)."
                                        % self._process.exitcode)
            self._finish_process()

    def _finish_process(self):
        self._poll.stop()
        if self._process is not None:
            self._process.join(2.0)
        self._process = None
        self._queue = None
        self._cancel = None
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def shutdown(self):
        if self._process is None:
            return
        self._cancel.set()
        self._process.join(3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(1.0)
        self._process = None
