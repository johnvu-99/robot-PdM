"""
Robot model selection and control mode.

Model discovery walks the filesystem, so it runs on its own short lived
QThread. The simulation thread is never asked to scan folders: a slow disk
must not be able to stall physics.
"""

import os

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QWidget,
)

import config
from simulator.joint_controller import CONTROL_MODES
from simulator.robot_loader import scan_models

CONTROL_MODE_LABELS = {
    "POSITION_CONTROL": "Position control",
    "VELOCITY_CONTROL": "Velocity control",
    "TORQUE_CONTROL": "Torque control (computed torque)",
}


class ModelDiscoveryWorker(QObject):

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    @pyqtSlot()
    def run(self):
        try:
            models = scan_models()
        except (OSError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(models)


def _normalise(path):
    return os.path.normcase(os.path.normpath(path)) if path else ""


class RobotPanel(QGroupBox):

    command = pyqtSignal(str, object)
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        QGroupBox.__init__(self, "Robot", parent)
        self._models = []
        self._loaded_path = config.URDF_PATH
        self._thread = None
        self._worker = None

        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignLeft)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumContentsLength(18)
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        form.addRow("Model", self.model_combo)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self._on_load_clicked)
        self.rescan_button = QPushButton("Rescan")
        self.rescan_button.clicked.connect(self.start_discovery)
        self.articulated_only = QCheckBox("Robots only")
        self.articulated_only.setChecked(True)
        self.articulated_only.setToolTip(
            "Hide files that declare no revolute or prismatic joints.")
        self.articulated_only.toggled.connect(self._populate)
        row_layout.addWidget(self.load_button)
        row_layout.addWidget(self.rescan_button)
        row_layout.addWidget(self.articulated_only, 1)
        form.addRow("", row)

        self.control_combo = QComboBox()
        for mode in CONTROL_MODES:
            self.control_combo.addItem(CONTROL_MODE_LABELS.get(mode, mode), mode)
        self.control_combo.setCurrentIndex(
            list(CONTROL_MODES).index(config.DEFAULT_CONTROL_MODE))
        self.control_combo.currentIndexChanged.connect(self._on_control_changed)
        form.addRow("Control", self.control_combo)

        self.info_label = QLabel("Scanning models...")
        self.info_label.setObjectName("hint")
        self.info_label.setWordWrap(True)
        form.addRow(self.info_label)

    # -- discovery ----------------------------------------------------------

    def start_discovery(self):
        if self._thread is not None:
            return
        self.rescan_button.setEnabled(False)
        self.info_label.setText("Scanning models...")
        self._thread = QThread(self)
        self._worker = ModelDiscoveryWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_discovered)
        self._worker.failed.connect(self._on_discovery_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_thread_finished(self):
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.rescan_button.setEnabled(True)

    def _on_discovered(self, models):
        self._models = models
        present = sum(1 for m in models if m.exists)
        missing = [m.relative_path for m in models if m.required and not m.exists]
        self._populate()
        text = "%d model files found." % present
        if missing:
            text += " Missing: %s" % ", ".join(missing)
            self.log.emit("Required models not found: %s" % ", ".join(missing))
        self.info_label.setText(text)
        self.log.emit("Model discovery: %d files." % present)

    def _on_discovery_failed(self, message):
        self.info_label.setText("Model scan failed: %s" % message)
        self.log.emit("Model scan failed: %s" % message)

    def shutdown(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)

    # -- population ---------------------------------------------------------

    def _populate(self, _checked=None):
        only_robots = self.articulated_only.isChecked()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        model = self.model_combo.model()
        for info in self._models:
            if only_robots and not info.required and not info.articulated:
                continue
            self.model_combo.addItem(info.label(), info.path)
            item = model.item(self.model_combo.count() - 1)
            if not info.exists:
                item.setEnabled(False)
        self.model_combo.blockSignals(False)
        self._select_loaded()

    def _select_loaded(self):
        target = _normalise(self._loaded_path)
        for index in range(self.model_combo.count()):
            path = self.model_combo.itemData(index)
            if not path:
                continue
            if _normalise(path) == target or _normalise(path).endswith(
                    _normalise(os.sep + self._loaded_path)):
                self.model_combo.setCurrentIndex(index)
                return

    def set_loaded(self, description):
        self._loaded_path = description.get("urdf", self._loaded_path)
        self._select_loaded()
        mode = description.get("control_mode")
        if mode in CONTROL_MODES:
            self.control_combo.blockSignals(True)
            self.control_combo.setCurrentIndex(list(CONTROL_MODES).index(mode))
            self.control_combo.blockSignals(False)

    # -- events -------------------------------------------------------------

    def _on_load_clicked(self):
        path = self.model_combo.currentData()
        if not path or not os.path.isfile(path):
            self.log.emit("Selected model file does not exist.")
            return
        self.command.emit("load_robot", {"path": path})

    def _on_control_changed(self, index):
        self.command.emit("set_control_mode",
                          {"mode": self.control_combo.itemData(index)})
