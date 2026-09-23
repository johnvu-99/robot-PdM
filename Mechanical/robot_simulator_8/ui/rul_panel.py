"""
RUL tab: real dataset loading, feature extraction, RUL training and evaluation,
and fault feature validation on Mechanical-datasets.

All work runs in a child process (predictive.dataset_jobs); this widget only
starts jobs, polls their queue and draws results.
"""

import multiprocessing
import os
import queue

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import config
from predictive import dataset_jobs

DISCLAIMER = ("Real rotating machinery data (gearboxes, rolling bearings). This is NOT KUKA "
              "robot telemetry: it is used to develop and validate fault features and the RUL "
              "method. A KUKA specific model would need real KUKA sensor data.")

METRIC_COLUMNS = ("Run", "Condition", "Life (h)", "MAE", "RMSE", "R2", "MAE % life",
                  "Late life time error % life")


class JobRunner(QObject):

    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self._process = None
        self._queue = None
        self._cancel = None
        self._job = ""
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self):
        return self._process is not None

    def start(self, job, params):
        if self.busy:
            return False
        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue()
        self._cancel = context.Event()
        self._job = job
        self._process = context.Process(target=dataset_jobs.process_main,
                                        args=(job, params, self._queue, self._cancel),
                                        name="DatasetJob")
        self._process.daemon = True
        self._process.start()
        self._timer.start()
        return True

    def cancel(self):
        if self._cancel is not None:
            self._cancel.set()

    def _poll(self):
        while self._queue is not None:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            kind = message.get("kind")
            if kind == "progress":
                self.progress.emit(message.get("text", ""), message.get("done", 0),
                                   message.get("total", 0))
            elif kind == "finished":
                self._end()
                self.finished.emit(message["job"], message["result"])
                return
            elif kind in ("error", "cancelled"):
                self._end()
                self.failed.emit(message.get("job", ""), message.get("message", "cancelled"))
                return
        if self._process is not None and not self._process.is_alive():
            job = self._job
            code = self._process.exitcode
            self._end()
            self.failed.emit(job, "worker process exited (code %s)" % code)

    def _end(self):
        self._timer.stop()
        if self._process is not None:
            self._process.join(2.0)
        self._process = None
        self._queue = None
        self._cancel = None

    def shutdown(self):
        if self._process is None:
            return
        self._cancel.set()
        self._process.join(3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(1.0)
        self._end()


class RULPanel(QWidget):

    log = pyqtSignal(str)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.runner = JobRunner(self)
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self._on_failed)
        self._curves = []
        self._model_path = ""
        self._buttons = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        disclaimer = QLabel(DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: #d9a326;")
        layout.addWidget(disclaimer)

        roots = QGridLayout()
        roots.setHorizontalSpacing(6)
        roots.setVerticalSpacing(2)
        self.root_edits = {}
        for row, (key, caption, default) in enumerate((
                ("MECHANICAL", "Mechanical-datasets", config.MECHANICAL_DATASET_DIR),
                ("XJTU_SY", "XJTU-SY", config.XJTU_SY_DIR),
                ("PHM2012", "PHM 2012", config.PHM2012_DIR))):
            edit = QLineEdit(default)
            browse = QPushButton("...")
            browse.setFixedWidth(32)
            browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
            roots.addWidget(QLabel(caption), row, 0)
            roots.addWidget(edit, row, 1)
            roots.addWidget(browse, row, 2)
            self.root_edits[key] = edit
        layout.addLayout(roots)

        controls = QHBoxLayout()
        controls.setSpacing(4)
        for caption, handler in (("Load Dataset", lambda: self._start("load")),
                                 ("Preprocess Dataset", lambda: self._start("preprocess")),
                                 ("Extract Features", lambda: self._start("extract")),
                                 ("Fault Validation", lambda: self._start("mechanical"))):
            button = QPushButton(caption)
            button.clicked.connect(handler)
            controls.addWidget(button)
            self._buttons.append(button)
        controls.addSpacing(8)
        self.protocol_combo = QComboBox()
        for key, text in config.RUL_PROTOCOLS.items():
            self.protocol_combo.addItem(text, key)
        controls.addWidget(self.protocol_combo, 1)
        self.model_combo = QComboBox()
        for kind in config.RUL_MODELS:
            self.model_combo.addItem(kind.replace("_", " ").title(), kind)
        self.model_combo.setCurrentIndex(list(config.RUL_MODELS).index(config.RUL_DEFAULT_MODEL))
        controls.addWidget(self.model_combo)
        for caption, handler in (("Train RUL Model", self._train),
                                 ("Evaluate RUL Model", self._evaluate),
                                 ("Load Saved Model", self._load_model)):
            button = QPushButton(caption)
            button.clicked.connect(handler)
            controls.addWidget(button)
            self._buttons.append(button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.runner.cancel)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)

        status_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setMaximumWidth(220)
        self.status = QLabel("No model loaded.")
        self.status.setObjectName("hint")
        status_row.addWidget(self.progress)
        status_row.addWidget(self.status, 1)
        layout.addLayout(status_row)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(METRIC_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(METRIC_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_run_selected)
        left_layout.addWidget(self.table, 2)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setObjectName("eventLog")
        left_layout.addWidget(self.report, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "operating time", units="h")
        self.plot.setLabel("left", "fraction")
        self.plot.setYRange(-0.05, 1.05)
        self.plot.addLegend(offset=(-10, 6))
        self.true_curve = self.plot.plot(pen=pg.mkPen("#97a0ad", width=2), name="True RUL fraction")
        self.pred_curve = self.plot.plot(pen=pg.mkPen("#4fa3e0", width=2), name="Predicted RUL fraction")
        self.hi_curve = self.plot.plot(pen=pg.mkPen("#d9a326", width=1), name="Health indicator")
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#e0474c"))
        self.plot.addItem(self.cursor)
        right_layout.addWidget(self.plot, 1)
        cursor_row = QHBoxLayout()
        self.life_slider = QSlider(Qt.Horizontal)
        self.life_slider.setRange(0, 1000)
        self.life_slider.setValue(700)
        self.life_slider.valueChanged.connect(self._update_readout)
        cursor_row.addWidget(QLabel("Point in life"))
        cursor_row.addWidget(self.life_slider, 1)
        right_layout.addLayout(cursor_row)
        self.readout = QLabel("")
        self.readout.setObjectName("metric")
        right_layout.addWidget(self.readout)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

    # -- job control --------------------------------------------------------

    def _roots(self):
        return dict((key, edit.text().strip()) for key, edit in self.root_edits.items())

    def _browse(self, edit):
        folder = QFileDialog.getExistingDirectory(self, "Dataset folder", edit.text())
        if folder:
            edit.setText(folder)

    def _start(self, job, extra=None):
        params = {"roots": self._roots()}
        params.update(extra or {})
        if not self.runner.start(job, params):
            self.status.setText("A dataset job is already running.")
            return
        for button in self._buttons:
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Running %s in a separate process..." % job)
        self.log.emit("Dataset job started: %s" % job)

    def _train(self):
        self._start("train_rul", {"protocol": self.protocol_combo.currentData(),
                                  "model": self.model_combo.currentData()})

    def _evaluate(self):
        if not self._model_path:
            self.status.setText("Train or load a model first.")
            return
        self._start("evaluate_rul", {"protocol": self.protocol_combo.currentData(),
                                     "model_path": self._model_path})

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load RUL model", config.MODEL_DIR,
                                              "RUL models (rul_*.joblib);;All joblib (*.joblib)")
        if path:
            self._model_path = path
            self.status.setText("Model: %s" % os.path.basename(path))
            self._evaluate()

    def _idle(self):
        for button in self._buttons:
            button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)

    def _on_progress(self, text, done, total):
        self.status.setText(text)
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(done)

    def _on_failed(self, job, message):
        self._idle()
        self.status.setText("%s failed: %s" % (job, message.splitlines()[0] if message else ""))
        self.report.appendPlainText("[%s] %s" % (job, message))
        self.log.emit("Dataset job %s failed: %s" % (job, message.splitlines()[0] if message else ""))

    def _on_finished(self, job, result):
        self._idle()
        seconds = result.get("seconds", 0.0)
        self.status.setText("%s finished in %.1f s." % (job, seconds))
        self.log.emit("Dataset job %s finished in %.1f s." % (job, seconds))
        handler = getattr(self, "_show_" + job, None)
        if handler is not None:
            handler(result)

    # -- results ------------------------------------------------------------

    def _show_load(self, result):
        lines = []
        for name, s in result["summaries"].items():
            if not s["available"]:
                lines.append("%s: not found at %s" % (name, s["root"]))
                continue
            lines.append("%s (%s): %d recordings, %d files. %s" % (
                name, s["role"], s["recordings"], s["files"], s["system"]))
            lines.append("    labels %s" % s["labels"])
            lines.append("    conditions %s" % s["conditions"])
        self.report.setPlainText("\n".join(lines))

    def _show_preprocess(self, result):
        lines = []
        for name, r in result["preprocess"].items():
            if not r.get("available"):
                lines.append("%s: not available" % name)
                continue
            lines.append("%s: %d recordings checked, %d problems, index %s"
                         % (name, r["recordings"], len(r["problems"]), r["index"]))
            lines.extend("    " + p for p in r["problems"][:10])
        self.report.setPlainText("\n".join(lines))

    def _show_extract(self, result):
        self.report.setPlainText("\n".join("%s: %s" % (k, v) for k, v in result["extract"].items())
                                 + "\nFeatures cached in %s" % config.PROCESSED_DIR)

    def _show_mechanical(self, result):
        lines = ["Mechanical-datasets fault feature validation (split by contiguous time "
                 "blocks within each recording, scaler fitted on training blocks):"]
        for r in result["mechanical"]:
            lines.append("%s: %d recordings, %d features, train %d / test %d segments" % (
                r["source"], r["recordings"], r["features"], r["train_segments"], r["test_segments"]))
            if "accuracy" in r:
                lines.append("    classification accuracy %.3f, macro F1 %.3f" % (r["accuracy"], r["macro_f1"]))
                lines.append("    top features: " + ", ".join("%s %.3f" % f for f in r["top_features"][:5]))
            for a in r.get("anomaly", []):
                lines.append("    anomaly [%s] false alarms %.0f%%" % (a["setup"], 100 * a["false_alarm_rate"]))
                for label, v in sorted(a["labels"].items()):
                    lines.append("        %-30s detected %.0f%%  ROC AUC %.2f"
                                 % (label, 100 * v["detection_rate"], v["roc_auc"]))
        self.report.setPlainText("\n".join(lines))

    def _show_train_rul(self, result):
        self._model_path = result["model_path"]
        lines = ["Trained %s (%s) on %d runs: %s" % (result["kind"], result["protocol"],
                                                    len(result["training_runs"]), ", ".join(result["training_runs"])),
                 "Validation runs (never used for training): %s" % ", ".join(result["validation_runs"]),
                 "Saved: %s" % result["model_path"]]
        if result["cv"]:
            s = result["cv_summary"]
            lines.append("Leave one run out CV on training data: MAE %.3f  RMSE %.3f  R2 %.2f"
                         % (s.get("mae", np.nan), s.get("rmse", np.nan), s.get("r2", np.nan)))
        self.report.setPlainText("\n".join(lines))
        self._fill_table(result["cv"], [])
        self.status.setText("Model trained: %s. Press Evaluate RUL Model." % os.path.basename(result["model_path"]))

    def _show_evaluate_rul(self, result):
        s = result["summary"]
        self.report.setPlainText(
            "Validation of %s trained on %s (%s)\nruns %d  MAE %.3f  RMSE %.3f  R2 %.2f  MAE %.1f%% of life\n"
            "RUL fraction target; time errors use elapsed x f / (1 - f) and are reported from %.0f%% of life."
            % (result["kind"], result["training_dataset"], result["protocol"], s.get("runs", 0),
               s.get("mae", np.nan), s.get("rmse", np.nan), s.get("r2", np.nan),
               s.get("mae_percent_life", np.nan), 100 * config.RUL_TIME_METRICS_FROM_LIFE))
        self._fill_table(result["metrics"], result["curves"])
        if self.table.rowCount():
            self.table.selectRow(0)

    def _fill_table(self, metrics, curves):
        self._curves = curves
        self.table.setRowCount(len(metrics))
        for row, m in enumerate(metrics):
            values = (m["run_id"], m["condition"], "%.2f" % (m["total_life_s"] / 3600.0),
                      "%.3f" % m["mae"], "%.3f" % m["rmse"], "%.2f" % m["r2"],
                      "%.1f" % m["mae_percent_life"],
                      "%.1f" % m["late_life_time_error_percent_life"] if "late_life_time_error_percent_life" in m else "-")
            for column, text in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(text))

    def _on_run_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self._curves):
            return
        curve = self._curves[rows[0].row()]
        hours = np.array(curve["times"]) / 3600.0
        self.true_curve.setData(hours, curve["true_rul"])
        self.pred_curve.setData(hours, curve["predicted_rul"])
        self.hi_curve.setData(hours, curve["health_indicator"])
        self.plot.setTitle(curve["run_id"])
        self.plot.enableAutoRange(axis="x")
        self._update_readout()

    def _update_readout(self, _value=None):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or rows[0].row() >= len(self._curves):
            self.readout.setText("")
            return
        curve = self._curves[rows[0].row()]
        life = np.array(curve["life_fraction"])
        index = int(np.argmin(np.abs(life - self.life_slider.value() / 1000.0)))
        t = curve["times"][index]
        true_f = curve["true_rul"][index]
        pred_f = curve["predicted_rul"][index]
        self.cursor.setValue(t / 3600.0)
        text = ("t = %.2f h (%.0f%% of life)   True RUL %.3f (%.2f h)   Predicted RUL %.3f   "
                "Error %+.3f (%+.1f%% of life)   Health indicator %.2f"
                % (t / 3600.0, 100 * life[index], true_f, true_f * curve["total_life_s"] / 3600.0,
                   pred_f, pred_f - true_f, 100 * (pred_f - true_f), curve["health_indicator"][index]))
        if pred_f < 0.99 and t > 0:
            text += "   Predicted remaining time %.2f h" % (t * pred_f / (1.0 - pred_f) / 3600.0)
        self.readout.setText(text)

    def shutdown(self):
        self.runner.shutdown()
