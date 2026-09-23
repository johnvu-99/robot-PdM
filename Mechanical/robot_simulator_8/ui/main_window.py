"""
Main window.

Owns the simulation thread, wires signals, and polls the frame mailbox. Every
widget update happens here, on the Qt main thread.
"""

from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

import config
from rendering.pybullet_renderer import FrameMailbox
from simulator.physics_engine import (
    SimulationWorker, STATE_EMERGENCY_STOP, STATE_PAUSED, STATE_RUNNING,
    STATE_STOPPED,
)
from ui.controls_panel import ControlsPanel, TelemetryPanel
from ui.simulator_view import SimulatorView
from monitoring.monitoring_service import MonitoringService

STYLESHEET = """
QMainWindow, QWidget { background-color: #16181d; color: #d7dae0;
    font-family: "Segoe UI", "Noto Sans", sans-serif; font-size: 12px; }
QGroupBox { background-color: #1c1f26; border: 1px solid #2b2f38;
    border-radius: 3px; margin-top: 14px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #97a0ad; }
QLabel#hint { color: #7d8794; }
QLabel#jointName { color: #97a0ad; }
QLabel#metric { color: #e8ebf0; font-family: Consolas, "DejaVu Sans Mono", monospace; }
QLabel#chip { color: #c8cdd6; font-family: Consolas, "DejaVu Sans Mono", monospace;
    padding: 4px 10px; border: 1px solid #2b2f38; border-radius: 3px;
    background-color: #1c1f26; }
QLabel#stateChip { font-weight: bold; padding: 4px 14px; border-radius: 3px; }
QPushButton { background-color: #262b34; border: 1px solid #343a45;
    border-radius: 3px; padding: 6px 12px; color: #d7dae0; }
QPushButton:hover { background-color: #2e3440; }
QPushButton:pressed { background-color: #20242c; }
QPushButton:disabled { color: #5c636e; background-color: #1d2027; }
QPushButton#start { border-color: #2f7d51; }
QPushButton#start:hover { background-color: #2a6b46; }
QPushButton#estop { background-color: #7a2224; border-color: #a33033;
    font-weight: bold; color: #ffe9e9; }
QPushButton#estop:hover { background-color: #8f2a2c; }
QComboBox { background-color: #22262e; border: 1px solid #343a45;
    border-radius: 3px; padding: 4px 8px; }
QComboBox QAbstractItemView { background-color: #22262e; selection-background-color: #2f7d51; }
QSlider::groove:horizontal { height: 4px; background: #2b2f38; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 3px;
    background: #8a94a3; }
QSlider::handle:horizontal:disabled { background: #3a404b; }
QSlider::sub-page:horizontal { background: #3f6f52; border-radius: 2px; }
QTableWidget { background-color: #1a1d23; gridline-color: #262a33;
    alternate-background-color: #1d2027; border: none;
    font-family: Consolas, "DejaVu Sans Mono", monospace; }
QHeaderView::section { background-color: #22262e; color: #97a0ad; border: none;
    border-right: 1px solid #2b2f38; padding: 4px; }
QPlainTextEdit#eventLog { background-color: #14161b; border: 1px solid #262a33;
    color: #b9c0ca; font-family: Consolas, "DejaVu Sans Mono", monospace;
    font-size: 11px; }
QScrollBar:vertical { background: #1a1d23; width: 10px; }
QScrollBar::handle:vertical { background: #343a45; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QSplitter::handle { background-color: #22262e; }
"""

STATE_COLORS = {
    STATE_STOPPED: ("#2b2f38", "#c8cdd6"),
    STATE_RUNNING: ("#235c3d", "#d8f3e2"),
    STATE_PAUSED: ("#6b5312", "#f7ecd0"),
    STATE_EMERGENCY_STOP: ("#7a2224", "#ffe4e4"),
}

STATE_CAPTIONS = {
    STATE_STOPPED: "Stopped",
    STATE_RUNNING: "Running",
    STATE_PAUSED: "Paused",
    STATE_EMERGENCY_STOP: "Emergency stop",
}


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setMinimumSize(*config.WINDOW_MIN_SIZE)
        self.setStyleSheet(STYLESHEET)

        self.mailbox = FrameMailbox()
        self.monitoring = MonitoringService()
        self.monitoring.start()
        self._state = STATE_STOPPED
        self._shutting_down = False

        self._build_layout()
        self._start_worker()

        self._poll_timer = QTimer(self)
        self._poll_timer.setTimerType(Qt.PreciseTimer)
        self._poll_timer.setInterval(int(1000.0 / config.UI_POLL_HZ))
        self._poll_timer.timeout.connect(self._poll_frame)
        self._poll_timer.start()

    # -- construction -------------------------------------------------------

    def _build_layout(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.controls = ControlsPanel()
        self.view = SimulatorView()
        self.telemetry = TelemetryPanel(self.monitoring)
        self.telemetry.anomaly.snapshot_ready.connect(self._on_health)

        outer.addWidget(self._build_top_bar())

        upper = QWidget()
        upper_layout = QHBoxLayout(upper)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(0)
        upper_layout.addWidget(self.controls)
        upper_layout.addWidget(self.view, 1)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(upper)
        splitter.addWidget(self.telemetry)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setCollapsible(0, False)
        outer.addWidget(splitter, 1)

        self.setCentralWidget(central)

        self.controls.command.connect(self._send_command)
        self.controls.log.connect(self._on_log)
        self.view.command.connect(self._send_command)
        self._apply_state(STATE_STOPPED)

    def _build_top_bar(self):
        bar = QFrame()
        bar.setFixedHeight(48)
        bar.setStyleSheet("QFrame { background-color: #1c1f26; "
                          "border-bottom: 1px solid #2b2f38; }")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(lambda: self._send_command("start", {}))

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(lambda: self._send_command("pause", {}))

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(lambda: self._send_command("stop", {}))

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(lambda: self._send_command("reset", {}))

        self.estop_button = QPushButton("Emergency stop")
        self.estop_button.setObjectName("estop")
        self.estop_button.clicked.connect(
            lambda: self._send_command("emergency_stop", {}))

        for button in (self.start_button, self.pause_button, self.stop_button,
                       self.reset_button, self.estop_button):
            layout.addWidget(button)

        layout.addSpacing(12)
        self.state_chip = QLabel(STATE_CAPTIONS[STATE_STOPPED])
        self.state_chip.setObjectName("stateChip")
        layout.addWidget(self.state_chip)

        layout.addSpacing(8)
        self.robot_chip = QLabel("Robot  ---")
        self.robot_chip.setObjectName("chip")
        layout.addWidget(self.robot_chip)

        layout.addStretch(1)

        self.physics_chip = QLabel("Physics  ---")
        self.physics_chip.setObjectName("chip")
        self.render_chip = QLabel("Render  ---")
        self.render_chip.setObjectName("chip")
        self.health_chip = QLabel("Robot health  --")
        self.health_chip.setObjectName("chip")
        layout.addWidget(self.health_chip)
        self.rtf_chip = QLabel("Real time  ---")
        self.rtf_chip.setObjectName("chip")
        for chip in (self.physics_chip, self.render_chip, self.rtf_chip):
            layout.addWidget(chip)

        return bar

    def _start_worker(self):
        self.thread = QThread(self)
        self.worker = SimulationWorker(self.mailbox)
        self.worker.monitor_sink = self.monitoring.submit
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.ready.connect(self._on_ready)
        self.worker.telemetry_ready.connect(self._on_telemetry)
        self.worker.chart_data.connect(self.telemetry.charts.append_chunk)
        self.worker.stats_ready.connect(self._on_stats)
        self.worker.state_changed.connect(self._on_state_changed)
        self.worker.log_message.connect(self._on_log)
        self.worker.finished.connect(self.thread.quit)

        self.thread.start()
        self.controls.start()

    # -- commands -----------------------------------------------------------

    def _send_command(self, name, payload):
        if self._shutting_down:
            return
        self.worker.post(name, payload)

    # -- worker signals -----------------------------------------------------

    def _on_ready(self, description):
        self.monitoring.command("configure", description)
        self.controls.on_robot_ready(description)
        self.telemetry.build_rows(description)
        short_name = "/".join(description["urdf"].replace("\\", "/").split("/")[-2:])
        self.robot_chip.setText("Robot  %s  (%d axes)" % (short_name, description["count"]))
        self.view.set_status("Ready. Press Start.")
        self._on_log("Physics fixed at %.0f Hz, dt = %.6f s."
                     % (description["physics_hz"], 1.0 / description["physics_hz"]))

    def _on_telemetry(self, snapshot):
        self.telemetry.update_telemetry(snapshot)
        self.controls.update_telemetry(snapshot)

    def _on_stats(self, stats):
        self.telemetry.update_stats(stats)
        self.controls.set_logging_status(stats.get("logging", {}))
        self.physics_chip.setText("Physics  %.1f Hz" % stats["physics_hz"])
        self.render_chip.setText("Render  %.1f FPS" % stats["render_fps"])
        self.rtf_chip.setText("Real time  %.2fx" % stats["real_time_factor"])

    def _on_health(self, snap):
        value = snap.get("robot_health")
        self.health_chip.setText("Robot health  --" if value is None else "Robot health  %.0f%%" % value)

    def _on_state_changed(self, state):
        self._apply_state(state)

    def _on_log(self, text):
        self.telemetry.append_log(text)

    def _apply_state(self, state):
        self._state = state
        background, foreground = STATE_COLORS.get(state, STATE_COLORS[STATE_STOPPED])
        self.state_chip.setText(STATE_CAPTIONS.get(state, state))
        self.state_chip.setStyleSheet(
            "QLabel#stateChip { background-color: %s; color: %s; }"
            % (background, foreground))

        running = state == STATE_RUNNING
        estopped = state == STATE_EMERGENCY_STOP
        self.start_button.setEnabled(not running and not estopped)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(not estopped)
        self.estop_button.setEnabled(not estopped)

        self.view.set_banner("EMERGENCY STOP - press Reset to clear"
                             if estopped else "")

    # -- frame pump ---------------------------------------------------------

    def _poll_frame(self):
        self.view.update_from_mailbox(self.mailbox)

    # -- shutdown -----------------------------------------------------------

    def closeEvent(self, event):
        self._shutting_down = True
        self._poll_timer.stop()
        self.controls.shutdown()
        self.telemetry.rul.shutdown()
        self.worker.request_stop()
        self.thread.quit()
        if not self.thread.wait(3000):
            self.thread.terminate()
            self.thread.wait(500)
        self.monitoring.stop()
        self.monitoring.join(2.0)
        QMainWindow.closeEvent(self, event)
