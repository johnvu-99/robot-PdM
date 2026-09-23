"""Log tab: maintenance event log with CSV export."""

import os
import time

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import config
from predictive.event_log import EVENT_COLUMNS, export_csv

COLORS = {"CRITICAL": "#e0474c", "WARNING": "#d9a326", "INFO": "#b9c0ca"}


class EventLogPanel(QWidget):

    def __init__(self, service, parent=None):
        QWidget.__init__(self, parent)
        self.service = service
        self._last_id = 0
        self._events = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.follow = QCheckBox("Follow newest")
        self.follow.setChecked(True)
        self.only_important = QCheckBox("Warnings and critical only")
        self.only_important.toggled.connect(self._rebuild)
        export = QPushButton("Export CSV")
        export.clicked.connect(self._export)
        clear = QPushButton("Clear view")
        clear.clicked.connect(self._clear)
        self.count = QLabel("")
        self.count.setObjectName("hint")
        for widget in (self.follow, self.only_important, export, clear):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.count)
        layout.addLayout(row)

        self.table = QTableWidget(0, len(EVENT_COLUMNS))
        self.table.setHorizontalHeaderLabels([c.replace("_", " ").title() for c in EVENT_COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(EVENT_COLUMNS) - 1, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

    def poll(self):
        new = self.service.pop_events(self._last_id)
        if not new:
            return
        self._last_id = new[-1]["id"]
        self._events.extend(new)
        if len(self._events) > config.EVENT_LOG_MAX:
            self._events = self._events[-config.EVENT_LOG_MAX:]
            self._rebuild()
            return
        for event in new:
            self._append(event)
        self._finish()

    def _visible(self, event):
        return not self.only_important.isChecked() or event["severity"] in ("WARNING", "CRITICAL")

    def _append(self, event):
        if not self._visible(event):
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        color = QColor(COLORS.get(event["severity"], COLORS["INFO"]))
        for column, key in enumerate(EVENT_COLUMNS):
            item = QTableWidgetItem(str(event[key]))
            item.setForeground(color)
            self.table.setItem(row, column, item)

    def _finish(self):
        self.count.setText("%d events" % len(self._events))
        if self.follow.isChecked():
            self.table.scrollToBottom()

    def _rebuild(self, _checked=None):
        self.table.setRowCount(0)
        for event in self._events:
            self._append(event)
        self._finish()

    def _clear(self):
        self._events = []
        self.table.setRowCount(0)
        self.count.setText("")

    def _export(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        default = os.path.join(config.LOG_DIR, "events_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
        path, _ = QFileDialog.getSaveFileName(self, "Export event log", default, "CSV files (*.csv)")
        if path:
            export_csv(self._events, path)
