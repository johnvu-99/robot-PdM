"""
Buffered CSV telemetry writer.

The file is opened once and written from a dedicated background thread. The
simulation thread only appends batches to a queue, so disk latency can never
stall physics. Rows are formatted in the writer thread as well.
"""

import csv
import logging
import os
import queue
import threading
import time

from monitoring.telemetry import CHANNEL_INDEX

log = logging.getLogger(__name__)

CSV_COLUMNS = (
    "timestamp",
    "robot_model",
    "joint_id",
    "joint_name",
    "position",
    "target_position",
    "tracking_error",
    "velocity",
    "acceleration",
    "motor_torque",
    "reaction_force_x",
    "reaction_force_y",
    "reaction_force_z",
    "reaction_torque_x",
    "reaction_torque_y",
    "reaction_torque_z",
    "power",
    "temperature",
    "fault_type",
    "health_status",
    "fault_severity",
    "ground_truth_health",
    "scenario",
)

_NUMERIC_COLUMNS = (
    "position", "target_position", "tracking_error", "velocity", "acceleration",
    "motor_torque", "reaction_force_x", "reaction_force_y", "reaction_force_z",
    "reaction_torque_x", "reaction_torque_y", "reaction_torque_z", "power",
    "temperature",
)
_NUMERIC_INDICES = tuple(CHANNEL_INDEX[name] for name in _NUMERIC_COLUMNS)

_STOP = object()


class CsvTelemetryWriter(object):
    """
    One CSV file. Call start(), submit() any number of times, then finish().
    finish() never blocks; poll is_alive() or call join() during shutdown.
    """

    def __init__(self, path, meta, flush_interval=1.0, buffer_bytes=1 << 20):
        self.path = path
        self.meta = dict(meta)
        self.flush_interval = float(flush_interval)
        self.buffer_bytes = int(buffer_bytes)

        self.rows_written = 0
        self.error = None
        self._queue = queue.Queue()
        self._thread = None
        self._finished = False

    # -- producer side (simulation thread) ----------------------------------

    def start(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Opening here surfaces permission errors to the caller immediately.
        self._handle = open(self.path, "w", newline="", encoding="utf-8",
                            buffering=self.buffer_bytes)
        self._writer = csv.writer(self._handle)
        self._writer.writerow(CSV_COLUMNS)
        self._thread = threading.Thread(target=self._run, name="CsvTelemetryWriter")
        self._thread.daemon = True
        self._thread.start()

    def submit(self, records):
        """records: list of (timestamp, matrix (n, C)). Never blocks."""
        if records and not self._finished:
            self._queue.put(records)

    def finish(self):
        if not self._finished:
            self._finished = True
            self._queue.put(_STOP)

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def bytes_written(self):
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    # -- writer thread ------------------------------------------------------

    def _run(self):
        last_flush = time.perf_counter()
        try:
            while True:
                try:
                    item = self._queue.get(timeout=self.flush_interval)
                except queue.Empty:
                    self._handle.flush()
                    last_flush = time.perf_counter()
                    continue
                if item is _STOP:
                    break
                self._write_records(item)
                if time.perf_counter() - last_flush >= self.flush_interval:
                    self._handle.flush()
                    last_flush = time.perf_counter()
        except (IOError, OSError, ValueError) as exc:
            self.error = str(exc)
            log.error("CSV writer failed for %s: %s", self.path, exc)
        finally:
            try:
                self._handle.close()
            except (IOError, OSError) as exc:
                self.error = self.error or str(exc)

    def _write_records(self, records):
        model = self.meta.get("robot_model", "")
        joint_ids = self.meta.get("joint_ids", [])
        joint_names = self.meta.get("joint_names", [])
        fault_type = self.meta.get("fault_type", "NORMAL")
        health = self.meta.get("health_status", "UNKNOWN")
        indices = _NUMERIC_INDICES
        rows = []
        for timestamp, matrix, labels in records:
            stamp = "%.6f" % timestamp
            for j in range(matrix.shape[0]):
                values = matrix[j]
                row = [stamp, model,
                       joint_ids[j] if j < len(joint_ids) else j,
                       joint_names[j] if j < len(joint_names) else ""]
                row.extend("%.6g" % values[i] for i in indices)
                if labels is not None and j < len(labels):
                    label = labels[j]
                    row.extend((label[0], label[3], "%.3f" % label[1], "%.1f" % label[2], label[4]))
                else:
                    row.extend((fault_type, health, "0.000", "", ""))
                rows.append(row)
        self._writer.writerows(rows)
        self.rows_written += len(rows)
