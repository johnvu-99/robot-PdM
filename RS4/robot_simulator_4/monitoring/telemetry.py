"""
Joint telemetry recording.

Runs inside the simulation thread and is called once per physics step. Each
step becomes one (timestamp, matrix) entry in a collections.deque ring buffer,
where the matrix has one row per movable joint and one column per channel.
Nothing here touches Qt or the filesystem: consumers pull decimated chunks for
charts and CSV at their own rates.
"""

import collections

import numpy as np

CHANNELS = (
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
    "reaction_force_magnitude",
)
CHANNEL_INDEX = dict((name, i) for i, name in enumerate(CHANNELS))
NUM_CHANNELS = len(CHANNELS)

CH_POSITION = CHANNEL_INDEX["position"]
CH_TARGET = CHANNEL_INDEX["target_position"]
CH_ERROR = CHANNEL_INDEX["tracking_error"]
CH_VELOCITY = CHANNEL_INDEX["velocity"]
CH_ACCELERATION = CHANNEL_INDEX["acceleration"]
CH_TORQUE = CHANNEL_INDEX["motor_torque"]
CH_REACTION_START = CHANNEL_INDEX["reaction_force_x"]
CH_POWER = CHANNEL_INDEX["power"]
CH_FORCE_MAGNITUDE = CHANNEL_INDEX["reaction_force_magnitude"]


class _Decimator(object):
    """Selects samples at an average rate using the simulation clock."""

    def __init__(self, rate_hz, dt):
        self.dt = dt
        self.period = 0.0
        self.accumulator = 0.0
        self.set_rate(rate_hz)

    def set_rate(self, rate_hz):
        rate_hz = float(rate_hz) if rate_hz else 0.0
        self.period = 1.0 / rate_hz if rate_hz > 0.0 else 0.0
        # Start "due", so the first sample after a start is kept.
        self.accumulator = self.period

    def due(self):
        if self.period <= 0.0:
            return False
        self.accumulator += self.dt
        if self.accumulator + 1e-9 >= self.period:
            self.accumulator -= self.period
            if self.accumulator > self.period:
                self.accumulator = 0.0
            return True
        return False


class TelemetryRecorder(object):

    def __init__(self, dt, buffer_seconds, chart_hz):
        self.dt = float(dt)
        self._buffer = collections.deque(maxlen=max(1, int(round(buffer_seconds / dt))))
        self._chart = _Decimator(chart_hz, self.dt)
        self._csv = _Decimator(0.0, self.dt)
        self._chart_pending = []
        self._csv_pending = []
        self._previous_velocity = None
        self._latest = None
        self.samples_recorded = 0

    # -- configuration ------------------------------------------------------

    def reset(self):
        """Forget history. Called on stop, reset and robot swap."""
        self._buffer.clear()
        self._chart_pending = []
        self._csv_pending = []
        self._previous_velocity = None
        self._latest = None

    def set_csv_rate(self, rate_hz):
        """0 or None disables CSV sample collection."""
        self._csv.set_rate(rate_hz)
        self._csv_pending = []

    @property
    def csv_enabled(self):
        return self._csv.period > 0.0

    # -- recording ----------------------------------------------------------

    def record(self, timestamp, positions, targets, velocities, torques, reactions,
               labels=None):
        """
        Append one physics step. reactions has shape (n, 6): Fx Fy Fz Mx My Mz
        from the joint force/torque sensors. labels is the injector's immutable
        per joint ground truth tuple, stored by reference.
        """
        n = positions.shape[0]
        if self._previous_velocity is None or self._previous_velocity.shape[0] != n:
            acceleration = np.zeros(n)
        else:
            acceleration = (velocities - self._previous_velocity) / self.dt
        self._previous_velocity = np.array(velocities, dtype=np.float64)

        row = np.empty((n, NUM_CHANNELS), dtype=np.float64)
        row[:, CH_POSITION] = positions
        row[:, CH_TARGET] = targets
        row[:, CH_ERROR] = targets - positions
        row[:, CH_VELOCITY] = velocities
        row[:, CH_ACCELERATION] = acceleration
        row[:, CH_TORQUE] = torques
        row[:, CH_REACTION_START:CH_REACTION_START + 6] = reactions
        row[:, CH_POWER] = torques * velocities
        row[:, CH_FORCE_MAGNITUDE] = np.sqrt(np.sum(reactions[:, 0:3] ** 2, axis=1))

        entry = (float(timestamp), row, labels)
        self._buffer.append(entry)
        self._latest = entry
        self.samples_recorded += 1
        if self._chart.due():
            self._chart_pending.append(entry)
        if self._csv.due():
            self._csv_pending.append(entry)
        return row

    # -- consumers ----------------------------------------------------------

    @property
    def latest(self):
        return self._latest

    def __len__(self):
        return len(self._buffer)

    def take_chart_chunk(self):
        """Returns (times (k,), data (k, n, C)) or None. Rows are shared, never mutated."""
        pending = self._chart_pending
        if not pending:
            return None
        self._chart_pending = []
        times = np.array([entry[0] for entry in pending], dtype=np.float64)
        data = np.stack([entry[1] for entry in pending])
        return times, data

    def take_csv_records(self):
        """List of (timestamp, matrix (n, C), labels or None)."""
        pending = self._csv_pending
        self._csv_pending = []
        return pending

    def snapshot(self):
        """Shallow copy of the full ring buffer, for export."""
        return list(self._buffer)
