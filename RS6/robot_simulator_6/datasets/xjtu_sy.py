"""
XJTU-SY bearing datasets (github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets).

Role: DEGRADATION LEARNING AND RUL TRAINING.

The repository only links to the data. After download the layout is
    <root>/35Hz12kN/Bearing1_1/1.csv, 2.csv, ...
    <root>/37.5Hz11kN/Bearing2_1/...
    <root>/40Hz10kN/Bearing3_1/...
Each CSV is one minute's 1.28 s snapshot at 25.6 kHz with two columns,
horizontal and vertical vibration, usually with a header row. Folders are
found recursively, so an extra wrapping directory is fine.

These are accelerated life tests of rolling element bearings. Bearing failure
physics are not assumed to match KUKA joint failure; the data is used to
develop and demonstrate the degradation and RUL method.
"""

import os
import re

import config
from datasets.base_adapter import Recording, natural_key
from datasets.run_to_failure import RunToFailureAdapter
from datasets.signal_features import parse_numeric_text


class XJTUSYDatasetAdapter(RunToFailureAdapter):

    name = "XJTU_SY"
    role = "Degradation learning and RUL training"
    system = "Rolling element bearings, accelerated run to failure (XJTU-SY)"

    def __init__(self, root=None):
        RunToFailureAdapter.__init__(self, root or config.XJTU_SY_DIR)

    def discover(self):
        recordings = []
        if not self.available():
            return recordings
        for current, directories, files in os.walk(self.root):
            directories.sort()
            name = os.path.basename(current)
            if not re.match(r"Bearing\d+_\d+$", name):
                continue
            csvs = [os.path.join(current, f) for f in files
                    if f.lower().endswith(".csv") and re.match(r"\d+\.csv$", f)]
            if len(csvs) < 2:
                continue
            csvs.sort(key=natural_key)
            condition = os.path.basename(os.path.dirname(current))
            recordings.append(Recording(
                self.name, "XJTU_%s" % name, "XJTU_%s" % name, self.system, condition,
                "RUN_TO_FAILURE", csvs, config.XJTU_SAMPLE_RATE_HZ, self.snapshot_channels,
                snapshot_interval=config.XJTU_SNAPSHOT_INTERVAL_S))
        return recordings

    def read_snapshot(self, path):
        with open(path, "r", errors="replace") as handle:
            text = handle.read()
        first_line, _, rest = text.partition("\n")
        try:
            [float(v) for v in re.split(r"[,;\t]", first_line.strip()) if v]
        except ValueError:
            text = rest          # header row such as Horizontal_vibration_signals
        data = parse_numeric_text(text, 2)
        if data.shape[0] < 16:
            raise ValueError("snapshot too short: %s" % path)
        return data
