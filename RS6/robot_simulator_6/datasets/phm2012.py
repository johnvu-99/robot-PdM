"""
IEEE PHM 2012 / PRONOSTIA (github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset).

Role: INDEPENDENT RUL BENCHMARK AND VALIDATION.

<root>/Learning_set/Bearing1_1/acc_00001.csv ...   6 complete runs
<root>/Full_Test_Set/Bearing1_3/acc_00001.csv ...  11 complete runs
Each acc file: 2560 rows of hour, minute, second, microsecond, horizontal,
vertical acceleration (comma or semicolon separated), every 10 s.
temp_*.csv files are ignored. Test_set is truncated and not used.
"""

import glob
import os
import re

import config
from datasets.base_adapter import Recording, natural_key
from datasets.run_to_failure import RunToFailureAdapter
from datasets.signal_features import parse_numeric_text

PHM_CONDITIONS = {
    "1": "1800rpm_4000N",
    "2": "1650rpm_4200N",
    "3": "1500rpm_5000N",
}


class PHM2012DatasetAdapter(RunToFailureAdapter):

    name = "PHM2012"
    role = "Independent RUL benchmark and validation"
    system = "Rolling element bearings, accelerated run to failure (PRONOSTIA)"

    def __init__(self, root=None):
        RunToFailureAdapter.__init__(self, root or config.PHM2012_DIR)

    def discover(self):
        recordings = []
        for subset in config.PHM_COMPLETE_SETS:
            for folder in sorted(glob.glob(os.path.join(self.root, subset, "Bearing*_*"))):
                name = os.path.basename(folder)
                match = re.match(r"Bearing(\d+)_(\d+)$", name)
                if not match:
                    continue
                files = sorted(glob.glob(os.path.join(folder, "acc_*.csv")), key=natural_key)
                if len(files) < 2:
                    continue
                recordings.append(Recording(
                    self.name, "PHM_%s" % name, "PHM_%s" % name, self.system,
                    PHM_CONDITIONS.get(match.group(1), "condition%s" % match.group(1)),
                    "RUN_TO_FAILURE", files, config.PHM_SAMPLE_RATE_HZ, self.snapshot_channels,
                    snapshot_interval=config.PHM_SNAPSHOT_INTERVAL_S,
                    meta={"subset": subset}))
        return recordings

    def read_snapshot(self, path):
        with open(path, "r", errors="replace") as handle:
            data = parse_numeric_text(handle.read(), 6)
        if data.shape[0] < 16:
            raise ValueError("snapshot too short: %s" % path)
        return data[:, 4:6]
