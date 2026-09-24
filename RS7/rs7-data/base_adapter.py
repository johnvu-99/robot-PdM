"""
Common adapter interface for real condition monitoring datasets.

Adapters load files, identify channels, normalise metadata, segment signals
and extract features. They never import Qt, PyBullet or the simulator, and they
never describe their data as robot telemetry: every Recording carries
domain = "REAL_MACHINE" and the physical system it came from.
"""

import hashlib
import os
import re

import numpy as np

import config

DOMAIN_REAL_MACHINE = "REAL_MACHINE"


def natural_key(path):
    """Sort 'acc_00010.csv' after 'acc_00009.csv' and '10.csv' after '9.csv'."""
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def file_number(path):
    numbers = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return int(numbers[-1]) if numbers else 0


class Recording(object):
    """One recording: a stationary file (fault data) or a run to failure."""

    def __init__(self, dataset, recording_id, run_id, system, condition, label, files,
                 sample_rate, channels, snapshot_interval=None, meta=None):
        self.dataset = dataset
        self.recording_id = recording_id
        self.run_id = run_id                # leakage unit: never split across train/test
        self.system = system                # physical system, e.g. "rolling bearing"
        self.domain = DOMAIN_REAL_MACHINE
        self.condition = condition
        self.label = label
        self.files = list(files)
        self.sample_rate = float(sample_rate)
        self.channels = tuple(channels)
        self.snapshot_interval = snapshot_interval
        self.meta = dict(meta or {})

    def signature(self):
        """Changes when files are added, removed or modified (cache key)."""
        digest = hashlib.sha1()
        for path in self.files:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            digest.update(("%s|%d|%d" % (os.path.basename(path), stat.st_size,
                                         int(stat.st_mtime))).encode("utf-8"))
        return digest.hexdigest()

    def describe(self):
        return {
            "dataset": self.dataset, "recording_id": self.recording_id, "run_id": self.run_id,
            "system": self.system, "domain": self.domain, "condition": self.condition,
            "label": self.label, "files": len(self.files), "sample_rate": self.sample_rate,
            "channels": list(self.channels), "snapshot_interval": self.snapshot_interval,
        }


class DatasetAdapter(object):

    name = "BASE"
    role = ""
    system = ""

    def __init__(self, root):
        self.root = root

    def available(self):
        return bool(self.root) and os.path.isdir(self.root)

    def discover(self):
        raise NotImplementedError

    def summary(self, recordings):
        labels = {}
        conditions = {}
        for recording in recordings:
            labels[recording.label] = labels.get(recording.label, 0) + 1
            conditions[recording.condition] = conditions.get(recording.condition, 0) + 1
        return {
            "dataset": self.name, "role": self.role, "system": self.system,
            "root": self.root, "available": self.available(),
            "recordings": len(recordings),
            "files": sum(len(r.files) for r in recordings),
            "labels": labels, "conditions": conditions,
        }

    def cache_path(self, recording, kind):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", recording.recording_id)
        return os.path.join(config.PROCESSED_DIR, self.name, "%s_%s.npz" % (safe, kind))

    @staticmethod
    def load_cache(path, signature):
        if not os.path.isfile(path):
            return None
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != signature:
                return None
            return dict((key, data[key]) for key in data.files)

    @staticmethod
    def save_cache(path, signature, **arrays):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp.npz"
        np.savez_compressed(temporary, signature=np.array(signature), **arrays)
        os.replace(temporary, path)
