"""
Description of a partner collected dataset.

A dataset folder contains:

    dataset.json     what was measured and how the files are laid out
    recordings.csv   one row per data file: run, asset, condition, label, timing
    <data files>     CSV, Excel (.xlsx) or MATLAB (.mat)

This module only reads and checks those two description files. It never loads
signal data and has no Qt, PyBullet or ML imports.
"""

import csv
import json
import os

import config

CHANNEL_TYPES = (
    "position", "target_position", "velocity", "acceleration", "torque", "current",
    "voltage", "power", "vibration", "temperature", "ambient_temperature", "force", "other",
)
LABEL_MODES = ("fault_labels", "run_to_failure", "unlabeled")
FORMATS = ("csv", "xlsx", "mat")
RECORDING_COLUMNS = ("file", "run_id", "asset_id", "condition", "label", "fault_joint",
                     "start_time_s", "failure_time_s", "notes")
REQUIRED_RECORDING_COLUMNS = ("file", "run_id")


class SchemaError(ValueError):
    pass


class Channel(object):

    def __init__(self, data):
        self.name = str(data["name"])
        self.column = data.get("column", self.name)
        self.type = data.get("type", "other")
        self.unit = data.get("unit", "")
        self.scale = float(data.get("scale", 1.0))
        if self.type not in CHANNEL_TYPES:
            raise SchemaError("channel %s: unknown type %s (allowed: %s)"
                              % (self.name, self.type, ", ".join(CHANNEL_TYPES)))

    def to_dict(self):
        return {"name": self.name, "column": self.column, "type": self.type,
                "unit": self.unit, "scale": self.scale}


class Recording(object):
    """One row of recordings.csv."""

    def __init__(self, row, root):
        self.file = row["file"].strip()
        self.path = os.path.normpath(os.path.join(root, self.file))
        self.run_id = row["run_id"].strip()
        self.asset_id = (row.get("asset_id") or "").strip() or "asset"
        self.condition = (row.get("condition") or "").strip() or "default"
        self.label = (row.get("label") or "").strip()
        # Which joint the label describes. Empty: the label applies to every
        # joint in the file. Other joints in the file are then healthy.
        self.fault_joint = (row.get("fault_joint") or "").strip()
        self.start_time_s = _optional_float(row.get("start_time_s"))
        self.failure_time_s = _optional_float(row.get("failure_time_s"))
        self.notes = (row.get("notes") or "").strip()

    def to_dict(self):
        return {"file": self.file, "run_id": self.run_id, "asset_id": self.asset_id,
                "condition": self.condition, "label": self.label, "fault_joint": self.fault_joint,
                "start_time_s": self.start_time_s, "failure_time_s": self.failure_time_s}


def _optional_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


class DatasetSchema(object):

    def __init__(self, root, data, recordings):
        self.root = root
        self.name = data.get("name", os.path.basename(os.path.abspath(root)))
        self.system = data.get("system", "")
        self.description = data.get("description", "")
        self.format = data.get("format", "csv").lower()
        self.sample_rate_hz = data.get("sample_rate_hz")
        self.time_column = data.get("time_column")
        self.joint_column = data.get("joint_column")
        self.csv_delimiter = data.get("csv_delimiter", ",")
        self.sheet = data.get("sheet")
        self.mat_variable = data.get("mat_variable")
        self.window_s = float(data.get("window_seconds", config.PARTNER_DEFAULT_WINDOW_S))
        self.temperature_window_s = float(data.get("temperature_window_seconds",
                                                   config.PARTNER_TEMPERATURE_WINDOW_S))
        # Seconds of warm-up dropped from the start of every recording.
        self.warmup_skip_s = float(data.get("warmup_skip_seconds", config.PARTNER_DEFAULT_WARMUP_SKIP_S))
        self.hop_s = float(data.get("hop_seconds", config.PARTNER_DEFAULT_HOP_S))
        self.label_mode = data.get("label_mode", "unlabeled")
        self.healthy_label = data.get("healthy_label", "HEALTHY")
        self.channels = [Channel(c) for c in data.get("channels", [])]
        self.recordings = recordings
        self.raw = data

    @property
    def channel_types(self):
        return sorted(set(c.type for c in self.channels))

    def to_dict(self):
        return {
            "name": self.name, "system": self.system, "format": self.format,
            "sample_rate_hz": self.sample_rate_hz, "time_column": self.time_column,
            "joint_column": self.joint_column, "window_seconds": self.window_s,
            "hop_seconds": self.hop_s, "temperature_window_seconds": self.temperature_window_s,
            "warmup_skip_seconds": self.warmup_skip_s, "label_mode": self.label_mode,
            "healthy_label": self.healthy_label,
            "channels": [c.to_dict() for c in self.channels],
            "recordings": len(self.recordings),
        }


def load_schema(root):
    """Read dataset.json and recordings.csv. Raises SchemaError with a readable message."""
    schema_path = os.path.join(root, config.PARTNER_SCHEMA_FILE)
    recordings_path = os.path.join(root, config.PARTNER_RECORDINGS_FILE)
    if not os.path.isfile(schema_path):
        raise SchemaError("missing %s in %s" % (config.PARTNER_SCHEMA_FILE, root))
    if not os.path.isfile(recordings_path):
        raise SchemaError("missing %s in %s" % (config.PARTNER_RECORDINGS_FILE, root))
    with open(schema_path, encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except ValueError as exc:
            raise SchemaError("%s is not valid JSON: %s" % (config.PARTNER_SCHEMA_FILE, exc))

    problems = []
    if data.get("format", "csv").lower() not in FORMATS:
        problems.append("format must be one of %s" % ", ".join(FORMATS))
    if data.get("label_mode", "unlabeled") not in LABEL_MODES:
        problems.append("label_mode must be one of %s" % ", ".join(LABEL_MODES))
    if not data.get("channels"):
        problems.append("channels list is empty")
    if not data.get("sample_rate_hz") and not data.get("time_column"):
        problems.append("set sample_rate_hz, time_column, or both")
    names = [c.get("name") for c in data.get("channels", [])]
    if len(names) != len(set(names)):
        problems.append("channel names must be unique")
    if problems:
        raise SchemaError("; ".join(problems))

    with open(recordings_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_RECORDING_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SchemaError("%s is missing columns: %s" % (config.PARTNER_RECORDINGS_FILE, ", ".join(missing)))
        recordings = []
        for line, row in enumerate(reader, start=2):
            if not (row.get("file") or "").strip():
                continue
            try:
                recordings.append(Recording(row, root))
            except ValueError as exc:
                raise SchemaError("%s line %d: %s" % (config.PARTNER_RECORDINGS_FILE, line, exc))
    if not recordings:
        raise SchemaError("%s lists no recordings" % config.PARTNER_RECORDINGS_FILE)

    try:
        schema = DatasetSchema(root, data, recordings)
    except (KeyError, TypeError) as exc:
        raise SchemaError("invalid channel definition: %s" % exc)

    if schema.label_mode == "fault_labels":
        empty = [r.file for r in recordings if not r.label]
        if empty:
            raise SchemaError("label_mode is fault_labels but these recordings have no label: %s"
                              % ", ".join(empty[:5]))
    return schema
