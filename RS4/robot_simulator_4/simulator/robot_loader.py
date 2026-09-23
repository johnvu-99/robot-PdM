"""
Robot model discovery.

Walks the configured model folders for URDF and SDF files and returns a sorted
catalogue. Pure filesystem work: no PyBullet client and no Qt, so it can run on
any thread without touching the simulation.
"""

import logging
import os
import re

import pybullet_data

import config

log = logging.getLogger(__name__)

PRIORITY_REQUIRED = 0
PRIORITY_KUKA = 1
PRIORITY_OTHER = 2

# Movable joint declarations in URDF (type="revolute") and SDF (type='revolute').
_MOVABLE_JOINT_PATTERN = re.compile(
    r"<joint[^>]*type\s*=\s*[\"'](revolute|prismatic|continuous)[\"']",
    re.IGNORECASE)


class RobotModelInfo(object):
    """One entry of the model catalogue. Plain data, safe to pass to Qt."""

    __slots__ = ("relative_path", "path", "root_label", "file_type", "exists",
                 "required", "priority", "movable_hint")

    def __init__(self, relative_path, path, root_label, exists=True,
                 required=False, movable_hint=0):
        self.relative_path = relative_path
        self.path = path
        self.root_label = root_label
        self.file_type = os.path.splitext(relative_path)[1].lower().lstrip(".")
        self.exists = bool(exists)
        self.required = bool(required)
        self.movable_hint = int(movable_hint)
        self.priority = PRIORITY_OTHER
        if required:
            self.priority = PRIORITY_REQUIRED
        elif _is_priority_name(relative_path):
            self.priority = PRIORITY_KUKA

    @property
    def articulated(self):
        return self.movable_hint > 0

    def label(self):
        suffix = ""
        if not self.exists:
            suffix = "  (not found)"
        elif self.root_label != "pybullet_data":
            suffix = "  [%s]" % self.root_label
        return self.relative_path + suffix

    def __repr__(self):
        return "RobotModelInfo(%r, exists=%r)" % (self.relative_path, self.exists)


def _is_priority_name(relative_path):
    lowered = relative_path.lower()
    for keyword in config.PRIORITY_KEYWORDS:
        if keyword in lowered:
            return True
    return False


def _normalise(relative_path):
    return relative_path.replace("\\", "/").lstrip("./")


def default_search_roots():
    """Ordered (label, folder) pairs. Earlier roots win on duplicate names."""
    roots = []
    try:
        roots.append(("pybullet_data", pybullet_data.getDataPath()))
    except (AttributeError, OSError) as exc:
        log.warning("pybullet_data path unavailable: %s", exc)

    for folder in config.EXTRA_MODEL_SEARCH_PATHS:
        roots.append(("bullet3", folder))

    extra = os.environ.get(config.MODEL_SEARCH_ENV_VAR, "")
    for folder in extra.split(os.pathsep):
        folder = folder.strip()
        if folder:
            roots.append(("env", folder))
    return roots


def count_movable_joints(path):
    """Cheap text sniff. Real joint data comes from PyBullet at load time."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(config.MODEL_SNIFF_BYTES)
    except (IOError, OSError) as exc:
        log.debug("Could not read %s: %s", path, exc)
        return 0
    text = raw.decode("utf-8", "replace")
    return len(_MOVABLE_JOINT_PATTERN.findall(text))


def _walk_models(root):
    """Yield (relative_path, absolute_path) under root, depth limited."""
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for current, directories, files in os.walk(root):
        depth = current.rstrip(os.sep).count(os.sep) - base_depth
        directories[:] = sorted(d for d in directories if not d.startswith("."))
        if depth >= config.MODEL_SCAN_MAX_DEPTH:
            directories[:] = []
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in config.MODEL_EXTENSIONS:
                continue
            absolute = os.path.join(current, name)
            yield _normalise(os.path.relpath(absolute, root)), absolute


def scan_models(roots=None):
    """
    Return a sorted list of RobotModelInfo.

    Required models come first (including placeholders for missing ones), then
    anything with a KUKA keyword, then everything else alphabetically.
    """
    if roots is None:
        roots = default_search_roots()

    found = {}
    scanned = 0
    for label, folder in roots:
        if not folder or not os.path.isdir(folder):
            log.info("Model folder skipped (not found): %s", folder)
            continue
        for relative, absolute in _walk_models(folder):
            if scanned >= config.MODEL_SCAN_MAX_FILES:
                log.warning("Model scan stopped at %d files.", scanned)
                break
            scanned += 1
            if relative in found:
                continue
            found[relative] = RobotModelInfo(
                relative, absolute, label,
                movable_hint=count_movable_joints(absolute))

    required = set(_normalise(r) for r in config.REQUIRED_MODELS)
    for relative in required:
        if relative in found:
            info = found[relative]
            found[relative] = RobotModelInfo(
                info.relative_path, info.path, info.root_label,
                required=True, movable_hint=info.movable_hint)
        else:
            found[relative] = RobotModelInfo(relative, "", "", exists=False,
                                             required=True)

    models = list(found.values())
    models.sort(key=lambda m: (m.priority, m.relative_path.lower()))
    log.info("Model scan: %d files, %d catalogue entries.", scanned, len(models))
    return models
