"""
Saving and loading monitoring models with joblib.

A bundle holds the baseline, the Isolation Forest models, and the metadata
needed to refuse a model that does not match the loaded robot.
"""

import datetime
import os

import joblib

import config

BUNDLE_VERSION = 1


def default_path(robot_model):
    stem = os.path.splitext(os.path.basename(robot_model or "robot"))[0]
    folder = os.path.basename(os.path.dirname(robot_model or "")) or "robot"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(config.MODEL_DIR, "anomaly_%s_%s_%s.joblib" % (folder, stem, stamp))


def save_bundle(path, robot_model, joint_names, baseline, forest, info):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    bundle = {
        "version": BUNDLE_VERSION,
        "created": datetime.datetime.now().isoformat(),
        "robot_model": robot_model,
        "joint_names": list(joint_names),
        "baseline": baseline.to_dict() if baseline is not None else None,
        "forest": None,
        "info": dict(info),
    }
    if forest is not None and forest.trained:
        bundle["forest"] = {
            "feature_names": forest.feature_names,
            "joint_names": forest.joint_names,
            "models": forest.models,
            "training_windows": forest.training_windows,
        }
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("version") != BUNDLE_VERSION:
        raise ValueError("not a version %d anomaly model bundle" % BUNDLE_VERSION)
    return bundle
