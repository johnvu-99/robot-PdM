"""
Labelled simulation dataset generator.

Runs in its own process (multiprocessing, spawn safe), so generating data never
competes with the interactive simulation for the GIL. Each episode creates a
fresh PyBullet DIRECT client and steps the exact StepPipeline used by the live
simulator, as fast as the CPU allows.

Output:
    data/simulation/normal/<run>.csv
    data/simulation/faulty/<run>.csv
    data/simulation/manifest_<timestamp>.json

Every CSV row carries its ground truth label. Faulty runs start healthy and
the fault begins at fault_onset_s, so each file contains NORMAL samples
followed by labelled faulty samples.

No Qt imports: this module must be importable in a child process.
"""

import datetime
import json
import logging
import os
import platform
import time

import numpy as np
import pybullet as p

import config
from monitoring.telemetry import TelemetryRecorder
from simulator.fault_injector import FAULT_TYPES, FaultInjector
from simulator.joint_controller import JointController
from simulator.robot import Robot
from simulator.step_pipeline import StepPipeline, configure_client
from simulator.trajectory import TrajectoryGenerator
from storage.csv_logger import CsvTelemetryWriter

log = logging.getLogger(__name__)


class GenerationCancelled(Exception):
    pass


def build_plan(preset_name):
    """List of episode dictionaries for a preset in config.DATASET_PRESETS."""
    preset = config.DATASET_PRESETS[preset_name]
    runs = []
    seed = config.DATASET_SEED

    for pattern in preset["normal_patterns"]:
        for speed in preset["normal_speeds"]:
            runs.append({
                "name": "normal_%s_speed%.1f" % (pattern.lower(), speed),
                "folder": "normal", "pattern": pattern, "speed": speed,
                "seconds": preset["run_seconds"], "fault": None, "seed": seed,
            })
            seed += 1

    pattern_index = 0
    for fault_type in FAULT_TYPES:
        severities = (config.DATASET_CAPACITY_FAULT_SEVERITIES
                      if fault_type in config.DATASET_CAPACITY_FAULTS
                      else preset["fault_severities"])
        if preset_name == "QUICK":
            severities = severities[-1:]
        for joint in preset["fault_joints"]:
            for severity in severities:
                pattern = config.DATASET_FAULT_PATTERNS[pattern_index % len(config.DATASET_FAULT_PATTERNS)]
                pattern_index += 1
                onset = preset["fault_onset_s"]
                runs.append({
                    "name": "%s_j%d_sev%.2f_%s" % (fault_type.lower(), joint + 1, severity,
                                                  pattern.lower()),
                    "folder": "faulty", "pattern": pattern, "speed": 1.0,
                    "seconds": preset["run_seconds"], "seed": seed,
                    "fault": {"type": fault_type, "joint": joint, "severity": severity,
                              "start": onset,
                              # Degradation ramps over the rest of the run.
                              "duration": (preset["run_seconds"] - onset
                                           if fault_type == "TORQUE_DEGRADATION" else 0.0)},
                })
                seed += 1

    for name in preset["degradation_scenarios"]:
        joint = preset["fault_joints"][0]
        runs.append({
            "name": "degradation_%s_j%d" % (name.lower(), joint + 1),
            "folder": "faulty", "pattern": "POINT_TO_POINT", "speed": 1.0,
            "seconds": preset["degradation_seconds"] + preset["fault_onset_s"], "seed": seed,
            "degradation": {"scenario": name, "joint": joint, "start": preset["fault_onset_s"],
                            "duration": preset["degradation_seconds"],
                            "final_health": config.DEGRADATION_DEFAULT_FINAL_HEALTH},
            "fault": None,
        })
        seed += 1
    return runs


def run_episode(run, out_root, control_mode=None, cancel_event=None):
    """Simulate one episode and write its CSV. Returns a manifest entry."""
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("could not create a PyBullet client")
    writer = None
    try:
        configure_client(client)
        robot = Robot(client, config.URDF_PATH, config.ROBOT_BASE_POSITION,
                      config.USE_FIXED_BASE)
        robot.load()
        controller = JointController(client, robot)
        controller.set_control_mode(control_mode or config.DEFAULT_CONTROL_MODE)
        controller.release_motors()
        trajectory = TrajectoryGenerator(robot.n, robot.home_pose, robot.lower_limits,
                                         robot.upper_limits, controller.velocity_limits)
        trajectory.set_speed(run["speed"])
        trajectory.resync(robot.home_pose)
        trajectory.set_mode(run["pattern"], transition_time=1.0)

        injector = FaultInjector(client, robot, controller, config.FIXED_DT, seed=run["seed"])
        if run.get("fault"):
            fault = run["fault"]
            injector.add_fault(fault["joint"], fault["type"], fault["severity"],
                               fault["start"], fault["duration"])
        if run.get("degradation"):
            d = run["degradation"]
            injector.add_scenario(d["scenario"], d["joint"], d["start"], d["duration"],
                                  d["final_health"])

        recorder = TelemetryRecorder(config.FIXED_DT, 1.0, 0.0)
        recorder.set_csv_rate(config.DATASET_CSV_RATE_HZ)
        pipeline = StepPipeline(client, robot, controller, trajectory, injector, recorder)

        folder = os.path.join(out_root, run["folder"])
        path = os.path.join(folder, run["name"] + ".csv")
        writer = CsvTelemetryWriter(path, {
            "robot_model": config.URDF_PATH,
            "joint_ids": list(robot.joint_indices),
            "joint_names": list(robot.joint_names),
        }, flush_interval=5.0)
        writer.start()

        steps = int(round(run["seconds"] / config.FIXED_DT))
        sim_time = 0.0
        label_counts = {}
        for step in range(steps):
            sim_time = pipeline.step(sim_time, braking=False)
            if step % 240 == 239:
                records = recorder.take_csv_records()
                for _, _, labels in records:
                    for label in labels:
                        label_counts[label[0]] = label_counts.get(label[0], 0) + 1
                writer.submit(records)
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled()
        records = recorder.take_csv_records()
        for _, _, labels in records:
            for label in labels:
                label_counts[label[0]] = label_counts.get(label[0], 0) + 1
        writer.submit(records)
        writer.finish()
        writer.join()
        if writer.error:
            raise IOError(writer.error)

        entry = dict(run)
        entry.update({
            "file": os.path.relpath(path, out_root).replace("\\", "/"),
            "rows": writer.rows_written,
            "csv_rate_hz": config.DATASET_CSV_RATE_HZ,
            "control_mode": controller.control_mode,
            "robot_model": config.URDF_PATH,
            "label_counts": label_counts,
        })
        return entry
    finally:
        if writer is not None and writer.is_alive():
            writer.finish()
            writer.join(5.0)
        p.disconnect(physicsClientId=client)


def _lower_priority():
    """Keep the interactive simulator responsive while generating."""
    try:
        if platform.system() == "Windows":
            import ctypes
            below_normal = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, below_normal)
        else:
            os.nice(10)
    except (AttributeError, OSError) as exc:
        log.info("Could not lower generator priority: %s", exc)


def generate(preset_name, out_root=None, progress_queue=None, cancel_event=None,
             control_mode=None):
    out_root = out_root or config.SIMULATION_DATA_DIR
    runs = build_plan(preset_name)
    manifest = {
        "created": datetime.datetime.now().isoformat(),
        "preset": preset_name,
        "physics_hz": config.PHYSICS_HZ,
        "csv_rate_hz": config.DATASET_CSV_RATE_HZ,
        "label_set": ["NORMAL"] + list(FAULT_TYPES),
        "note": ("Simulated KUKA iiwa telemetry from PyBullet. Labels are ground "
                 "truth from the fault injector, not estimates."),
        "runs": [],
    }

    def report(kind, **fields):
        if progress_queue is not None:
            message = {"kind": kind}
            message.update(fields)
            progress_queue.put(message)

    started = time.perf_counter()
    for index, run in enumerate(runs):
        report("progress", index=index, total=len(runs), name=run["name"])
        try:
            entry = run_episode(run, out_root, control_mode, cancel_event)
        except GenerationCancelled:
            report("cancelled", done=index, total=len(runs))
            return None
        manifest["runs"].append(entry)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(out_root, "manifest_%s.json" % stamp)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    report("finished", total=len(runs), manifest=manifest_path,
           seconds=time.perf_counter() - started)
    return manifest_path


def process_main(preset_name, out_root, progress_queue, cancel_event, control_mode):
    """Entry point for multiprocessing. Errors are reported, never raised silently."""
    _lower_priority()
    try:
        generate(preset_name, out_root, progress_queue, cancel_event, control_mode)
    except Exception as exc:  # the child must always report back to the UI
        progress_queue.put({"kind": "error", "message": "%s: %s" % (type(exc).__name__, exc)})
