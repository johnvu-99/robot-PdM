"""
One physics step, shared by the interactive worker and the dataset generator.

Both call StepPipeline.step(), so a generated dataset runs exactly the same
control, fault and measurement code as the live simulator. No Qt, no timing:
the caller decides when to step (wall clock accumulator or as fast as possible).
"""

import numpy as np
import pybullet as p
import pybullet_data

import config
from simulator.thermal_model import ThermalModel


def configure_client(client_id):
    """World setup used by every PyBullet client in the project."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.resetSimulation(physicsClientId=client_id)
    p.setGravity(0.0, 0.0, config.GRAVITY, physicsClientId=client_id)
    p.setTimeStep(config.FIXED_DT, physicsClientId=client_id)
    p.setRealTimeSimulation(0, physicsClientId=client_id)
    p.setPhysicsEngineParameter(
        fixedTimeStep=config.FIXED_DT,
        numSolverIterations=config.SOLVER_ITERATIONS,
        numSubSteps=1,
        deterministicOverlappingPairs=1,
        physicsClientId=client_id,
    )
    if config.LOAD_GROUND_PLANE:
        p.loadURDF("plane.urdf", physicsClientId=client_id)


class StepPipeline(object):

    def __init__(self, client_id, robot, controller, trajectory, injector, recorder):
        self.thermal = ThermalModel(robot.n, config.FIXED_DT, robot.max_forces) \
            if config.THERMAL_ENABLED else None
        self.client_id = client_id
        self.robot = robot
        self.controller = controller
        self.trajectory = trajectory
        self.injector = injector
        self.recorder = recorder
        self.reset_state()

    def reset_state(self):
        n = self.robot.n
        home = self.robot.home_pose
        # Measured signals (what sensors report, what control and logs see).
        self.positions = home.copy()
        self.velocities = np.zeros(n)
        self.torques = np.zeros(n)
        # True joint state from PyBullet.
        self.true_positions = home.copy()
        self.true_velocities = np.zeros(n)
        # Commanded reference, before actuator faults such as backlash.
        self.targets = home.copy()
        self.target_velocities = np.zeros(n)
        self.target_accelerations = np.zeros(n)
        self.temperatures = np.full(n, config.THERMAL_AMBIENT_C)
        if self.thermal is not None:
            self.thermal.reset()

    def step(self, sim_time, braking):
        """
        Advance one fixed step starting at sim_time. Returns the new time.
        braking=True applies the emergency brake instead of the trajectory.
        """
        dt = config.FIXED_DT
        if braking:
            self.injector.begin_step(sim_time, self.true_positions)
            self.controller.engage_brakes()
        else:
            q_ref, qd_ref, qdd_ref = self.trajectory.step(dt)
            reference = self.controller.clamp_positions(q_ref)
            drive_q, drive_qd = self.injector.begin_step(
                sim_time, self.true_positions, reference, qd_ref)
            self.controller.apply(drive_q, drive_qd, qdd_ref,
                                  self.positions, self.velocities,
                                  sensor_offset=self.injector.sensor_offset,
                                  true_velocities=self.true_velocities)
            self.targets = reference
            self.target_velocities = np.asarray(qd_ref)
            self.target_accelerations = qdd_ref

        p.stepSimulation(physicsClientId=self.client_id)
        sim_time += dt

        q, qd, tau = self.robot.get_states()
        applied = self.controller.applied_torques
        if applied is not None:
            tau = applied
        self.true_positions = q
        self.true_velocities = qd
        mq, mqd, mtau = self.injector.measure(q, qd, tau)
        self.positions = mq
        self.velocities = mqd
        self.torques = mtau

        if self.thermal is not None:
            # Driven by the TRUE torque and speed: temperature is a physical
            # consequence of the losses, not of what a faulty sensor reports.
            self.thermal.set_cooling_scale(self.injector.cooling_scale)
            self.temperatures = self.thermal.step(tau, qd).copy()

        self.recorder.record(sim_time, mq, np.asarray(self.targets, dtype=np.float64),
                             mqd, np.asarray(mtau, dtype=np.float64),
                             self.robot.reactions, self.injector.labels,
                             self.temperatures if self.thermal is not None else None)
        return sim_time
