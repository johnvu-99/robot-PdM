"""
Motor thermal model.

A first order lumped model per joint, stepped with the physics but outside the
rigid body solver (temperature does not feed back into the dynamics):

    P_loss = k_copper * tau^2 + k_friction * |omega| + P_idle      [W]
    dT/dt  = (P_loss * R_thermal - (T - T_ambient)) / tau_thermal  [C/s]

    T          motor/gearbox temperature at the sensor position
    R_thermal  steady state rise per watt of loss          [C/W]
    tau_thermal thermal time constant (minutes, not seconds) [s]

Steady state is T_ambient + P_loss * R_thermal, so a joint that works harder
settles hotter, and any fault that raises torque or friction raises temperature
slowly afterwards. Cooling faults raise R_thermal.

This is a plausible simulation model for generating labelled data, not a
calibrated model of a specific KUKA motor. No Qt. No ML.
"""

import numpy as np

import config


class ThermalModel(object):

    def __init__(self, n_joints, dt, max_forces=None):
        self.n = int(n_joints)
        self.dt = float(dt)
        self.ambient = float(config.THERMAL_AMBIENT_C)
        # Losses scale with the joint's rating: a joint built for 300 N m does
        # not overheat at the same torque as a small wrist motor.
        rating = np.full(self.n, config.FALLBACK_MAX_FORCE, dtype=np.float64)
        if max_forces is not None and len(max_forces) == self.n:
            rating = np.maximum(np.asarray(max_forces, dtype=np.float64), 1.0)
        self.copper = config.THERMAL_COPPER_LOSS * (config.THERMAL_REFERENCE_TORQUE / rating) ** 2
        self.friction = np.full(self.n, config.THERMAL_FRICTION_LOSS)
        self.idle = np.full(self.n, config.THERMAL_IDLE_LOSS)
        self.resistance = np.full(self.n, config.THERMAL_RESISTANCE_C_PER_W)
        self.time_constant = np.full(self.n, config.THERMAL_TIME_CONSTANT_S)
        # Fault scaling (1.0 = healthy), set by the fault injector.
        self.cooling_scale = np.ones(self.n)
        self.temperature = np.full(self.n, self.ambient)
        self.loss = np.zeros(self.n)

    def reset(self, ambient=None):
        if ambient is not None:
            self.ambient = float(ambient)
        self.temperature[:] = self.ambient
        self.loss[:] = 0.0
        self.cooling_scale[:] = 1.0

    def set_cooling_scale(self, scale):
        """>1 means worse cooling: the same loss settles at a higher temperature."""
        self.cooling_scale = np.clip(np.asarray(scale, dtype=np.float64), 1.0, 20.0)

    def step(self, torques, velocities):
        tau = np.asarray(torques, dtype=np.float64)
        omega = np.abs(np.asarray(velocities, dtype=np.float64))
        self.loss = self.copper * tau * tau + self.friction * omega + self.idle
        steady = self.ambient + self.loss * self.resistance * self.cooling_scale
        alpha = self.dt / np.maximum(self.time_constant, 1e-6)
        self.temperature += alpha * (steady - self.temperature)
        return self.temperature

    def describe(self):
        return {"ambient_c": self.ambient,
                "resistance_c_per_w": self.resistance.tolist(),
                "time_constant_s": self.time_constant.tolist()}
