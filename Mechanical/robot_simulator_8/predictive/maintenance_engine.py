"""
SIMULATION-BASED MAINTENANCE RECOMMENDATIONS.

Maps a joint's health result and its evidence pattern to one of the actions
below, with the reasoning spelled out. These are rules written for the
simulator's fault models. They are not certified industrial diagnoses and must
not be used to maintain real equipment.
"""

import config

DISCLAIMER = ("SIMULATION-BASED MAINTENANCE RECOMMENDATIONS - rule based suggestions for the "
              "simulated robot, not certified industrial diagnoses.")

NO_ACTION = "No action required"
MONITOR = "Monitor joint"
INSPECT_JOINT = "Inspect joint"
LUBRICATION = "Check lubrication"
MOTOR_LOAD = "Inspect motor load"
GEARBOX = "Inspect gearbox"
ENCODER = "Inspect encoder"
SCHEDULE = "Schedule maintenance"
STOP = "Stop machine immediately"

ACTIONS = (NO_ACTION, MONITOR, INSPECT_JOINT, LUBRICATION, MOTOR_LOAD, GEARBOX, ENCODER,
           SCHEDULE, STOP)
PRIORITY = dict((action, index) for index, action in enumerate(ACTIONS))


def _elevated(z, name):
    value = z.get(name)
    return value is not None and value >= config.MAINT_EVIDENCE_Z


def _probable_cause(z, trend_z, temperature_rise=None, temperature_ratio=None):
    """(action, reason) for the dominant evidence pattern, or (None, None). z is signed."""
    hot = (temperature_rise is not None and temperature_rise >= config.THERMAL_WARN_RISE_C) or \
          (temperature_ratio is not None and temperature_ratio >= config.THERMAL_RATIO_WARN)
    error = _elevated(z, "tracking_error")
    torque = _elevated(z, "torque")
    load = _elevated(z, "force") or _elevated(z, "power")
    acceleration = _elevated(z, "acceleration")
    torque_capped = z.get("torque") is not None and z["torque"] <= config.MAINT_TORQUE_DROP_Z
    rising = trend_z is not None and trend_z >= config.MAINT_TREND_Z

    if hot and (_elevated(z, "torque") or rising_torque(z, trend_z)):
        return LUBRICATION, ("temperature %.0f C over ambient together with raised torque: "
                             "friction losses heating the joint" % temperature_rise)
    if hot and (_elevated(z, "force") or _elevated(z, "power")):
        return MOTOR_LOAD, ("temperature %.0f C over ambient with extra load: the joint is "
                            "working beyond its thermal rating" % temperature_rise)
    if hot:
        return INSPECT_JOINT, ("temperature %.0f C over ambient without extra torque or load: "
                               "check the cooling path (fan, filter, ambient)" % temperature_rise)
    if acceleration and z["acceleration"] >= max(z.get("tracking_error", 0.0), z.get("torque", 0.0)):
        return ENCODER, ("acceleration noise is the dominant deviation: a noisy position "
                         "measurement makes the servo shake the joint")
    if error and torque_capped:
        return MOTOR_LOAD, ("tracking fails while torque is below normal: the motor or drive "
                            "cannot deliver the required torque for the load")
    if load and not error:
        return MOTOR_LOAD, "reaction force or power above baseline while tracking holds: extra load"
    if error and torque:
        return GEARBOX, "tracking error and torque up together: drivetrain resistance or seizure"
    if torque and rising and not error:
        return LUBRICATION, "torque rising over time with normal tracking: increasing friction"
    if error and not torque:
        return GEARBOX, "tracking error without extra torque: backlash or play in the drivetrain"
    if torque:
        return LUBRICATION, "elevated torque with normal tracking"
    if load:
        return MOTOR_LOAD, "reaction force or power above baseline"
    return None, None


def rising_torque(z, trend_z):
    return trend_z is not None and trend_z >= config.MAINT_TREND_Z and z.get("torque", 0.0) > 1.0


def recommend(health_result, z, trend_z=None, temperature_rise=None, temperature_ratio=None):
    """Returns dict(action, secondary, reason)."""
    health = health_result.get("health")
    severity = health_result.get("severity")
    z = z or {}
    if health is None:
        return {"action": MONITOR, "secondary": None,
                "reason": "no baseline evidence yet: train a baseline to score this joint"}

    cause, cause_reason = _probable_cause(z, trend_z, temperature_rise, temperature_ratio)
    strongest = max(z.values()) if z else 0.0

    seizure_signature = _elevated(z, "tracking_error") and (_elevated(z, "torque") or _elevated(z, "force"))
    if health < config.MAINT_STOP_HEALTH and seizure_signature:
        return {"action": STOP, "secondary": cause,
                "reason": "health %.0f%% with tracking error and load both far above baseline "
                          "(z up to %.0f): continuing risks damage%s"
                          % (health, strongest, ("; %s" % cause_reason) if cause_reason else "")}
    if severity == "CRITICAL":
        return {"action": SCHEDULE, "secondary": cause,
                "reason": "health %.0f%%%s" % (health, ("; " + cause_reason) if cause_reason else "")}
    if severity == "WARNING":
        if cause is not None:
            return {"action": cause, "secondary": SCHEDULE if health < 55.0 else None,
                    "reason": cause_reason}
        return {"action": INSPECT_JOINT, "secondary": None,
                "reason": "health %.0f%% without a single dominant signal" % health}
    if severity == "MINOR_DEGRADATION":
        rul = health_result.get("rul_s")
        return {"action": MONITOR, "secondary": cause,
                "reason": "minor degradation%s%s" % (
                    ("; %s" % cause_reason) if cause_reason else "",
                    ("; trend reaches CRITICAL in about %.0f s" % rul) if rul else "")}
    return {"action": NO_ACTION, "secondary": None, "reason": "all evidence within baseline"}
