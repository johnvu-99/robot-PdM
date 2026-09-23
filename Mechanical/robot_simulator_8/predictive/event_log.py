"""
Maintenance event log.

Turns the stream of health results into discrete, timestamped events, only on
edges (a status change, a deviation crossing its threshold, a new
recommendation), so a persistent fault produces a few meaningful lines instead
of one per evaluation.
"""

import collections
import csv
import datetime

import config

EVENT_COLUMNS = ("timestamp", "sim_time", "robot", "joint", "event", "severity", "details")

SEVERITY_RANK = {"HEALTHY": 0, "MINOR_DEGRADATION": 1, "WARNING": 2, "CRITICAL": 3, "NO_DATA": -1}


class EventLog(object):

    def __init__(self):
        self.events = collections.deque(maxlen=config.EVENT_LOG_MAX)
        self._next_id = 1
        self._state = {}

    def reset_state(self):
        """Forget edge state (simulation restarted) but keep the history."""
        self._state = {}

    def _add(self, sim_time, robot, joint, event, severity, details):
        self.events.append({
            "id": self._next_id,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "sim_time": round(float(sim_time), 2),
            "robot": robot, "joint": joint, "event": event,
            "severity": severity, "details": details,
        })
        self._next_id += 1

    def since(self, last_id):
        return [e for e in self.events if e["id"] > last_id]

    def update(self, sim_time, robot, joint, health, recommendation, if_status,
               deviations, trend_z, temperature_rise=None, temperature_ratio=None):
        state = self._state.setdefault(joint, {
            "severity": "HEALTHY", "anomaly": False, "armed": {}, "held": {}, "trend": False,
            "trend_held": 0, "action": None,
        })
        label = "Joint %d" % (joint + 1)

        anomaly = if_status in ("WARNING", "CRITICAL")
        if anomaly and not state["anomaly"]:
            self._add(sim_time, robot, joint + 1, "%s anomaly detected" % label, "WARNING",
                      "isolation forest %s" % if_status)
        state["anomaly"] = anomaly

        for feature, percent in deviations.items():
            armed = state["armed"].get(feature, True)
            held = state["held"].get(feature, 0) + 1 if percent >= config.EVENT_DEVIATION_PERCENT else 0
            state["held"][feature] = held
            if armed and held >= config.EVENT_PERSISTENCE:
                self._add(sim_time, robot, joint + 1, "%s %s %+.0f%%" % (
                    label, feature.replace("_RMS", " RMS").replace("_", " "), percent),
                    "INFO", "above baseline mean")
                state["armed"][feature] = False
            elif not armed and percent < config.EVENT_DEVIATION_PERCENT * config.EVENT_DEVIATION_HYSTERESIS:
                state["armed"][feature] = True

        if trend_z is not None and trend_z >= config.MAINT_TREND_Z:
            state["trend_held"] += 1
        elif trend_z is None or trend_z < config.MAINT_TREND_Z * config.EVENT_DEVIATION_HYSTERESIS:
            state["trend_held"] = 0
        if temperature_rise is not None:
            hot = (temperature_rise >= config.THERMAL_WARN_RISE_C
                   or (temperature_ratio is not None and temperature_ratio >= config.THERMAL_RATIO_WARN))
            if hot and not state.get("hot"):
                self._add(sim_time, robot, joint + 1, "%s temperature %.0f C over ambient" % (
                    label, temperature_rise), "WARNING",
                    "%.1fx its normal rise" % temperature_ratio if temperature_ratio
                    else "thermal warning level is %.0f C" % config.THERMAL_WARN_RISE_C)
            elif not hot and temperature_rise < config.THERMAL_WARN_RISE_C * config.EVENT_DEVIATION_HYSTERESIS \
                    and state.get("hot"):
                self._add(sim_time, robot, joint + 1, "%s temperature back to normal" % label,
                          "INFO", "%.0f C over ambient" % temperature_rise)
            state["hot"] = hot

        rising = state["trend_held"] >= config.EVENT_PERSISTENCE or (state["trend"] and state["trend_held"] > 0)
        if rising and not state["trend"]:
            self._add(sim_time, robot, joint + 1, "%s degradation trend detected" % label, "WARNING",
                      "trend z=%.1f" % trend_z)
        state["trend"] = rising

        severity = health["severity"]
        if severity != "NO_DATA" and severity != state["severity"]:
            worse = SEVERITY_RANK[severity] > SEVERITY_RANK.get(state["severity"], 0)
            self._add(sim_time, robot, joint + 1, "%s %s" % (label, severity),
                      severity if severity in ("WARNING", "CRITICAL") else "INFO",
                      "health %.0f%% (%s)" % (health["health"], "worse" if worse else "improved"))
            state["severity"] = severity

        action = recommendation["action"] if health["health"] is not None else None
        if action != state.get("pending_action"):
            state["pending_action"] = action
            state["pending_count"] = 0
        state["pending_count"] = state.get("pending_count", 0) + 1
        # A recommendation must hold for a few evaluations before it is logged,
        # except "stop", which is logged immediately.
        settled = state["pending_count"] >= config.EVENT_PERSISTENCE or action == "Stop machine immediately"
        if action is not None and action != state["action"] and settled:
            if state["action"] is not None or action != "No action required":
                self._add(sim_time, robot, joint + 1,
                          "Maintenance recommendation: %s" % action.lower(),
                          "CRITICAL" if action == "Stop machine immediately" else "INFO",
                          recommendation["reason"])
            state["action"] = action


def export_csv(events, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVENT_COLUMNS)
        for event in events:
            writer.writerow([event[c] for c in EVENT_COLUMNS])
    return path
