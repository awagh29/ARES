"""Scoring, using the same definitions as the MATLAB solution.

``evaluate_axis_run`` ports ``src/+ares/+data/evaluateAxisRun.m`` (signed 3-D
box distance minus the vehicle radius at every native position sample) and
``mission_success`` adds the extra conditions ``runMission.m`` applies to the
map-correction controller. Extra Gazebo-only measurements (route deviation,
wind) are reported separately and never change pass/fail.
"""

from __future__ import annotations

import math

import numpy as np

from .navigation.options import NavigationOptions
from .scene import SceneMap


def signed_box_distance(positions: np.ndarray, scene: SceneMap) -> np.ndarray:
    """Per-sample minimum signed distance from the drone centre to any building box."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    best = np.full(positions.shape[0], np.inf)
    for k in range(scene.box_min.shape[0]):
        center = (scene.box_min[k] + scene.box_max[k]) / 2.0
        half = (scene.box_max[k] - scene.box_min[k]) / 2.0
        delta = np.abs(positions - center) - half
        outside = np.sqrt(np.sum(np.maximum(delta, 0.0) ** 2, axis=1))
        inside = np.minimum(np.max(delta, axis=1), 0.0)
        best = np.minimum(best, outside + inside)
    return best


def evaluate_axis_run(time_s, position_ned, options: NavigationOptions, scene: SceneMap) -> dict:
    """Port of ``evaluateAxisRun`` with the cfg values ``initializeSimulation`` sets."""
    o = options
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    position = np.asarray(position_ned, dtype=float).reshape(-1, 3)
    radius = o.VehicleRadiusM
    required = o.MinimumSurfaceClearanceM
    goal = o.goal_state[0:3]
    metrics: dict = {
        "positionSampleCount": int(time_s.size),
        "requiredObstacleClearanceM": required,
        "requiredObstacleCenterDistanceM": required + radius,
        "clearanceEvaluation": "signed-3d-box-distance-at-native-position-samples",
    }
    valid = (position.shape[0] == time_s.size and time_s.size >= 2 and np.all(np.isfinite(time_s))
             and np.all(np.isfinite(position)) and np.all(np.diff(time_s) >= 0))
    metrics["telemetryValid"] = bool(valid)
    if not valid:
        metrics.update(missionSuccess=False, safetyChecksPassed=False, terminationReason="invalid-telemetry")
        return metrics
    metrics["initialPositionErrorM"] = float(np.linalg.norm(position[0] - o.start_state[0:3]))
    metrics["initialPositionMatchesPlan"] = metrics["initialPositionErrorM"] <= 1e-4
    distance_to_goal = np.linalg.norm(position - goal, axis=1)
    in_goal = distance_to_goal <= o.GoalPositionToleranceM
    metrics["finalWaypointErrorM"] = float(distance_to_goal[-1])
    metrics["goalReachedAtAnyTime"] = bool(np.any(in_goal))
    metrics["goalReachedAtTermination"] = bool(in_goal[-1])
    metrics["firstGoalEntryTimeS"] = float(time_s[np.argmax(in_goal)]) if np.any(in_goal) else None
    altitude = -position[:, 2]
    metrics["minimumAltitudeM"] = float(np.min(altitude))
    metrics["maximumAltitudeM"] = float(np.max(altitude))
    # initializeSimulation: altitude limits come from WorldBoundsNedM(3,:)
    minimum_altitude = -o.bounds[2, 1]
    maximum_altitude = -o.bounds[2, 0]
    metrics["altitudeWithinLimits"] = bool(metrics["minimumAltitudeM"] >= minimum_altitude
                                           and metrics["maximumAltitudeM"] <= maximum_altitude)
    signed = signed_box_distance(position, scene)
    minimum_index = int(np.argmin(signed))
    metrics["minimumObstacleCenterDistanceM"] = float(signed[minimum_index])
    metrics["minimumObstacleClearanceM"] = float(signed[minimum_index] - radius)
    metrics["minimumClearanceTimeS"] = float(time_s[minimum_index])
    metrics["minimumClearancePositionNedM"] = position[minimum_index].tolist()
    metrics["collisionOccurred"] = metrics["minimumObstacleClearanceM"] <= 0
    metrics["nearCollisionOccurred"] = metrics["minimumObstacleClearanceM"] < required
    metrics["durationS"] = float(time_s[-1] - time_s[0])
    metrics["finalTimeS"] = float(time_s[-1])
    metrics["totalPathLengthM"] = float(np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1)))
    metrics["reachedTimeLimit"] = bool(time_s[-1] >= o.StopTimeS - 1e-3)
    metrics["timedOut"] = metrics["reachedTimeLimit"] and not metrics["goalReachedAtTermination"]
    metrics["safetyChecksPassed"] = metrics["altitudeWithinLimits"] and not metrics["nearCollisionOccurred"]
    metrics["missionSuccess"] = (metrics["goalReachedAtTermination"] and metrics["safetyChecksPassed"]
                                 and metrics["initialPositionMatchesPlan"])
    if metrics["goalReachedAtTermination"]:
        metrics["terminationReason"] = "goal-reached"
    elif metrics["timedOut"]:
        metrics["terminationReason"] = "time-limit"
    else:
        metrics["terminationReason"] = "early-stop-before-goal"
    return metrics


def route_deviation(position_ned: np.ndarray, route_ned: np.ndarray) -> np.ndarray:
    """Distance from each sample to the nearest point of the planned route polyline."""
    position = np.asarray(position_ned, dtype=float).reshape(-1, 3)
    route = np.asarray(route_ned, dtype=float).reshape(-1, 3)
    best = np.full(position.shape[0], np.inf)
    for a, b in zip(route[:-1], route[1:]):
        segment = b - a
        length2 = float(segment @ segment)
        if length2 == 0.0:
            t = np.zeros(position.shape[0])
        else:
            t = np.clip((position - a) @ segment / length2, 0.0, 1.0)
        closest = a + t[:, None] * segment
        best = np.minimum(best, np.linalg.norm(position - closest, axis=1))
    return best


def mission_success(axis_metrics: dict, controller_goal_at_end: bool, all_filters_feasible: bool,
                    availability_matches_schedule: bool, crashed: bool) -> tuple[bool, str]:
    """The extra checks ``runMission.m`` applies on top of ``evaluateAxisRun``."""
    success = bool(axis_metrics.get("missionSuccess") and controller_goal_at_end and all_filters_feasible
                   and availability_matches_schedule and not crashed)
    reason = axis_metrics.get("terminationReason", "invalid-telemetry")
    if crashed:
        reason = "crash"
    elif not all_filters_feasible:
        reason = "map-filter-infeasible-or-invalid"
    elif not controller_goal_at_end and axis_metrics.get("reachedTimeLimit"):
        reason = "time-limit-before-terminal-state"
    return success, reason


def safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None
