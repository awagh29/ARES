"""Multirotor waypoint follower used by the Simulink block.

Lookahead distance is the straight-line distance from the vehicle to a point
on the current path segment. A segment advances on the documented hyperplane
condition or inside the transition radius. Hover-at-last is off, and desired
yaw is the heading of the active segment because the waypoint input has no yaw.
"""

from __future__ import annotations

import math

import numpy as np

from .mission import TRANSITION_RADIUS_M, WAYPOINT_LOOKAHEAD_M, WAYPOINT_MIN_LOOKAHEAD_M


class WaypointFollower:
    def __init__(self, waypoints_ned: np.ndarray):
        self.waypoints = np.asarray(waypoints_ned, dtype=float)
        if len(self.waypoints) < 2:
            raise ValueError("The follower needs at least two waypoints.")
        self.segment = 0

    def step(self, position_ned) -> tuple[np.ndarray, float, bool]:
        position = np.asarray(position_ned, dtype=float)
        last = len(self.waypoints) - 1
        while self.segment < last - 1:
            start = self.waypoints[self.segment]
            end = self.waypoints[self.segment + 1]
            if self._reached(position, start, end):
                self.segment += 1
            else:
                break
        start = self.waypoints[self.segment]
        end = self.waypoints[self.segment + 1]
        finished = self.segment == last - 1 and self._reached(position, start, end)
        distance = max(WAYPOINT_LOOKAHEAD_M, WAYPOINT_MIN_LOOKAHEAD_M)
        point = self._carrot(position, start, end, distance)
        delta = end - start
        yaw = math.atan2(float(delta[1]), float(delta[0])) if np.hypot(delta[0], delta[1]) > 1e-9 else 0.0
        return point, yaw, finished

    @staticmethod
    def _reached(position: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
        if float(np.linalg.norm(position - end)) <= TRANSITION_RADIUS_M:
            return True
        return float(np.dot(position - end, end - start)) >= 0.0

    @staticmethod
    def _carrot(position: np.ndarray, start: np.ndarray, end: np.ndarray, distance: float) -> np.ndarray:
        direction = end - start
        span = float(np.linalg.norm(direction))
        if span < 1e-9:
            return end.copy()
        unit = direction / span
        relative = position - start
        projection = float(np.dot(unit, relative))
        cross_track_sq = float(np.dot(relative, relative) - projection * projection)
        remainder = distance * distance - max(cross_track_sq, 0.0)
        if remainder < 0.0:
            travel = min(max(projection, 0.0), span)
        else:
            # Forward intersection. Hover-at-last is off, so the final segment may extend.
            travel = projection + math.sqrt(remainder)
            if travel < 0.0:
                travel = 0.0
        return start + travel * unit
