"""One vehicle: waypoint follower, 3DVFH+, the Simulink PID cascade, and the guidance plant.

LiDAR is sampled at 10 Hz. The avoidance carrot is the scan position plus
0.5 m along the unit direction, and it is held until the next scan. While the
seed-513 schedule is off, those outputs are ignored and the 5 m waypoint
lookahead is tracked at the 1 ms plant rate. That is the Enable Obstacle
Avoidance switch in the Simulink model.
"""

from __future__ import annotations

import math

import numpy as np

from .guidance import acceleration_chart, initial_state, rk4_step, wrap_pi
from .lidar import cast_obstacles
from .mission import (
    AVOIDANCE_LOOKAHEAD_M,
    DRONE_EAST_OFFSETS_M,
    INITIAL_NED,
    LIDAR_DT_S,
    PLANT_DT_S,
    POSITION_GAINS,
    POSITION_LIMITS,
    VELOCITY_LIMITS,
    WAYPOINTS_NED,
    YAW_P_GAIN,
    YAW_RATE_LIMIT_RADPS,
    obstacle_boxes_ned,
)
from .pid import DiscretePID
from .schedule import lidar_guidance_enabled
from .vfh3d import VFH3D
from .waypoint import WaypointFollower


class Pilot:
    def __init__(self, drone_index: int, schedule_times: np.ndarray, schedule_enabled: np.ndarray):
        self.drone_index = drone_index
        self.name = f"uav_{drone_index + 1:03d}"
        self.east_offset = DRONE_EAST_OFFSETS_M[drone_index]
        self.schedule_times = schedule_times
        self.schedule_enabled = schedule_enabled
        self.box_min, self.box_max = obstacle_boxes_ned(self.east_offset)
        self.waypoints = WAYPOINTS_NED.copy()
        self.waypoints[:, 1] += self.east_offset
        self.state = initial_state(INITIAL_NED + np.array([0.0, self.east_offset, 0.0]))
        self.follower = WaypointFollower(self.waypoints)
        self.vfh = VFH3D()
        self.position_pid = [
            DiscretePID(*gains, *limits, PLANT_DT_S)
            for gains, limits in zip(POSITION_GAINS, POSITION_LIMITS)
        ]
        self.velocity_pid = [
            DiscretePID(*gains, *limits, PLANT_DT_S)
            for gains, limits in zip(POSITION_GAINS, VELOCITY_LIMITS)
        ]
        self.scans_done = 0
        self.finished = False
        self.avoidance_position = self.state[:3].copy()
        self.avoidance_yaw = 0.0
        self.using_avoidance = False
        self.lidar_enabled = True
        self.finish_time_s = None
        self.goal_distance = float(np.linalg.norm(self.state[:3] - self.waypoints[-1]))

    def step(self, step_index: int) -> None:
        if self.finished:
            return
        time_s = step_index * PLANT_DT_S
        lookahead, path_yaw, finished = self.follower.step(self.state[:3])
        if step_index % int(round(LIDAR_DT_S / PLANT_DT_S)) == 0:
            self._scan(lookahead)
        self.lidar_enabled = lidar_guidance_enabled(self.schedule_times, self.schedule_enabled, time_s)
        if self.lidar_enabled:
            desired = self.avoidance_position
            desired_yaw = self.avoidance_yaw
            self.using_avoidance = True
        else:
            desired = lookahead
            desired_yaw = path_yaw
            self.using_avoidance = False
        velocity_command = np.array([
            loop.step(float(desired[axis] - self.state[axis]))
            for axis, loop in enumerate(self.position_pid)
        ])
        acceleration = np.array([
            loop.step(float(velocity_command[axis] - self.state[3 + axis]))
            for axis, loop in enumerate(self.velocity_pid)
        ])
        roll, pitch, thrust = acceleration_chart(acceleration, float(self.state[8]))
        yaw_rate = min(
            YAW_RATE_LIMIT_RADPS,
            max(-YAW_RATE_LIMIT_RADPS, YAW_P_GAIN * wrap_pi(desired_yaw - float(self.state[8]))),
        )
        command = np.array((roll, pitch, yaw_rate, thrust))
        self.state = rk4_step(self.state, command, PLANT_DT_S)
        self.goal_distance = float(np.linalg.norm(self.state[:3] - self.waypoints[-1]))
        if finished and self.finish_time_s is None:
            self.finish_time_s = time_s
            self.finished = True

    def _scan(self, lookahead: np.ndarray) -> None:
        self.scans_done += 1
        hits = cast_obstacles(
            self.state[:3], float(self.state[6]), float(self.state[7]), float(self.state[8]),
            self.box_min, self.box_max,
        )
        direction, yaw, _status = self.vfh.step(
            self.state[:3], float(self.state[6]), float(self.state[7]), float(self.state[8]),
            hits, lookahead,
        )
        self.avoidance_position = self.state[:3] + AVOIDANCE_LOOKAHEAD_M * direction
        self.avoidance_yaw = yaw

    @property
    def position_ned(self) -> np.ndarray:
        return self.state[:3]

    def enu_position(self) -> tuple[float, float, float]:
        north, east, down = self.state[:3]
        return float(east), float(north), float(-down)


def minimum_surface_clearance(position_ned: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> float:
    """Signed distance from the vehicle center to the nearest building face, in meters."""
    point = np.asarray(position_ned, dtype=float)
    best = math.inf
    for lower, upper in zip(box_min, box_max):
        outside = np.maximum(lower - point, point - upper)
        outside = np.maximum(outside, 0.0)
        outside_distance = float(np.linalg.norm(outside))
        if outside_distance > 0.0:
            best = min(best, outside_distance)
        else:
            inside = np.minimum(point - lower, upper - point)
            best = min(best, -float(np.min(inside)))
    return best
