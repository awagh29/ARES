"""Port of ``MapCorrectionController.m`` (the MATLAB System block that flies the drone).

Each 20 Hz tick: follow the planned route, form a nominal acceleration, pass
it through the barrier QP (``correct_motion``) and allocate it to roll, pitch,
yaw-rate and thrust exactly as the MATLAB block does. The Gazebo runner uses
the safe acceleration ``acceleration_ned`` and ``yaw_rate``; the roll/pitch/
thrust allocation is kept so parity with MATLAB can be tested.
"""

from __future__ import annotations

import math
import time as wallclock
from dataclasses import dataclass

import numpy as np

from ..scene import SceneMap
from .barrier import BarrierInfo, correct_motion
from .lidar import measured_surfaces, virtual_lidar
from .options import NavigationOptions

GRAVITY = 9.81
MATLAB_PLANT_MASS_KG = 0.1


@dataclass(frozen=True)
class ControllerOutput:
    command: np.ndarray  # [roll, pitch, yaw_rate, thrust] for the MATLAB guidance plant
    acceleration_ned: np.ndarray  # safe acceleration chosen by the barrier QP
    nominal_ned: np.ndarray  # acceleration before the barrier QP
    reference_ned: np.ndarray  # active route waypoint
    yaw_rate: float
    info: BarrierInfo
    lidar_available: bool
    goal_reached: bool
    stop: bool
    waypoint_index: int  # 1-based, as in MATLAB
    measured_point_count: int
    compute_seconds: float


class MapCorrectionController:
    def __init__(self, scene: SceneMap, options: NavigationOptions, route_ned: np.ndarray):
        self.scene = scene
        self.options = options
        self.route = np.asarray(route_ned, dtype=float)
        self.reset()

    def reset(self) -> None:
        self.index = 1  # 0-based here; MATLAB starts at 2 (1-based)
        self.previous = np.zeros(3)
        self.last_time = -math.inf
        self.last_scan_index = -1
        self.measured_world = np.zeros((0, 3))
        self.cached: ControllerOutput | None = None

    def step(self, points, position, velocity, euler_zyx, lidar_available: bool, time_s: float) -> ControllerOutput:
        """``stepImpl``. ``points`` is the (10740,3) sensor-frame scan in MATLAB order, or None."""
        time_s = float(time_s)
        if time_s == self.last_time and self.cached is not None:
            return self.cached
        o = self.options
        timer = wallclock.perf_counter()
        if not (math.isfinite(time_s) and time_s >= 0 and (
                math.isinf(self.last_time) or abs(time_s - self.last_time - o.ControlSampleTimeS) < 1e-6)):
            raise RuntimeError(f"Expected chronological control ticks, got t={time_s} after {self.last_time}.")
        p = np.asarray(position, dtype=float).reshape(3)
        v = np.asarray(velocity, dtype=float).reshape(3)
        euler = np.asarray(euler_zyx, dtype=float).reshape(3)
        if not np.all(np.isfinite(np.concatenate((p, v, euler)))):
            raise RuntimeError("Motion state must be finite.")
        available = bool(lidar_available)
        measured_world = np.zeros((0, 3))
        if available and points is not None:
            # Supplement the map with measured surfaces; one nearest return per angular bin.
            scan_index = math.floor(time_s / 0.1 + 1e-7)
            if scan_index != self.last_scan_index:
                self.measured_world = measured_surfaces(self.scene, points, p, euler, o.GeometryUncertaintyM)
                self.last_scan_index = scan_index
            measured_world = self.measured_world
        elif not available:
            # The MATLAB block ray-casts the map here only for logging (sector
            # features); no value from the unavailable scan reaches the control law.
            pass
        while self.index < self.route.shape[0] - 1 and np.linalg.norm(p - self.route[self.index]) <= o.WaypointToleranceM:
            self.index += 1
        reference = self.route[self.index]
        delta = reference - p
        distance = float(np.linalg.norm(delta))
        speed = min(o.CruiseSpeedMps, o.PositionGain * distance,
                    math.sqrt(2 * o.BrakingDecelerationMps2 * max(distance - 0.1, 0.0)))
        desired_velocity = speed * delta / max(distance, 1e-12)
        nominal = o.VelocityGain * (desired_velocity - v)
        u, info = correct_motion(p, v, nominal, self.previous, self.scene.box_min, self.scene.box_max, o, measured_world)
        # Static NED allocator of the MATLAB guidance plant (kept for parity).
        yaw = euler[0]
        vertical = GRAVITY - u[2]
        pitch = math.atan2(-u[0] * math.cos(yaw) - u[1] * math.sin(yaw), vertical)
        roll = math.atan2((-u[0] * math.sin(yaw) + u[1] * math.cos(yaw)) * math.cos(pitch), vertical)
        thrust = MATLAB_PLANT_MASS_KG * vertical / (math.cos(pitch) * math.cos(roll))
        heading = math.atan2(delta[1], delta[0])
        if np.linalg.norm(delta[0:2]) < 0.1:
            heading = yaw
        yaw_rate = max(-0.5, min(0.5, math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))))
        command = np.array([roll, pitch, yaw_rate, thrust], dtype=np.float32)
        goal = (np.linalg.norm(p - o.goal_state[0:3]) <= o.GoalPositionToleranceM
                and np.linalg.norm(v - o.goal_state[3:6]) <= o.GoalVelocityToleranceMps)
        stop = (not info.Feasible) or (goal and time_s >= o.ControlSampleTimeS)
        output = ControllerOutput(
            command=command, acceleration_ned=u.copy(), nominal_ned=nominal, reference_ned=reference.copy(),
            yaw_rate=float(yaw_rate), info=info, lidar_available=available, goal_reached=bool(goal),
            stop=bool(stop), waypoint_index=self.index + 1, measured_point_count=int(measured_world.shape[0]),
            compute_seconds=wallclock.perf_counter() - timer)
        self.previous = u
        self.last_time = time_s
        self.cached = output
        return output


def debug_virtual_scan(scene: SceneMap, position, euler_zyx) -> np.ndarray:
    """The map ray cast the MATLAB block computes during outages (diagnostics only)."""
    cloud, _, _ = virtual_lidar(scene, position, euler_zyx)
    return cloud
