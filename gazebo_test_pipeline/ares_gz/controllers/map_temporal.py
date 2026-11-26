"""Atharva's map-temporal CNN + the original VFH/PID stack.

Every 0.1 s (a lidar tick) the MATLAB bridge runs his
``ares.mapTemporal.LidarReconstruction`` (real scan when healthy, CNN/GRU
reconstruction during outages), the UAV Toolbox Waypoint Follower and VFH
Obstacle Avoidance blocks with the model's settings, and his goal hold. It
returns the desired position/yaw. Every 0.01 s the PID cascade below turns
that into an acceleration command for Gazebo.
"""

from __future__ import annotations

import math

import numpy as np

from ..matlab_bridge import BridgeError, MatlabBridge
from ..navigation.options import NavigationOptions
from .base import Command, Controller, VehicleState
from .pid import PidCascade, yaw_rate_command

# Map-temporal mission domain (ares.mapTemporal.defaultOptions): N/E -5..40 m, altitude 1..10 m.
TRAINED_BOUNDS_NED = ((-5.0, 40.0), (-5.0, 40.0), (-10.0, -1.0))
OPTION_FIELDS = ("StartState", "GoalState", "StartEulerZYXRad", "StartBodyRatesRadps", "StopTimeS", "LidarMode",
                 "NumberOfOutages", "OutageDurationS", "InitialHealthyTimeS", "FinalHealthyTimeS",
                 "MinimumRecoveryTimeS", "OutageWindowEndS", "OutageIntervalsS", "Seed", "VehicleRadiusM",
                 "MinimumSurfaceClearanceM", "GoalPositionToleranceM", "GoalVelocityToleranceMps", "WorldBoundsNedM")


class MapTemporalController(Controller):
    name = "map-temporal-cnn"
    tick_s = 0.01
    guidance_period_s = 0.1

    def __init__(self, bridge: MatlabBridge, model_path: str, export_python: str = ""):
        self.bridge = bridge
        self.model_path = model_path
        self.export_python = export_python
        self.started = False

    def start(self, options: NavigationOptions, route_hint=None) -> dict:
        payload = {}
        for name in OPTION_FIELDS:
            value = getattr(options, name)
            payload[name] = [list(row) for row in value] if name in ("WorldBoundsNedM", "OutageIntervalsS") else value
        reply = self.bridge.init(payload, self.model_path, self.export_python)
        self.started = True
        self.pid = PidCascade(self.tick_s)
        self.reference = np.asarray(options.StartState[0:3], float)
        self.yaw_reference = float(options.StartEulerZYXRad[0])
        self.terminal = False
        self.guidance: dict = {}
        self.guidance_ticks = 0
        self.reconstruction_ticks = 0
        self.max_inference_s = 0.0
        return {"waypoints_ned": reply.get("waypoints"), "artifact_path": reply.get("artifact_path"),
                "model_source_sha256": reply.get("source_sha256"), "artifact_sha256": reply.get("artifact_sha256"),
                "vfh_backend": reply.get("vfh_backend"),
                "matlab_parity_passed": reply.get("parity_passed"),
                "avoidance_safety_distance_m": reply.get("avoidance_safety_distance"),
                "pid_rate_hz": round(1 / self.tick_s), "guidance_rate_hz": round(1 / self.guidance_period_s)}

    def step(self, state: VehicleState, lidar_available: bool, points_matlab_order) -> Command:
        t = state.time_s
        healthy = True
        if abs(t / self.guidance_period_s - round(t / self.guidance_period_s)) < 1e-6:
            try:
                self.guidance = self.bridge.step(round(t, 6), state.position_ned, state.velocity_ned,
                                                 state.euler_zyx, state.body_rates_frd, lidar_available,
                                                 points_matlab_order if lidar_available else None)
            except BridgeError as error:
                if "ARES:Temporal" in str(error):  # the model itself rejected its input/output
                    return Command(np.zeros(3), 0.0, stop=True, healthy=False,
                                   record={"controller_error": str(error)[:200]})
                raise
            self.reference = np.asarray(self.guidance["desired_position"], float)
            desired_yaw = self.guidance.get("desired_yaw")
            self.yaw_reference = float(desired_yaw) if desired_yaw is not None else math.nan
            self.terminal = bool(self.guidance.get("terminal"))
            self.guidance_ticks += 1
            if self.guidance.get("reconstruction_used"):
                self.reconstruction_ticks += 1
                self.max_inference_s = max(self.max_inference_s, float(self.guidance.get("inference_s") or 0.0))
        acceleration, velocity_command = self.pid.step(self.reference, state.position_ned, state.velocity_ned)
        yaw_rate = yaw_rate_command(self.yaw_reference, float(state.euler_zyx[0]))
        guidance = self.guidance
        record = {
            "ref_n": self.reference[0], "ref_e": self.reference[1], "ref_d": self.reference[2],
            "yaw_ref": self.yaw_reference, "vcmd_n": velocity_command[0], "vcmd_e": velocity_command[1],
            "vcmd_d": velocity_command[2], "reconstruction_used": int(bool(guidance.get("reconstruction_used"))),
            "inference_s": guidance.get("inference_s"), "vfh_status": guidance.get("avoidance_status"),
            "goal_hold": int(bool(guidance.get("holding"))), "history_steps": guidance.get("history_steps"),
            "blackout_age_s": guidance.get("blackout_age_s"),
        }
        return Command(acceleration, yaw_rate, stop=self.terminal, goal_reached=self.terminal,
                       healthy=healthy, record=record)

    def finish(self) -> None:
        if self.started:
            try:
                self.bridge.end_run()
            except BridgeError:
                pass
            self.started = False
