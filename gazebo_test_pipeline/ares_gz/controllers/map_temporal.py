"""Atharva's map-temporal CNN + his guidance stack.

Every 0.1 s (a lidar tick) the MATLAB bridge runs his
``ares.mapTemporal.LidarReconstruction`` (real scan when healthy, CNN/GRU
reconstruction during outages), the UAV Toolbox Waypoint Follower, his goal
hold and, depending on ``guidance``:

* ``planner``  (default) his trajectory-sampling local planner + braking governor
  (``ares.mapTemporal.SpeedGovernor``); VFH is bypassed;
* ``governor`` VFH -> position PID -> braking governor;
* ``vfh``      the original VFH/PID stack.

With the governor the bridge returns a velocity command (held for 0.1 s) that
replaces the position loop's output; otherwise it returns a desired position.
Every 0.01 s the PID cascade below turns that into an acceleration for Gazebo.
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
GUIDANCE_MODES = ("planner", "governor", "vfh")
# ares.mapTemporal.configureModel: PlannerPacketOrder / GovernorPacketOrder
PLANNER_FIELDS = ("recovering", "stalled", "recovery_entries", "feasible_candidates", "requested_speed_mps",
                  "predicted_surface_clearance_m", "window_goal_progress_m", "target_n", "target_e", "target_d",
                  "no_feasible_trajectory", "compute_s")
GOVERNOR_FIELDS = ("nominal_vn", "nominal_ve", "nominal_vd", "executed_vn", "executed_ve", "executed_vd",
                   "speed_cap_mps", "requested_corridor_m", "motion_corridor_m", "braking", "approaching",
                   "goal_distance_m")
OPTION_FIELDS = ("StartState", "GoalState", "StartEulerZYXRad", "StartBodyRatesRadps", "StopTimeS", "LidarMode",
                 "NumberOfOutages", "OutageDurationS", "InitialHealthyTimeS", "FinalHealthyTimeS",
                 "MinimumRecoveryTimeS", "OutageWindowEndS", "OutageIntervalsS", "Seed", "VehicleRadiusM",
                 "MinimumSurfaceClearanceM", "GoalPositionToleranceM", "GoalVelocityToleranceMps", "WorldBoundsNedM")


class MapTemporalController(Controller):
    name = "map-temporal-cnn"
    tick_s = 0.01
    guidance_period_s = 0.1

    def __init__(self, bridge: MatlabBridge, model_path: str, export_python: str = "", guidance: str = "planner"):
        if guidance not in GUIDANCE_MODES:
            raise ValueError(f"guidance must be one of {GUIDANCE_MODES}, not {guidance!r}")
        self.bridge = bridge
        self.model_path = model_path
        self.export_python = export_python
        self.guidance_mode = guidance
        self.started = False
        # Optional hook (swarm): (time, state, nominal velocity) -> velocity sent to the velocity PID.
        self.velocity_filter = None

    def start(self, options: NavigationOptions, route_hint=None) -> dict:
        payload = {}
        for name in OPTION_FIELDS:
            value = getattr(options, name)
            payload[name] = [list(row) for row in value] if name in ("WorldBoundsNedM", "OutageIntervalsS") else value
        reply = self.bridge.init(payload, self.model_path, self.export_python, self.guidance_mode)
        self.started = True
        self.pid = PidCascade(self.tick_s)
        self.reference = np.asarray(options.StartState[0:3], float)
        self.yaw_reference = float(options.StartEulerZYXRad[0])
        self.terminal = False
        self.guidance: dict = {}
        self.guidance_ticks = 0
        self.reconstruction_ticks = 0
        self.max_inference_s = 0.0
        self.velocity_command: np.ndarray | None = None  # governed command, zero-order hold
        self.planner_stats = {"recovery_entries": 0, "recovery_ticks": 0, "no_feasible_ticks": 0,
                              "braking_ticks": 0, "approach_ticks": 0, "max_compute_s": 0.0}
        self.recovery_base = 0
        self.goal_ned = np.asarray(options.GoalState[0:3], float)
        self.healthy = True
        self.last_error: str | None = None
        return {"guidance": reply.get("guidance", self.guidance_mode),
                "cruise_speed_mps": reply.get("cruise_speed_mps"), "waypoints_ned": reply.get("waypoints"), "artifact_path": reply.get("artifact_path"),
                "model_source_sha256": reply.get("source_sha256"), "artifact_sha256": reply.get("artifact_sha256"),
                "vfh_backend": reply.get("vfh_backend"),
                "matlab_parity_passed": reply.get("parity_passed"),
                "avoidance_safety_distance_m": reply.get("avoidance_safety_distance"),
                "pid_rate_hz": round(1 / self.tick_s), "guidance_rate_hz": round(1 / self.guidance_period_s)}

    def is_guidance_tick(self, time_s: float) -> bool:
        return abs(time_s / self.guidance_period_s - round(time_s / self.guidance_period_s)) < 1e-6

    def retarget(self, state: VehicleState, goal_ned) -> bool:
        """Load a new goal into the onboard guidance (swarm task). False if MATLAB rejects it."""
        try:
            self.bridge.retarget(goal_ned, state.position_ned, state.velocity_ned, state.euler_zyx)
        except BridgeError as error:
            self.last_error = str(error)[:300]
            return False
        self.recovery_base = self.planner_stats["recovery_entries"]
        self.goal_ned = np.asarray(goal_ned, float)
        self.velocity_command = None if self.guidance_mode == "vfh" else np.zeros(3)
        self.terminal = False
        return True

    def update_guidance(self, state: VehicleState, lidar_available: bool, points_matlab_order) -> bool:
        """One 10 Hz guidance request (MATLAB). Returns False when the model itself failed."""
        t = state.time_s
        try:
            request = getattr(self.pid, "position_request", np.zeros(3))
            self.guidance = self.bridge.step(round(t, 6), state.position_ned, state.velocity_ned,
                                             state.euler_zyx, state.body_rates_frd, lidar_available,
                                             points_matlab_order if lidar_available else None,
                                             velocity_request=request)
        except BridgeError as error:
            if "ARES:Temporal" in str(error):  # the model itself rejected its input/output
                self.healthy = False
                self.last_error = str(error)[:200]
                return False
            raise
        self.reference = np.asarray(self.guidance["desired_position"], float)
        desired_yaw = self.guidance.get("desired_yaw")
        self.yaw_reference = float(desired_yaw) if desired_yaw is not None else math.nan
        self.terminal = bool(self.guidance.get("terminal"))
        self.guidance_ticks += 1
        command = self.guidance.get("velocity_command")
        self.velocity_command = np.asarray(command, float) if command is not None else None
        self._count(self.guidance)
        if self.guidance.get("reconstruction_used"):
            self.reconstruction_ticks += 1
            self.max_inference_s = max(self.max_inference_s, float(self.guidance.get("inference_s") or 0.0))
        return True

    def step(self, state: VehicleState, lidar_available: bool, points_matlab_order) -> Command:
        if self.is_guidance_tick(state.time_s) and not self.update_guidance(state, lidar_available,
                                                                             points_matlab_order):
            return Command(np.zeros(3), 0.0, stop=True, healthy=False,
                           record={"controller_error": self.last_error})
        return self.control(state)

    def control(self, state: VehicleState) -> Command:
        """100 Hz: PID cascade on the latest guidance (optionally filtered by the swarm agent)."""
        override = self.velocity_command
        if self.velocity_filter is not None:
            nominal = override
            if nominal is None:  # vfh mode: the position loop's request is the nominal velocity
                error = np.asarray(self.reference, float) - state.position_ned
                nominal = np.clip(6.0 * error, -3.0, 3.0)
            override = np.asarray(self.velocity_filter(state.time_s, state, np.asarray(nominal, float)), float)
        acceleration, velocity_command = self.pid.step(self.reference, state.position_ned, state.velocity_ned,
                                                       velocity_override=override)
        healthy = self.healthy
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
        planner = guidance.get("planner") or []
        if self.guidance_mode == "planner" and len(planner) == len(PLANNER_FIELDS):
            record.update({f"planner_{name}": value for name, value in zip(PLANNER_FIELDS, planner)})
        governor = guidance.get("governor") or []
        if len(governor) == len(GOVERNOR_FIELDS):
            record.update({f"governor_{name}": value for name, value in zip(GOVERNOR_FIELDS[6:], governor[6:])})
        return Command(acceleration, yaw_rate, stop=self.terminal, goal_reached=self.terminal,
                       healthy=healthy, record=record)

    def _count(self, guidance: dict) -> None:
        stats = self.planner_stats
        planner = guidance.get("planner") or []
        if self.guidance_mode == "planner" and len(planner) == len(PLANNER_FIELDS):
            values = dict(zip(PLANNER_FIELDS, planner))
            stats["recovery_entries"] = int(max(stats["recovery_entries"], self.recovery_base + values["recovery_entries"]))
            stats["recovery_ticks"] += int(values["recovering"] > 0)
            stats["no_feasible_ticks"] += int(values["no_feasible_trajectory"] > 0)
            stats["max_compute_s"] = max(stats["max_compute_s"], float(values["compute_s"]))
        governor = guidance.get("governor") or []
        if len(governor) == len(GOVERNOR_FIELDS):
            values = dict(zip(GOVERNOR_FIELDS, governor))
            stats["braking_ticks"] += int(values["braking"] > 0)
            stats["approach_ticks"] += int(values["approaching"] > 0)

    def summary_extras(self) -> dict:
        extras = {"guidance": self.guidance_mode}
        stats = getattr(self, "planner_stats", None)
        if self.guidance_mode != "vfh" and stats:
            extras.update({
                "planner_recovery_entries": stats["recovery_entries"],
                "planner_recovery_ticks": stats["recovery_ticks"],
                "planner_no_feasible_ticks": stats["no_feasible_ticks"],
                "planner_max_compute_s": stats["max_compute_s"],
                "governor_braking_ticks": stats["braking_ticks"],
                "governor_approach_ticks": stats["approach_ticks"],
            })
        return extras

    def finish(self) -> None:
        if self.started:
            try:
                self.bridge.end_run()
            except BridgeError:
                pass
            self.started = False
