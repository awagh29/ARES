"""Fly one scenario in Gazebo with a controller under test and score it.

Every controller tick (lock-step with Gazebo):
  1. read the drone state (Gazebo odometry -> MATLAB NED conventions);
  2. lidar fault injection: during an outage the scan is withheld and the
     controller is told the lidar is unavailable, exactly like the MATLAB model;
  3. the controller returns an NED acceleration command and a yaw rate;
  4. the acceleration becomes a velocity set-point for Gazebo's multicopter
     controller (``CommandMapper``);
  5. wind: the drag from (air velocity - drone velocity) is applied as a force;
  6. Gazebo advances exactly one tick.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time as wallclock
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .controllers import Controller, VehicleState, make_controller
from .controllers.map_temporal import TRAINED_BOUNDS_NED
from .frames import ENU_NED, gazebo_orientation_to_ned_euler, ned_to_enu
from .gazebo import GazeboSession
from .metrics import evaluate_axis_run, route_deviation, signed_box_distance
from .navigation import availability_at, outage_schedule, plan_path, validate_options, virtual_lidar
from .navigation.lidar import gazebo_cloud_to_matlab_order
from .navigation.options import NavigationOptions
from .paths import RESULTS_ROOT
from .scene import SceneMap, scene_map
from .wind import PRESETS, DragModel, WindField, WindProfile
from .world import QuadParameters, world_sdf

CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "config" / "vehicle_calibration.json"
PHYSICS_STEP_S = 0.001


@dataclass(frozen=True)
class CommandMapper:
    """Turns the controller's acceleration into a velocity set-point for Gazebo's controller.

    Gazebo's MulticopterVelocityControl commands an acceleration of about
    ``gain * (v_cmd - v)``. Choosing ``v_cmd = v + u * dt / (1 - exp(-gain*dt))``
    makes the average acceleration over one tick equal the requested ``u``.
    ``gain``, ``frame`` and the lidar layout are measured by ``scripts/diagnose.py``.
    """

    gain_per_s: float = 2.7 / 1.52
    frame: str = "heading"  # "world" or "heading" (yaw-only body frame)
    lidar_flip_vertical: bool = False
    lidar_flip_horizontal: bool = False
    calibrated: bool = False

    def velocity_setpoint(self, velocity_ned, acceleration_ned, dt_s: float, yaw_enu: float) -> np.ndarray:
        factor = dt_s / (1.0 - math.exp(-self.gain_per_s * dt_s))
        setpoint_enu = ned_to_enu(np.asarray(velocity_ned) + np.asarray(acceleration_ned) * factor)
        if self.frame == "heading":
            c, s = math.cos(yaw_enu), math.sin(yaw_enu)
            setpoint_enu = np.array([c * setpoint_enu[0] + s * setpoint_enu[1],
                                     -s * setpoint_enu[0] + c * setpoint_enu[1], setpoint_enu[2]])
        return setpoint_enu

    @staticmethod
    def load() -> "CommandMapper":
        if CALIBRATION_FILE.exists():
            data = json.loads(CALIBRATION_FILE.read_text())
            return CommandMapper(gain_per_s=float(data["gain_per_s"]), frame=str(data["frame"]),
                                 lidar_flip_vertical=bool(data.get("lidar_flip_vertical", False)),
                                 lidar_flip_horizontal=bool(data.get("lidar_flip_horizontal", False)),
                                 calibrated=True)
        print("WARNING: no config/vehicle_calibration.json; run scripts/diagnose.py first. Using defaults.")
        return CommandMapper()


@dataclass
class RunConfig:
    scenario: str
    lidar_mode: str  # healthy | intermittent | blackout | custom
    wind: str | WindProfile = "calm"
    seed: int = 513
    controller: str = "map-temporal-cnn"
    overrides: dict = field(default_factory=dict)  # NavigationOptions fields (StartState, GoalState, ...)
    gui: bool = False
    headless_rendering: bool = True
    campaign: str = "adhoc"
    start_id: str = "default"
    drag: DragModel = field(default_factory=DragModel)

    def wind_profile(self) -> WindProfile:
        return self.wind if isinstance(self.wind, WindProfile) else PRESETS[self.wind]

    def options(self) -> NavigationOptions:
        values = dict(self.overrides)
        values.update(LidarMode=self.lidar_mode, Seed=int(self.seed))
        if self.controller == "map-temporal-cnn":
            values.setdefault("WorldBoundsNedM", TRAINED_BOUNDS_NED)
        return validate_options(values)


def plan_cached(scene: SceneMap, options: NavigationOptions):
    """Map-correction route (A* takes ~10 s in Python); cached by everything it depends on."""
    keys = ["StartState", "GoalState", "GridResolutionM", "WorldBoundsNedM", "VehicleRadiusM",
            "MinimumSurfaceClearanceM", "GeometryUncertaintyM", "TrackingMarginM", "PlanningExtraMarginM",
            "ClearanceWeight"]
    payload = json.dumps({key: getattr(options, key) for key in keys} | {
        "box_min": scene.box_min.tolist(), "box_max": scene.box_max.tolist()}, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    cache = RESULTS_ROOT / "plan_cache" / f"{digest}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        return np.asarray(data["WaypointsNedM"]), data
    plan = plan_path(scene, options)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(plan.to_dict(), indent=1))
    return plan.WaypointsNedM, plan.to_dict()


def vehicle_state(samples, time_s: float) -> VehicleState:
    """NED state from the latest odometry; velocity by finite difference of the last two 1 ms samples."""
    last = samples[-1]
    position = ENU_NED @ last.position_enu
    velocity = np.zeros(3)
    if len(samples) >= 2:
        previous = samples[-2]
        dt = last.time_s - previous.time_s
        if dt > 0:
            velocity = ENU_NED @ (last.position_enu - previous.position_enu) / dt
    euler = gazebo_orientation_to_ned_euler(*last.orientation_wxyz)
    w = last.angular_velocity  # body FLU -> FRD
    return VehicleState(time_s, position, velocity, euler, np.array([w[0], -w[1], -w[2]]))


def _is_multiple(value: float, period: float) -> bool:
    return abs(value / period - round(value / period)) < 1e-6


def run_scenario(config: RunConfig, controller: Controller | None = None, output_root: Path | None = None,
                 verbose: bool = True) -> dict:
    options = config.options()
    scene = scene_map()
    owns_controller = controller is None
    if controller is None:
        if config.controller != "map-correction":
            raise ValueError("Pass a MapTemporalController (it needs a MatlabBridge); run_tests.py does this.")
        controller = make_controller("map-correction", scene=scene)
    if config.controller == "map-correction":
        route, plan_info = plan_cached(scene, options)
    else:
        route, plan_info = np.vstack((options.start_state[0:3], options.goal_state[0:3])), {"planner": "none (waypoint follower + VFH)"}
    times, enabled, intervals = outage_schedule(options)
    dt = controller.tick_s
    wind = WindField(config.wind_profile(), config.seed, options.StopTimeS, dt)
    mapper = CommandMapper.load()
    quad = QuadParameters()
    run_id = str(uuid.uuid4())
    directory = ((output_root or RESULTS_ROOT / config.campaign) / config.scenario / config.start_id
                 / f"seed{config.seed:04d}_{run_id[:8]}")
    directory.mkdir(parents=True, exist_ok=True)
    world_file = directory / "world.sdf"
    world_file.write_text(world_sdf(scene, options.start_state[0:3], options.StartEulerZYXRad))
    manifest = {
        "schema": "ares-gz-run-v2", "run_id": run_id, "scenario": config.scenario, "campaign": config.campaign,
        "start_id": config.start_id, "start_ned": list(options.StartState[0:3]),
        "controller": config.controller, "seed": config.seed, "lidar_mode": options.LidarMode,
        "options": options.to_dict(), "route_ned": np.asarray(route).tolist(), "plan": plan_info,
        "outage_intervals_s": intervals.tolist(), "wind": wind.describe(), "drag": asdict(config.drag),
        "command_mapper": asdict(mapper), "vehicle": asdict(quad), "controller_tick_s": dt,
    }
    if verbose:
        print(f"[{config.scenario} start {config.start_id} {list(options.StartState[0:2])} seed {config.seed}] "
              f"{config.controller}, {len(intervals)} outage(s), "
              f"wind {wind.profile.name} -> {directory}")

    rows: list[dict] = []
    controller_healthy = True
    availability_ok = True
    crashed = False
    crash_kind = None
    goal_at_end = False
    error = None
    started = wallclock.monotonic()
    session = GazeboSession(world_file, directory / "logs", gui=config.gui,
                            headless_rendering=config.headless_rendering,
                            steps_per_tick=int(round(dt / PHYSICS_STEP_S)))
    ticks = int(round(options.StopTimeS / dt))
    try:
        manifest["controller_info"] = controller.start(options, route)
        session.start()
        start = options.start_state
        state = VehicleState(0.0, start[0:3].copy(), start[3:6].copy(),
                             np.asarray(options.StartEulerZYXRad, float), np.zeros(3))
        for k in range(ticks + 1):
            t = round(k * dt, 9)
            if k > 0:
                state = vehicle_state(session.odometry, t)
                state = VehicleState(t, state.position_ned, state.velocity_ned, state.euler_zyx, state.body_rates_frd)
            available = availability_at(times, enabled, t)
            points = None
            scan_age = None
            if available and _is_multiple(t, controller.guidance_period_s):
                if k == 0:
                    # Gazebo publishes its first scan after the first physics step; at t=0 use the
                    # identical ray cast of the same boxes (checked against Gazebo by diagnose.py).
                    points, _, _ = virtual_lidar(scene, state.position_ned, state.euler_zyx)
                    scan_age = 0.0
                elif session.latest_scan is not None:
                    points = gazebo_cloud_to_matlab_order(session.latest_scan.xyz, mapper.lidar_flip_vertical,
                                                          mapper.lidar_flip_horizontal)
                    scan_age = t - session.latest_scan.time_s
            command = controller.step(state, available, points)
            controller_healthy &= command.healthy
            if "reconstruction_used" in command.record and _is_multiple(t, 0.1):
                # runMission's proof: every outage scan is reconstructed, no healthy scan is.
                availability_ok &= bool(command.record["reconstruction_used"]) == (not available)
            air = wind.at(t)
            drag = config.drag.force_ned(air, state.velocity_ned)
            p = state.position_ned
            clearance = float(signed_box_distance(p, scene)[0] - options.VehicleRadiusM)
            row = {
                "time_s": t, "n": p[0], "e": p[1], "d": p[2],
                "vn": state.velocity_ned[0], "ve": state.velocity_ned[1], "vd": state.velocity_ned[2],
                "yaw": state.euler_zyx[0], "pitch": state.euler_zyx[1], "roll": state.euler_zyx[2],
                "lidar_available": int(available), "scan_used": int(points is not None), "scan_age_s": scan_age,
                "accel_n": command.acceleration_ned[0], "accel_e": command.acceleration_ned[1],
                "accel_d": command.acceleration_ned[2], "yaw_rate_cmd": command.yaw_rate_ned,
                "goal": int(command.goal_reached), "controller_ok": int(command.healthy),
                "wind_n": air[0], "wind_e": air[1], "wind_d": air[2],
                "drag_n": drag[0], "drag_e": drag[1], "drag_d": drag[2], "clearance_m": clearance,
            }
            row.update(command.record)
            rows.append(row)
            if -p[2] < 0.2 or clearance <= 0.0 or abs(state.euler_zyx[1]) > 1.2 or abs(state.euler_zyx[2]) > 1.2:
                crashed = True
                crash_kind = ("hit-ground" if -p[2] < 0.2 else "hit-building" if clearance <= 0.0
                              else "lost-attitude")
                break
            if command.stop or k == ticks:
                goal_at_end = command.goal_reached
                break
            yaw_enu = math.pi / 2 - state.euler_zyx[0]
            setpoint = mapper.velocity_setpoint(state.velocity_ned, command.acceleration_ned, dt, yaw_enu)
            session.enable_controller()
            session.send_velocity(setpoint, -command.yaw_rate_ned)  # NED yaw rate -> FLU yaw rate
            session.set_force_enu(ned_to_enu(drag))
            next_t = round((k + 1) * dt, 9)
            scan_due = next_t if _is_multiple(next_t, 0.1) else None
            session.step(next_t, need_scan_at=scan_due)
    except Exception as exc:  # keep the evidence; the run is reported as a pipeline error
        error = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  ERROR: {error}")
    finally:
        session.close()
        controller.finish()
        if owns_controller:
            controller.close()
    wall = wallclock.monotonic() - started
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))

    # Native-rate samples for scoring (spawn pose first, then every odometry message).
    native_t = np.asarray([0.0] + [sample.time_s for sample in session.odometry])
    native_p = np.asarray([options.start_state[0:3]] + [ENU_NED @ sample.position_enu for sample in session.odometry])
    axis = evaluate_axis_run(native_t, native_p, options, scene)
    final_velocity_error = None
    if rows:
        final_velocity_error = float(np.linalg.norm([rows[-1]["vn"], rows[-1]["ve"], rows[-1]["vd"]]
                                                    - options.goal_state[3:6]))
    goal_at_rest = bool(goal_at_end and axis.get("goalReachedAtTermination")
                        and final_velocity_error is not None
                        and final_velocity_error <= options.GoalVelocityToleranceMps)
    success = bool(axis.get("missionSuccess") and goal_at_rest and controller_healthy and availability_ok
                   and not crashed and error is None)
    if error:
        reason = "pipeline-error"
    elif crashed:
        reason = f"crash ({crash_kind})"
    elif not controller_healthy:
        reason = "controller-failure"
    elif axis.get("collisionOccurred"):
        reason = "collision"
    elif axis.get("nearCollisionOccurred"):
        reason = "too-close-to-building"
    elif axis.get("altitudeWithinLimits") is False:
        reason = "altitude-out-of-bounds"
    elif goal_at_rest:
        reason = "goal-state-reached"
    elif axis.get("reachedTimeLimit"):
        reason = "time-limit-before-terminal-state"
    else:
        reason = axis.get("terminationReason", "early-stop")
    deviation = route_deviation(native_p, route)
    flags = [row["lidar_available"] for row in rows]
    observed = (int(flags[0] == 0) + sum(1 for a, b in zip(flags, flags[1:]) if a == 1 and b == 0)) if flags else 0
    used = [row.get("reconstruction_used", 0) for row in rows if _is_multiple(row["time_s"], 0.1)]
    summary = {
        "schema": "ares-gz-summary-v2", "run_id": run_id, "scenario": config.scenario, "seed": config.seed,
        "start_id": config.start_id, "start_ned": list(options.StartState[0:3]), "campaign": config.campaign,
        "controller": config.controller, "lidar_mode": options.LidarMode, "wind": wind.profile.name,
        "mission_success": success, "termination_reason": reason, "error": error,
        "goal_reached_at_rest": goal_at_rest, "crashed": crashed, "controller_healthy": controller_healthy,
        "outages_used_reconstruction": availability_ok,
        "arrival_time_s": rows[-1]["time_s"] if rows else None,
        "minimum_clearance_m": axis.get("minimumObstacleClearanceM"),
        "required_clearance_m": options.MinimumSurfaceClearanceM,
        "collision": axis.get("collisionOccurred"), "near_collision": axis.get("nearCollisionOccurred"),
        "final_goal_error_m": axis.get("finalWaypointErrorM"), "final_velocity_error_mps": final_velocity_error,
        "max_route_deviation_m": float(np.nanmax(deviation)) if deviation.size else None,
        "planned_outages": int(len(intervals)), "observed_outages": observed,
        "outage_ticks": int(sum(1 for row in rows if not row["lidar_available"])), "control_ticks": len(rows),
        "reconstruction_scans": int(sum(used)), "max_wind_mps": wind.describe()["max_speed_mps"],
        "max_drag_n": max((math.hypot(row["drag_n"], row["drag_e"], row["drag_d"]) for row in rows), default=0.0),
        "wall_time_s": wall, "scans_received": session.scan_count, "missed_scans": session.missed_scans,
        "step_retries": session.step_retries, "axis_metrics": axis,
        "directory": str(directory),
    }
    if rows:
        columns: list[str] = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)
        with (directory / "control.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    np.savetxt(directory / "odometry_ned.csv", np.column_stack((native_t, native_p)), delimiter=",",
               header="time_s,n,e,d", comments="", fmt="%.6f")
    (directory / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if verbose:
        clearance = summary["minimum_clearance_m"]
        print(f"  -> success={success} ({reason}), end {summary['arrival_time_s']} s, "
              f"min clearance {clearance if clearance is None else round(clearance, 3)} m, "
              f"reconstructed scans {summary['reconstruction_scans']}, wall {wall:.0f} s")
    return summary
