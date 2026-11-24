"""Fly one scenario in Gazebo with the ported map-correction controller and score it.

Per control tick (20 Hz, lock-step):
  1. read the drone state (Gazebo odometry -> NED) and the latest lidar scan;
  2. lidar fault injection: during an outage the scan is withheld and the
     controller is told the lidar is unavailable (as in the MATLAB model);
  3. the controller returns a safe NED acceleration and a yaw rate;
  4. that acceleration becomes a velocity set-point for Gazebo's multicopter
     velocity controller (see ``CommandMapper``);
  5. wind: the drag force from (air velocity - drone velocity) is applied to the body;
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

from .frames import ENU_NED, gazebo_orientation_to_ned_euler, ned_to_enu
from .gazebo import GazeboSession
from .metrics import evaluate_axis_run, mission_success, route_deviation, signed_box_distance
from .navigation import MapCorrectionController, availability_at, outage_schedule, plan_path, validate_options
from .navigation.options import NavigationOptions
from .navigation.lidar import gazebo_cloud_to_matlab_order
from .paths import RESULTS_ROOT
from .scene import SceneMap, scene_map
from .wind import PRESETS, DragModel, WindField, WindProfile
from .world import QuadParameters, world_sdf

CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "config" / "vehicle_calibration.json"


@dataclass(frozen=True)
class CommandMapper:
    """Turns the controller's acceleration into a velocity set-point for Gazebo's controller.

    Gazebo's MulticopterVelocityControl commands an acceleration of about
    ``gain * (v_cmd - v)``. Choosing ``v_cmd = v + u * dt / (1 - exp(-gain*dt))``
    makes the average acceleration over one tick equal the requested ``u``.
    ``gain`` and ``frame`` are measured by ``scripts/diagnose.py``.
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
    overrides: dict = field(default_factory=dict)  # extra NavigationOptions fields (StartState, GoalState, ...)
    gui: bool = False
    headless_rendering: bool = True
    campaign: str = "adhoc"
    drag: DragModel = field(default_factory=DragModel)

    def wind_profile(self) -> WindProfile:
        return self.wind if isinstance(self.wind, WindProfile) else PRESETS[self.wind]

    def options(self) -> NavigationOptions:
        values = dict(self.overrides)
        values.update(LidarMode=self.lidar_mode, Seed=int(self.seed))
        return validate_options(values)


def _plan_cached(scene: SceneMap, options: NavigationOptions):
    """Route planning takes ~10 s in Python; cache it by everything it depends on."""
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


def _state(samples) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """NED position, finite-difference NED velocity and NED [yaw pitch roll] from the latest odometry."""
    last = samples[-1]
    position = ENU_NED @ last.position_enu
    velocity = np.zeros(3)
    if len(samples) >= 2:
        previous = samples[-2]
        dt = last.time_s - previous.time_s
        if dt > 0:
            velocity = ENU_NED @ (last.position_enu - previous.position_enu) / dt
    euler = gazebo_orientation_to_ned_euler(*last.orientation_wxyz)
    return last.time_s, position, velocity, euler


def run_scenario(config: RunConfig, output_root: Path | None = None, verbose: bool = True) -> dict:
    options = config.options()
    scene = scene_map()
    route, plan_dict = _plan_cached(scene, options)
    times, enabled, intervals = outage_schedule(options)
    wind = WindField(config.wind_profile(), config.seed, options.StopTimeS, options.ControlSampleTimeS)
    mapper = CommandMapper.load()
    quad = QuadParameters()
    run_id = str(uuid.uuid4())
    directory = (output_root or RESULTS_ROOT / config.campaign) / config.scenario / f"seed{config.seed:04d}_{run_id[:8]}"
    directory.mkdir(parents=True, exist_ok=True)
    world_file = directory / "world.sdf"
    world_file.write_text(world_sdf(scene, options.start_state[0:3], options.StartEulerZYXRad))
    manifest = {
        "schema": "ares-gz-run-v1", "run_id": run_id, "scenario": config.scenario, "campaign": config.campaign,
        "seed": config.seed, "lidar_mode": options.LidarMode, "options": options.to_dict(),
        "route_ned": np.asarray(route).tolist(), "plan": plan_dict, "outage_intervals_s": intervals.tolist(),
        "wind": wind.describe(), "drag": asdict(config.drag), "command_mapper": asdict(mapper),
        "vehicle": asdict(quad), "controller": "python port of src/+ares/+navigation (parity-tested)",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f"[{config.scenario} seed {config.seed}] route {plan_dict['LengthM']:.1f} m, "
              f"{len(intervals)} outage(s), wind {wind.profile.name} -> {directory}")

    controller = MapCorrectionController(scene, options, route)
    dt = options.ControlSampleTimeS
    ticks = int(round(options.StopTimeS / dt))
    rows: list[dict] = []
    feasible_all = True
    availability_ok = True
    crashed = False
    goal_at_end = False
    error = None
    started = wallclock.monotonic()
    session = GazeboSession(world_file, directory / "logs", gui=config.gui,
                            headless_rendering=config.headless_rendering,
                            steps_per_tick=int(round(dt / 0.001)))
    try:
        session.start()
        start = options.start_state
        position, velocity, euler = start[0:3].copy(), start[3:6].copy(), np.asarray(options.StartEulerZYXRad, float)
        for k in range(ticks + 1):
            t = k * dt
            if k > 0:
                _, position, velocity, euler = _state(session.odometry)
            available = availability_at(times, enabled, t)
            scan = session.latest_scan
            points = None
            scan_age = None
            if available and scan is not None:
                points = gazebo_cloud_to_matlab_order(scan.xyz, mapper.lidar_flip_vertical,
                                                      mapper.lidar_flip_horizontal)
                scan_age = t - scan.time_s
            out = controller.step(points, position, velocity, euler, available, t)
            feasible_all &= out.info.Feasible
            availability_ok &= (out.lidar_available == available)
            air = wind.at(t)
            drag = config.drag.force_ned(air, velocity)
            clearance = float(signed_box_distance(position, scene)[0] - options.VehicleRadiusM)
            rows.append({
                "time_s": t, "n": position[0], "e": position[1], "d": position[2],
                "vn": velocity[0], "ve": velocity[1], "vd": velocity[2],
                "yaw": euler[0], "pitch": euler[1], "roll": euler[2],
                "lidar_available": int(available), "scan_used": int(points is not None),
                "scan_age_s": scan_age, "measured_points": out.measured_point_count,
                "accel_n": out.acceleration_ned[0], "accel_e": out.acceleration_ned[1], "accel_d": out.acceleration_ned[2],
                "nominal_n": out.nominal_ned[0], "nominal_e": out.nominal_ned[1], "nominal_d": out.nominal_ned[2],
                "ref_n": out.reference_ned[0], "ref_e": out.reference_ned[1], "ref_d": out.reference_ned[2],
                "yaw_rate_cmd": out.yaw_rate, "filter_feasible": int(out.info.Feasible),
                "minimum_h": out.info.MinimumH, "minimum_psi1": out.info.MinimumPsi1,
                "waypoint_index": out.waypoint_index, "goal": int(out.goal_reached),
                "wind_n": air[0], "wind_e": air[1], "wind_d": air[2],
                "drag_n": drag[0], "drag_e": drag[1], "drag_d": drag[2],
                "clearance_m": clearance, "compute_s": out.compute_seconds,
            })
            if -position[2] < 0.2 or clearance <= 0.0 or abs(euler[1]) > 1.2 or abs(euler[2]) > 1.2:
                crashed = True
                break
            if out.stop or k == ticks:
                goal_at_end = out.goal_reached
                break
            yaw_enu = math.pi / 2 - euler[0]
            setpoint = mapper.velocity_setpoint(velocity, out.acceleration_ned, dt, yaw_enu)
            session.enable_controller()
            session.send_velocity(setpoint, -out.yaw_rate)  # NED yaw rate -> FLU yaw rate
            session.set_force_enu(ned_to_enu(drag))
            next_t = (k + 1) * dt
            scan_due = math.floor(next_t / 0.1 + 1e-7) * 0.1
            session.step(next_t, need_scan_at=scan_due)
    except Exception as exc:  # keep the evidence; the run is reported as an error
        error = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  ERROR: {error}")
    finally:
        session.close()
    wall = wallclock.monotonic() - started

    # Native-rate samples for scoring (spawn pose first, then every odometry message).
    native_t = [0.0] + [sample.time_s for sample in session.odometry]
    native_p = [options.start_state[0:3]] + [ENU_NED @ sample.position_enu for sample in session.odometry]
    native_t = np.asarray(native_t)
    native_p = np.asarray(native_p)
    axis = evaluate_axis_run(native_t, native_p, options, scene)
    success, reason = mission_success(axis, goal_at_end, feasible_all, availability_ok, crashed)
    if error:
        success, reason = False, "pipeline-error"
    deviation = route_deviation(native_p, route) if native_p.shape[0] else np.array([np.nan])
    observed = 0
    if rows:
        flags = [row["lidar_available"] for row in rows]
        observed = int(flags[0] == 0) + sum(1 for a, b in zip(flags, flags[1:]) if a == 1 and b == 0)
    summary = {
        "schema": "ares-gz-summary-v1", "run_id": run_id, "scenario": config.scenario, "seed": config.seed,
        "lidar_mode": options.LidarMode, "wind": wind.profile.name,
        "mission_success": success, "termination_reason": reason, "error": error,
        "goal_reached_at_rest": goal_at_end, "crashed": crashed, "all_filters_feasible": feasible_all,
        "availability_matches_schedule": availability_ok,
        "arrival_time_s": rows[-1]["time_s"] if rows else None,
        "minimum_clearance_m": axis.get("minimumObstacleClearanceM"),
        "required_clearance_m": options.MinimumSurfaceClearanceM,
        "collision": axis.get("collisionOccurred"), "near_collision": axis.get("nearCollisionOccurred"),
        "max_route_deviation_m": float(np.nanmax(deviation)) if deviation.size else None,
        "mean_route_deviation_m": float(np.nanmean(deviation)) if deviation.size else None,
        "planned_outages": int(len(intervals)), "observed_outages": observed,
        "outage_ticks": int(sum(1 for row in rows if not row["lidar_available"])), "control_ticks": len(rows),
        "max_wind_mps": wind.describe()["max_speed_mps"],
        "max_drag_n": max((math.hypot(row["drag_n"], row["drag_e"], row["drag_d"]) for row in rows), default=0.0),
        "minimum_barrier_h": min((row["minimum_h"] for row in rows), default=None),
        "max_compute_s": max((row["compute_s"] for row in rows), default=None),
        "wall_time_s": wall, "scans_received": session.scan_count,
        "axis_metrics": axis, "directory": str(directory),
    }
    if rows:
        with (directory / "control.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savetxt(directory / "odometry_ned.csv", np.column_stack((native_t, native_p)), delimiter=",",
               header="time_s,n,e,d", comments="", fmt="%.6f")
    (directory / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if verbose:
        clearance = summary["minimum_clearance_m"]
        print(f"  -> success={success} ({reason}), arrival {summary['arrival_time_s']} s, "
              f"min clearance {clearance if clearance is None else round(clearance, 3)} m, "
              f"max route deviation {summary['max_route_deviation_m']:.2f} m, wall {wall:.0f} s")
    return summary
