"""Fly one three-drone search-and-rescue mission in Gazebo with the hub in the loop.

Every 0.01 s tick (lock-step, like the single-drone runner):
  * each drone's state comes from its own Gazebo odometry;
  * every 0.1 s: LiDAR fault injection per drone, victim camera, each drone's
    MATLAB guidance (Atharva's CNN + planner + braking governor) runs in
    parallel, retargeting first if the hub gave it a new goal; each drone
    sends an uplink; the hub ticks and sends downlinks (through the simulated
    radio links);
  * each drone's nominal velocity passes its CBF share (``agent.py``) and the
    velocity PID; Gazebo advances one tick for all drones together.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import time as wallclock
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..controllers import VehicleState
from ..controllers.map_temporal import TRAINED_BOUNDS_NED, MapTemporalController
from ..frames import ENU_NED, ned_to_enu
from ..gazebo import GazeboSession
from ..metrics import signed_box_distance
from ..navigation import availability_at, outage_schedule, validate_options, virtual_lidar
from ..navigation.lidar import gazebo_cloud_to_matlab_order
from ..paths import RESULTS_ROOT
from ..run import PHYSICS_STEP_S, CommandMapper, vehicle_state
from ..scene import scene_map
from ..wind import DragModel, WindField, WindProfile
from ..world import QuadParameters, world_sdf
from .agent import DroneAgent
from .comms import PROFILES, CommsProfile, Link
from .hub import Hub, HubConfig, Uplink
from .mapping import ring_from_scan
from .trust import QUARANTINED
from .victims import Victim, VictimDetector


@dataclass(frozen=True)
class DroneSpec:
    id: str
    north: float
    east: float
    slot: tuple

    @property
    def model(self) -> str:
        return self.id.lower()


@dataclass
class SwarmConfig:
    scenario: str
    lidar_mode: str  # healthy | intermittent | blackout
    wind: WindProfile
    seed: int
    drones: list
    victims: list
    faulty: list | str = "all"  # drones whose LiDAR follows lidar_mode; others stay healthy
    outage_sync: str = "independent"  # independent | simultaneous
    comms: CommsProfile = field(default_factory=lambda: PROFILES["nominal"])
    hub: HubConfig = field(default_factory=HubConfig)
    mission: dict = field(default_factory=dict)  # NavigationOptions overrides shared by all drones
    campaign: str = "swarm"
    lidar_source: str = "gazebo"  # gazebo (GPU LiDAR) | raycast (same buildings + other drones, no rendering)
    gui: bool = False
    headless_rendering: bool = True
    drag: DragModel = field(default_factory=DragModel)

    def drone_options(self, spec: DroneSpec, index: int):
        faulty = self.faulty == "all" or spec.id in self.faulty
        mode = self.lidar_mode if faulty else "healthy"
        seed = self.seed if self.outage_sync == "simultaneous" else self.seed * 100 + index + 1
        altitude = float(self.hub.altitude_down)
        values = dict(self.mission)
        values.update(StartState=[spec.north, spec.east, altitude, 0.0, 0.0, 0.0],
                      GoalState=[float(spec.slot[0]), float(spec.slot[1]), altitude, 0.0, 0.0, 0.0],
                      LidarMode=mode, Seed=int(seed))
        values.setdefault("WorldBoundsNedM", TRAINED_BOUNDS_NED)
        return validate_options(values)


def with_drones(scene, positions, size=(0.47, 0.47, 0.11)):
    """The scene plus the other drones as small boxes, so a ray-cast LiDAR sees its neighbours."""
    from dataclasses import replace as dc_replace
    if not positions:
        return scene
    half = np.array(size) / 2.0
    centers = np.asarray(positions, float).reshape(-1, 3)
    return dc_replace(scene, box_min=np.vstack((scene.box_min, centers - half)),
                      box_max=np.vstack((scene.box_max, centers + half)))


def _pairwise_min(positions: dict) -> tuple[float, tuple]:
    best, pair = np.inf, None
    for a, b in itertools.combinations(sorted(positions), 2):
        distance = float(np.linalg.norm(positions[a] - positions[b]))
        if distance < best:
            best, pair = distance, (a, b)
    return best, pair


def run_swarm(config: SwarmConfig, controllers: dict, output_root: Path | None = None, verbose: bool = True) -> dict:
    scene = scene_map()
    specs = {spec.id: spec for spec in config.drones}
    ids = list(specs)
    options = {d: config.drone_options(specs[d], k) for k, d in enumerate(ids)}
    schedules = {d: outage_schedule(options[d]) for d in ids}
    dt = 0.01
    stop_s = float(options[ids[0]].StopTimeS)
    wind = WindField(config.wind, config.seed, stop_s, dt)
    mapper = CommandMapper.load()
    run_id = str(uuid.uuid4())
    directory = (output_root or RESULTS_ROOT / config.campaign) / config.scenario / f"seed{config.seed:04d}_{run_id[:8]}"
    directory.mkdir(parents=True, exist_ok=True)
    world_file = directory / "world.sdf"
    world_file.write_text(world_sdf(scene, None, drones=[(specs[d].model, options[d].start_state[0:3], (0.0, 0.0, 0.0))
                                                         for d in ids],
                                    markers=[(f"victim_{v.id}", v.north, v.east) for v in config.victims],
                                    lidar=config.lidar_source == "gazebo"))
    hub = Hub(ids, {d: specs[d].slot for d in ids}, scene, config.hub)
    agents = {d: DroneAgent(d) for d in ids}
    uplinks = {d: Link(config.comms, config.seed * 1000 + 10 + k) for k, d in enumerate(ids)}
    downlinks = {d: Link(config.comms, config.seed * 1000 + 20 + k) for k, d in enumerate(ids)}
    detector = VictimDetector(config.victims, scene, seed=config.seed)
    manifest = {
        "schema": "ares-gz-swarm-run-v1", "run_id": run_id, "lidar_source": config.lidar_source, "scenario": config.scenario, "campaign": config.campaign,
        "seed": config.seed, "lidar_mode": config.lidar_mode, "faulty": config.faulty, "outage_sync": config.outage_sync,
        "drones": [asdict(spec) for spec in config.drones], "victims": [asdict(v) for v in config.victims],
        "hub": asdict(config.hub), "comms": asdict(config.comms), "wind": wind.describe(),
        "outage_intervals_s": {d: schedules[d][2].tolist() for d in ids},
        "options": {d: options[d].to_dict() for d in ids}, "command_mapper": asdict(mapper),
        "vehicle": asdict(QuadParameters()),
    }
    if verbose:
        faults = {d: len(schedules[d][2]) for d in ids}
        print(f"[swarm {config.scenario} seed {config.seed}] lidar {config.lidar_mode} on {config.faulty} "
              f"({config.outage_sync}), outages {faults}, wind {config.wind.name}, comms {config.comms.name} -> {directory}")

    rows = {d: [] for d in ids}
    crashed, crash_info, error = None, None, None
    retarget_count = {d: 0 for d in ids}
    min_pair, min_pair_info = np.inf, None
    session = GazeboSession(world_file, directory / "logs", gui=config.gui, headless_rendering=config.headless_rendering,
                            steps_per_tick=int(round(dt / PHYSICS_STEP_S)), models=[specs[d].model for d in ids])
    pool = ThreadPoolExecutor(max_workers=len(ids))
    started = wallclock.monotonic()
    ticks = int(round(stop_s / dt))
    end_reason = "time-limit"
    try:
        list(pool.map(lambda d: controllers[d].start(options[d]), ids))  # MATLAB inits overlap
        for d in ids:
            controllers[d].velocity_filter = (lambda t, state, nominal, agent=agents[d]:
                                              agent.filter(t, state.position_ned, nominal))
        session.start()
        states = {d: VehicleState(0.0, options[d].start_state[0:3].copy(), np.zeros(3), np.zeros(3), np.zeros(3))
                  for d in ids}
        seq = {d: 0 for d in ids}
        for k in range(ticks + 1):
            t = round(k * dt, 9)
            if k > 0:
                for d in ids:
                    s = vehicle_state(session.odometry_by[specs[d].model], t)
                    states[d] = VehicleState(t, s.position_ned, s.velocity_ned, s.euler_zyx, s.body_rates_frd)
            available = {d: availability_at(schedules[d][0], schedules[d][1], t) for d in ids}
            guidance_tick = controllers[ids[0]].is_guidance_tick(t)
            if guidance_tick:
                points = {}
                for d in ids:
                    points[d] = None
                    if not available[d]:
                        continue
                    if k == 0 or config.lidar_source == "raycast":
                        others = [states[o].position_ned for o in ids if o != d]
                        points[d], _, _ = virtual_lidar(with_drones(scene, others), states[d].position_ned,
                                                        states[d].euler_zyx)
                    elif session.scan_by[specs[d].model] is not None:
                        points[d] = gazebo_cloud_to_matlab_order(session.scan_by[specs[d].model].xyz,
                                                                 mapper.lidar_flip_vertical, mapper.lidar_flip_horizontal)

                def guide(d):
                    rejected = None
                    goal = agents[d].needs_retarget(t)
                    if goal is not None:
                        if controllers[d].retarget(states[d], goal):
                            agents[d].goal_ned = goal
                            retarget_count[d] += 1
                        else:
                            rejected = agents[d].downlink.task_id
                            agents[d].rejected_goal = goal  # hold; do not retry the same target every tick
                    ok = controllers[d].update_guidance(states[d], available[d], points[d])
                    return d, ok, rejected

                results = list(pool.map(guide, ids))
                for d, ok, rejected in results:
                    state = states[d]
                    seq[d] += 1
                    ring = None
                    if available[d] and points[d] is not None:
                        ring = ring_from_scan(points[d], state.position_ned, state.euler_zyx, scene)
                    sightings = detector.detect(d, state.position_ned, t)
                    uplinks[d].send(t, Uplink(
                        d, t, seq[d], state.position_ned.copy(), state.velocity_ned.copy(), bool(available[d]),
                        bool(controllers[d].guidance.get("reconstruction_used")), ring, True, sightings,
                        agents[d].downlink.task_id if agents[d].downlink else None, rejected,
                        bool(controllers[d].terminal)))
                for d in ids:
                    for message in uplinks[d].receive(t):
                        hub.receive(message)
                for message in hub.tick(t):
                    downlinks[message.drone].send(t, message)
                if verbose and k % 1000 == 0 and k > 0:
                    modes = " ".join(f"{d}:{hub.drones[d].mode[:4]}" for d in ids)
                    print(f"  t={t:5.0f} s  wall {wallclock.monotonic() - started:5.0f} s  {hub.phase}  {modes}  "
                          f"search {hub.coverage_log[-1][1] if hub.coverage_log else 0:.0%}  "
                          f"victims {len(hub.victims)}  min sep {min_pair:.2f} m", flush=True)
            for d in ids:
                agents[d].receive(downlinks[d].receive(t))
            positions = {d: states[d].position_ned for d in ids}
            distance, pair = _pairwise_min(positions)
            if distance < min_pair:
                min_pair, min_pair_info = distance, {"time_s": t, "pair": pair}
            air = wind.at(t)
            for d in ids:
                state = states[d]
                command = controllers[d].control(state)
                drag = config.drag.force_ned(air, state.velocity_ned)
                p = state.position_ned
                clearance = float(signed_box_distance(p, scene)[0] - options[d].VehicleRadiusM)
                downlink = agents[d].downlink
                row = {"time_s": t, "n": p[0], "e": p[1], "d": p[2], "vn": state.velocity_ned[0],
                       "ve": state.velocity_ned[1], "vd": state.velocity_ned[2], "yaw": state.euler_zyx[0],
                       "pitch": state.euler_zyx[1], "roll": state.euler_zyx[2], "lidar_available": int(available[d]),
                       "clearance_m": clearance, "wind_n": air[0], "wind_e": air[1], "wind_d": air[2],
                       "agent_mode": agents[d].mode(t), "hub_mode": downlink.mode if downlink else "",
                       "task": downlink.task_id if downlink else "", "task_kind": downlink.task_kind if downlink else "",
                       "speed_cap": downlink.speed_cap if downlink else 0.0, "min_pair_m": distance,
                       **{f"cbf_{key}": value for key, value in agents[d].last_filter.items()},
                       "goal_n": agents[d].goal_ned[0] if agents[d].goal_ned is not None else "",
                       "goal_e": agents[d].goal_ned[1] if agents[d].goal_ned is not None else ""}
                row.update(command.record)
                rows[d].append(row)
                if -p[2] < 0.2 or clearance <= 0.0 or abs(state.euler_zyx[1]) > 1.2 or abs(state.euler_zyx[2]) > 1.2:
                    crashed = d
                    crash_info = ("hit-ground" if -p[2] < 0.2 else "hit-building" if clearance <= 0.0 else "lost-attitude")
                if not command.healthy:
                    error = error or f"{d}: controller failure {controllers[d].last_error}"
                yaw_enu = math.pi / 2 - state.euler_zyx[0]
                setpoint = mapper.velocity_setpoint(state.velocity_ned, command.acceleration_ned, dt, yaw_enu)
                session.enable_controller(specs[d].model)
                session.send_velocity(setpoint, -command.yaw_rate_ned, specs[d].model)
                session.set_force_enu(ned_to_enu(drag), specs[d].model)
            if crashed:
                end_reason = f"crash ({crashed}: {crash_info})"
                break
            if hub.phase == "RENDEZVOUS" and all(r.arrived for r in hub.drones.values() if r.mode != "LOST"):
                end_reason = "all-drones-at-rendezvous"
                break
            if k == ticks:
                break
            next_t = round((k + 1) * dt, 9)
            scan_due = config.lidar_source == "gazebo" and controllers[ids[0]].is_guidance_tick(next_t)
            session.step(next_t, need_scan_at=next_t if scan_due else None)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        end_reason = "pipeline-error"
        if verbose:
            print(f"  ERROR: {error}")
    finally:
        pool.shutdown(wait=True)
        session.close()
        for d in ids:
            controllers[d].velocity_filter = None
            controllers[d].finish()
    wall = wallclock.monotonic() - started

    # Native-rate (1 kHz) pairwise separation and building clearance.
    native = {}
    for d in ids:
        samples = session.odometry_by[specs[d].model]
        native[d] = {round(s.time_s, 4): ENU_NED @ s.position_enu for s in samples}
    common = sorted(set.intersection(*(set(v) for v in native.values()))) if native else []
    native_pair = np.inf
    for time_key in common:
        distance, _ = _pairwise_min({d: native[d][time_key] for d in ids})
        native_pair = min(native_pair, distance)
    per_drone = {}
    for d in ids:
        positions = np.array(list(native[d].values())) if native[d] else np.zeros((0, 3))
        clearance = float(np.min(signed_box_distance(positions, scene)) - options[d].VehicleRadiusM) if len(positions) else None
        record = hub.drones[d]
        outages = [row["lidar_available"] for row in rows[d]]
        per_drone[d] = {
            "min_building_clearance_m": clearance, "final_mode": record.mode, "arrived": record.arrived,
            "failure_episodes": record.trust.episodes, "final_trust": record.trust.last.trust if record.trust.last else None,
            "outage_ticks": int(sum(1 for a in outages if not a)), "retargets": retarget_count[d],
            "comms_loss_ticks": agents[d].comms_loss_ticks, **controllers[d].summary_extras(),
            "reconstruction_scans": int(controllers[d].reconstruction_ticks),
            "uplink": uplinks[d].stats(), "downlink": downlinks[d].stats(),
        }
    flag_latency = {}
    for d in ids:
        intervals = schedules[d][2]
        if len(intervals):
            onset = float(intervals[0][0])
            flagged = next((e["time_s"] for e in hub.events if e["drone"] == d and "-> DEGRADED" in e["event"]
                            and e["time_s"] >= onset - 1e-9), None)
            contained = next((e["time_s"] for e in hub.events if e["drone"] == d and "-> QUARANTINED" in e["event"]
                              and e["time_s"] >= onset - 1e-9), None)
            flag_latency[d] = {"first_outage_s": onset, "flagged_after_s": None if flagged is None else round(flagged - onset, 2),
                               "quarantined_after_s": None if contained is None else round(contained - onset, 2)}
    hub_summary = hub.summary()
    truth = {v.id for v in config.victims}
    found = {v["truth_id"] for v in hub_summary["victims"]}
    confirmed = {v["truth_id"] for v in hub_summary["victims"] if v["status"] == "CONFIRMED"}
    separation = config.hub.separation_m
    invariant_held = bool(min(native_pair, min_pair) >= separation - 1e-6)
    clearance_ok = all((per_drone[d]["min_building_clearance_m"] or 0) >= options[d].MinimumSurfaceClearanceM
                       for d in ids if per_drone[d]["final_mode"] != QUARANTINED)
    active_arrived = all(per_drone[d]["arrived"] for d in ids if per_drone[d]["final_mode"] not in (QUARANTINED, "LOST"))
    success = bool(error is None and not crashed and invariant_held and clearance_ok and active_arrived
                   and confirmed == truth)
    reasons = []
    if error:
        reasons.append("pipeline-error")
    if crashed:
        reasons.append(f"crash ({crashed}: {crash_info})")
    if not invariant_held:
        reasons.append(f"separation {min(native_pair, min_pair):.2f} m < {separation} m")
    if not clearance_ok:
        reasons.append("too-close-to-building")
    if not active_arrived:
        reasons.append("not-all-at-rendezvous")
    if confirmed != truth:
        reasons.append(f"victims confirmed {len(confirmed)}/{len(truth)}")
    summary = {
        "schema": "ares-gz-swarm-summary-v1", "run_id": run_id, "scenario": config.scenario, "seed": config.seed,
        "campaign": config.campaign, "lidar_mode": config.lidar_mode, "faulty": config.faulty,
        "outage_sync": config.outage_sync, "wind": config.wind.name, "comms": config.comms.name,
        "mission_success": success, "termination_reason": "; ".join(reasons) or end_reason, "end_reason": end_reason,
        "error": error, "end_time_s": rows[ids[0]][-1]["time_s"] if rows[ids[0]] else None,
        "min_pairwise_distance_m": float(min(native_pair, min_pair)), "separation_required_m": separation,
        "separation_invariant_held": invariant_held, "closest_pair": min_pair_info,
        "coverage": hub_summary["coverage"], "victims_total": len(truth), "victims_found": len(found & truth),
        "victims_confirmed": len(confirmed & truth), "failure_detection": flag_latency,
        "drones": per_drone, "hub": hub_summary, "wall_time_s": wall, "missed_scans": session.missed_scans,
        "step_retries": session.step_retries, "directory": str(directory),
    }
    for d in ids:
        if rows[d]:
            drone_dir = directory / "drones" / d
            drone_dir.mkdir(parents=True, exist_ok=True)
            columns: list[str] = []
            for row in rows[d]:
                columns.extend(key for key in row if key not in columns)
            with (drone_dir / "control.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows[d])
    if hub.trust_log:
        with (directory / "trust.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(hub.trust_log[0]))
            writer.writeheader()
            writer.writerows(hub.trust_log)
    (directory / "hub_events.json").write_text(json.dumps(hub.events, indent=1, default=float))
    np.save(directory / "shared_map_logodds.npy", hub.shared.logodds())
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    (directory / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if verbose:
        print(f"  -> success={success} ({summary['termination_reason']}), end {summary['end_time_s']} s, "
              f"min separation {summary['min_pairwise_distance_m']:.2f} m (needs {separation}), "
              f"coverage {summary['coverage']:.0%}, victims {summary['victims_confirmed']}/{len(truth)} confirmed, "
              f"wall {wall:.0f} s")
    return summary


def build_controllers(ids: list[str], model_path: str, export_python: str, guidance: str, port: int,
                      manual_matlab: bool, log_root: Path) -> tuple[dict, list]:
    """One MATLAB guidance session per drone (they run in parallel)."""
    from ..matlab_bridge import MatlabBridge

    def start(k_d):
        k, d = k_d
        return MatlabBridge(port=port + k, launch=not manual_matlab, log_file=log_root / f"matlab_bridge_{d}.log")

    with ThreadPoolExecutor(max_workers=len(ids)) as pool:  # MATLAB start-ups overlap
        bridges = list(pool.map(start, enumerate(ids)))
    controllers = {d: MapTemporalController(bridge, model_path, export_python, guidance)
                   for d, bridge in zip(ids, bridges)}
    return controllers, bridges


def victims_from(items) -> list[Victim]:
    return [Victim(str(item["id"]), float(item["north"]), float(item["east"])) for item in items or []]
