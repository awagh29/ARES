#!/usr/bin/env python3
"""Smoke-test the MATLAB guidance bridge WITHOUT Gazebo.

A point-mass drone integrates the PID acceleration command; its lidar is the
map ray cast. This checks MATLAB start-up from WSL, the socket link, Atharva's
reconstruction, the UAV Toolbox follower/VFH blocks and the PID cascade
together. It is not a physics test.

    python3 scripts/bridge_smoke.py [--mode healthy|intermittent|blackout] [--seconds 25] [--start W1]
                                    [--guidance planner|governor|vfh]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ares_gz.controllers.base import VehicleState  # noqa: E402
from ares_gz.controllers.map_temporal import MapTemporalController, TRAINED_BOUNDS_NED  # noqa: E402
from ares_gz.matlab_bridge import MatlabBridge  # noqa: E402
from ares_gz.metrics import signed_box_distance  # noqa: E402
from ares_gz.model_artifact import resolve_model  # noqa: E402
from ares_gz.navigation import availability_at, outage_schedule, validate_options, virtual_lidar  # noqa: E402
from ares_gz.scenarios import load_matrix  # noqa: E402
from ares_gz.scene import scene_map  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="healthy", choices=["healthy", "intermittent", "blackout"])
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=513)
    parser.add_argument("--manual-matlab", action="store_true")
    parser.add_argument("--guidance", choices=["planner", "governor", "vfh"], help="default from the YAML")
    parser.add_argument("--start", help="start point id from the YAML (e.g. W1); default mission.StartState")
    args = parser.parse_args()

    matrix = load_matrix()
    model, info = resolve_model(matrix.controller["model_path"])
    print("model:", info)
    mission = dict(matrix.mission)
    if args.start:
        start = {s.id: s for s in matrix.starts}[args.start]
        mission["StartState"] = start.state(float(mission["StartState"][2]))
    options = validate_options(dict(mission, LidarMode=args.mode, Seed=args.seed,
                                     WorldBoundsNedM=TRAINED_BOUNDS_NED))
    guidance = args.guidance or matrix.controller.get("guidance", "planner")
    times, enabled, intervals = outage_schedule(options)
    print("outages (s):", intervals.tolist())
    scene = scene_map()
    started = time.monotonic()
    with MatlabBridge(launch=not args.manual_matlab) as bridge:
        print(f"MATLAB connected after {time.monotonic() - started:.0f} s")
        controller = MapTemporalController(bridge, model, guidance=guidance)
        print("init:", controller.start(options))
        dt = controller.tick_s
        p = options.start_state[0:3].copy()
        v = np.zeros(3)
        yaw = 0.0
        minimum = np.inf
        timing = {"raycast": 0.0, "round_trip": 0.0, "matlab": 0.0, "calls": 0}
        for k in range(int(round(args.seconds / dt)) + 1):
            t = round(k * dt, 9)
            available = availability_at(times, enabled, t)
            points = None
            guidance_tick = abs(t / 0.1 - round(t / 0.1)) < 1e-6
            if available and guidance_tick:
                tick = time.perf_counter()
                points, _, _ = virtual_lidar(scene, p, [yaw, 0.0, 0.0])
                timing["raycast"] += time.perf_counter() - tick
            state = VehicleState(t, p.copy(), v.copy(), np.array([yaw, 0.0, 0.0]), np.zeros(3))
            tick = time.perf_counter()
            command = controller.step(state, available, points)
            if guidance_tick:
                timing["round_trip"] += time.perf_counter() - tick
                timing["matlab"] += float(controller.guidance.get("bridge_s") or 0.0)
                timing["recon"] = timing.get("recon", 0.0) + float(controller.guidance.get("reconstruction_block_s") or 0)
                timing["vfh"] = timing.get("vfh", 0.0) + float(controller.guidance.get("avoidance_s") or 0)
                timing["planner"] = timing.get("planner", 0.0) + float(controller.guidance.get("governor_s") or 0)
                timing["calls"] += 1
            minimum = min(minimum, float(signed_box_distance(p, scene)[0]) - 0.1)
            if abs(t - round(t)) < 1e-9:
                record = command.record
                print(f"t={t:5.1f}  pos=({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f})  lidar={int(available)}  "
                      f"cnn={record.get('reconstruction_used')}  infer={record.get('inference_s') or 0:.2f}s  "
                      f"ref=({record['ref_n']:.1f},{record['ref_e']:.1f})  clearance_min={minimum:.2f}")
            if command.stop:
                print(f"terminal state reached at t={t:.2f} s")
                break
            v = v + command.acceleration_ned * dt
            p = p + v * dt
            yaw = yaw + command.yaw_rate_ned * dt
        controller.finish()
    calls = max(timing["calls"], 1)
    print(f"per guidance call: round trip {timing['round_trip'] / calls * 1000:.0f} ms "
          f"(MATLAB compute {timing['matlab'] / calls * 1000:.0f} ms: reconstruction block "
          f"{timing.get('recon', 0) / calls * 1000:.0f} ms, VFH {timing.get('vfh', 0) / calls * 1000:.0f} ms, "
          f"planner+governor {timing.get('planner', 0) / calls * 1000:.0f} ms), Python ray cast "
          f"{timing['raycast'] / calls * 1000:.0f} ms, {timing['calls']} calls")
    print(f"final position {np.round(p, 2).tolist()}, goal {options.goal_state[0:3].tolist()}, "
          f"min surface clearance {minimum:.2f} m, wall {time.monotonic() - started:.0f} s")
    print("controller summary:", controller.summary_extras())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
