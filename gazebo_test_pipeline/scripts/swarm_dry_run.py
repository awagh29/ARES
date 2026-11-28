#!/usr/bin/env python3
"""Exercise the swarm hub WITHOUT Gazebo or MATLAB (seconds, not minutes).

Point-mass drones fly a simple go-to-goal + building-repulsion stand-in for the
onboard planner, using the map ray cast as LiDAR. Everything else is the real
code: outage schedules, victim camera, radio links, hub (trust, shared map,
frontiers, allocation, victim confirmation, safe modes) and each drone's CBF
filter. Use it to check hub behaviour and tune settings, not to score the model.

    python3 scripts/swarm_dry_run.py --scenario S2_blackout
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ares_gz.metrics import signed_box_distance  # noqa: E402
from ares_gz.navigation import availability_at, outage_schedule, virtual_lidar  # noqa: E402
from ares_gz.scene import scene_map  # noqa: E402
from ares_gz.swarm.agent import DroneAgent  # noqa: E402
from ares_gz.swarm.comms import PROFILES, Link  # noqa: E402
from ares_gz.swarm.hub import Hub, HubConfig, Uplink  # noqa: E402
from ares_gz.swarm.mapping import ring_from_scan  # noqa: E402
from ares_gz.swarm.run_swarm import SwarmConfig, victims_from  # noqa: E402
from ares_gz.swarm.victims import VictimDetector  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_swarm_tests import DEFAULT_MATRIX, load  # noqa: E402


def nominal_velocity(p, goal, cloud_offsets, cruise=1.5):
    to_goal = goal - p
    distance = np.linalg.norm(to_goal[:2])
    v = np.zeros(3)
    if distance > 1e-6:
        v[:2] = to_goal[:2] / distance * min(cruise, 0.7 * distance)
    if cloud_offsets is not None and len(cloud_offsets):
        horizontal = cloud_offsets[:, :2]
        ranges = np.linalg.norm(horizontal, axis=1)
        near = ranges < 3.0
        if near.any():
            push = -(horizontal[near] / ranges[near, None] ** 3).sum(axis=0)
            v[:2] += 2.0 * push
    speed = np.linalg.norm(v)
    return v if speed <= cruise else v * cruise / speed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="S2_blackout")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--comms", default=None)
    args = parser.parse_args()
    data = load(DEFAULT_MATRIX)
    scenario = next(s for s in data["scenarios"] if s["id"].startswith(args.scenario))
    hub_config = HubConfig(**(data.get("hub") or {}))
    config = SwarmConfig(scenario["id"], scenario["lidar"], data["wind"][scenario["wind"]], args.seed,
                         data["drone_specs"], victims_from(data["victims"]), scenario.get("faulty", "all"),
                         scenario.get("outages", "independent"), PROFILES[args.comms or data.get("comms", "nominal")],
                         hub_config, dict(data.get("mission") or {}))
    scene = scene_map()
    ids = [d.id for d in config.drones]
    options = {d: config.drone_options(spec, k) for k, (d, spec) in enumerate(zip(ids, config.drones))}
    schedules = {d: outage_schedule(options[d]) for d in ids}
    hub = Hub(ids, {s.id: s.slot for s in config.drones}, scene, hub_config)
    agents = {d: DroneAgent(d) for d in ids}
    up = {d: Link(config.comms, 10 + k) for k, d in enumerate(ids)}
    down = {d: Link(config.comms, 20 + k) for k, d in enumerate(ids)}
    detector = VictimDetector(config.victims, scene, seed=args.seed)
    p = {d: options[d].start_state[0:3].copy() for d in ids}
    v = {d: np.zeros(3) for d in ids}
    goal = {d: options[d].goal_state[0:3].copy() for d in ids}
    offsets = {d: None for d in ids}
    dt, stop = 0.01, float(options[ids[0]].StopTimeS)
    min_pair, min_clear = np.inf, {d: np.inf for d in ids}
    for k in range(int(stop / dt) + 1):
        t = round(k * dt, 9)
        if k % 10 == 0:
            for d in ids:
                available = availability_at(schedules[d][0], schedules[d][1], t)
                ring = None
                if available:
                    cloud, _, _ = virtual_lidar(scene, p[d], [0.0, 0.0, 0.0])
                    ring = ring_from_scan(cloud, p[d], [0.0, 0.0, 0.0], scene)
                    hits = np.all(np.isfinite(cloud), axis=1)
                    offsets[d] = (cloud[hits] @ np.diag([1.0, -1.0, -1.0])) + np.array([0, 0, -0.4])
                else:
                    truth, _, _ = virtual_lidar(scene, p[d], [0.0, 0.0, 0.0])  # stand-in for the CNN
                    hits = np.all(np.isfinite(truth), axis=1)
                    offsets[d] = (truth[hits] @ np.diag([1.0, -1.0, -1.0])) + np.array([0, 0, -0.4])
                wanted = agents[d].needs_retarget(t)
                if wanted is not None:
                    goal[d] = wanted
                    agents[d].goal_ned = wanted
                up[d].send(t, Uplink(d, t, k, p[d].copy(), v[d].copy(), available, not available, ring, True,
                                     detector.detect(d, p[d], t),
                                     agents[d].downlink.task_id if agents[d].downlink else None, None,
                                     bool(np.linalg.norm(goal[d] - p[d]) < 0.35 and np.linalg.norm(v[d]) < 0.25)))
            for d in ids:
                for message in up[d].receive(t):
                    hub.receive(message)
            for message in hub.tick(t):
                down[message.drone].send(t, message)
        for d in ids:
            agents[d].receive(down[d].receive(t))
            nominal = nominal_velocity(p[d], goal[d], offsets[d])
            safe = agents[d].filter(t, p[d], nominal)
            v[d] = v[d] + np.clip(safe - v[d], -1.0 * dt * 5, 1.0 * dt * 5)
            p[d] = p[d] + v[d] * dt
            min_clear[d] = min(min_clear[d], float(signed_box_distance(p[d], scene)[0] - 0.1))
        for a, b in itertools.combinations(ids, 2):
            min_pair = min(min_pair, float(np.linalg.norm(p[a] - p[b])))
        if hub.phase == "RENDEZVOUS" and all(r.arrived for r in hub.drones.values() if r.mode != "LOST"):
            break
    summary = hub.summary()
    print(f"{scenario['id']}: ended t={t:.1f} s phase {summary['phase']}  min pair {min_pair:.2f} m  "
          f"coverage {summary['coverage']:.0%}")
    print("min building clearance:", {d: round(c, 2) for d, c in min_clear.items()})
    print("modes:", summary["modes"], " arrived:", summary["arrived"])
    for victim in summary["victims"]:
        print(f"  victim {victim['id']} ({victim['truth_id']}) {victim['status']} first by {victim['first_reported_by']} "
              f"at {victim['first_reported_s']} s (lidar failed: {victim['reporter_lidar_failed']}), "
              f"confirmed by {victim['confirmed_by']} at {victim['confirmed_s']}")
    print("events:")
    for event in hub.events[:40] + [{"time_s": 0, "drone": "...", "event": "..."}] + hub.events[-30:]:
        print(f"  {event['time_s']:6.1f}  {event['drone'] or 'HUB':5} {event['event']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
