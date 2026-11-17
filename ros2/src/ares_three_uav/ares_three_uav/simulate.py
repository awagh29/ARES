"""Closed-loop flight of the three vehicles on the guidance plant.

The plant, PID cascade, waypoint follower and 3DVFH+ avoidance are stepped
here. Gazebo is the scene and the playback viewer; it is not a second
dynamics model. Run ``python3 -m ares_three_uav.simulate`` from a path that
contains this package.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .frames import enu_orientation, ned_to_enu
from .mission import PLANT_DT_S, STOP_TIME_S, VEHICLE_RADIUS_M
from .pilot import Pilot, minimum_surface_clearance
from .schedule import intermittent_schedule
from .world import live_world, playback_world


def fly(stop_time_s: float = STOP_TIME_S):
    times, enabled, intervals = intermittent_schedule(stop_time_s)
    pilots = [Pilot(index, times, enabled) for index in range(3)]
    steps = int(round(stop_time_s / PLANT_DT_S))
    sample_stride = int(round(0.1 / PLANT_DT_S))
    rows = []
    playback = {pilot.name: [] for pilot in pilots}
    clearance = [math.inf] * len(pilots)
    for step in range(steps):
        time_s = step * PLANT_DT_S
        for index, pilot in enumerate(pilots):
            pilot.step(step)
            gap = minimum_surface_clearance(pilot.position_ned, pilot.box_min, pilot.box_max)
            clearance[index] = min(clearance[index], gap)
            if step % sample_stride == 0:
                north, east, down = pilot.position_ned
                enu_x, enu_y, enu_z = ned_to_enu(north, east, down)
                roll, pitch, yaw = (float(pilot.state[6]), float(pilot.state[7]), float(pilot.state[8]))
                enu_roll, enu_pitch, enu_yaw, _quat = enu_orientation(roll, pitch, yaw)
                rows.append({
                    "time_s": time_s,
                    "drone": pilot.name,
                    "north_m": float(north),
                    "east_m": float(east),
                    "down_m": float(down),
                    "enu_x_m": enu_x,
                    "enu_y_m": enu_y,
                    "enu_z_m": enu_z,
                    "yaw_rad": yaw,
                    "lidar_guidance_enabled": int(pilot.lidar_enabled),
                    "avoidance_used": int(pilot.using_avoidance),
                    "goal_distance_m": pilot.goal_distance,
                })
                if step % (sample_stride * 2) == 0:
                    playback[pilot.name].append((time_s, enu_x, enu_y, enu_z, enu_roll, enu_pitch, enu_yaw))
    summary = {
        "stop_time_s": stop_time_s,
        "outage_intervals_s": intervals.tolist(),
        "schedule_time_s": times.tolist(),
        "schedule_enabled": enabled.astype(int).tolist(),
        "required_center_clearance_m": VEHICLE_RADIUS_M + 1.0,
        "drones": [],
    }
    for pilot, gap in zip(pilots, clearance):
        summary["drones"].append({
            "name": pilot.name,
            "east_offset_m": pilot.east_offset,
            "final_position_ned_m": pilot.position_ned.tolist(),
            "goal_ned_m": pilot.waypoints[-1].tolist(),
            "goal_distance_m": pilot.goal_distance,
            "finish_time_s": pilot.finish_time_s,
            "minimum_center_clearance_m": None if math.isinf(gap) else float(gap),
            "goal_reached": pilot.goal_distance <= 1.0,
            "clearance_ok": float(gap) >= VEHICLE_RADIUS_M + 1.0,
        })
    return pilots, rows, summary, playback


def write_outputs(destination: Path, rows, summary, playback) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    world_dir = Path(__file__).resolve().parents[1] / "worlds"
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / "ares_three_uav.sdf").write_text(live_world())
    ordered = [(name, playback[name]) for name in playback]
    (world_dir / "ares_three_uav_playback.sdf").write_text(playback_world(ordered))


def main() -> None:
    _pilots, rows, summary, playback = fly()
    destination = Path(__file__).resolve().parents[4] / "results" / "gazebo_three_uav"
    write_outputs(destination, rows, summary, playback)
    for drone in summary["drones"]:
        print(
            f"{drone['name']}: goal {drone['goal_distance_m']:.3f} m at "
            f"{drone['finish_time_s']:.2f} s, "
            f"minimum center clearance {drone['minimum_center_clearance_m']:.3f} m"
        )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
