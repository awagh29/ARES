#!/usr/bin/env python3
"""Measure how this Gazebo install behaves and write config/vehicle_calibration.json.

Run once after installing Gazebo (inside WSL, from gazebo_test_pipeline):
    python3 scripts/diagnose.py [--gui] [--no-headless-rendering]

Checks, each in a short lock-step session:
  1. hover     - the velocity controller holds altitude with a zero command;
  2. frame     - whether velocity commands are in the world or heading frame;
  3. gain      - first-order velocity tracking rate (1/s) used by CommandMapper;
  4. wind      - a known external force moves the drone the right way and the
                 "publish only the change" wrench scheme adds up correctly;
  5. lidar     - Gazebo's scan matches the MATLAB ray cast at the same pose,
                 and which row/column order makes it match.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ares_gz.frames import ENU_NED, gazebo_orientation_to_ned_euler  # noqa: E402
from ares_gz.gazebo import GazeboSession  # noqa: E402
from ares_gz.navigation import virtual_lidar  # noqa: E402
from ares_gz.navigation.lidar import gazebo_cloud_to_matlab_order  # noqa: E402
from ares_gz.run import CALIBRATION_FILE  # noqa: E402
from ares_gz.scene import scene_map  # noqa: E402
from ares_gz.world import world_sdf  # noqa: E402

DT = 0.05


class Probe:
    """A lock-step session with simple helpers."""

    def __init__(self, start_ned, args, workdir: Path, name: str):
        self.scene = scene_map()
        world = workdir / f"{name}.sdf"
        world.write_text(world_sdf(self.scene, start_ned))
        self.session = GazeboSession(world, workdir / name, gui=args.gui,
                                     headless_rendering=not args.no_headless_rendering)
        self.t = 0.0

    def __enter__(self):
        self.session.start()
        return self

    def __exit__(self, *exc):
        self.session.close()

    def run(self, seconds: float, velocity_enu=(0.0, 0.0, 0.0), force_enu=(0.0, 0.0, 0.0), need_scan=False):
        """Hold a command for `seconds`; return list of (t, position_enu, velocity_enu)."""
        trace = []
        for _ in range(int(round(seconds / DT))):
            self.session.enable_controller()
            self.session.send_velocity(velocity_enu, 0.0)
            self.session.set_force_enu(force_enu)
            self.t += DT
            scan_at = math.floor(self.t / 0.1 + 1e-7) * 0.1 if need_scan else None
            self.session.step(self.t, need_scan_at=scan_at)
            odometry = self.session.odometry
            last, previous = odometry[-1], odometry[-2] if len(odometry) > 1 else odometry[-1]
            dt = max(last.time_s - previous.time_s, 1e-9)
            trace.append((last.time_s, last.position_enu.copy(), (last.position_enu - previous.position_enu) / dt))
        return trace

    def yaw_enu(self) -> float:
        euler_ned = gazebo_orientation_to_ned_euler(*self.session.odometry[-1].orientation_wxyz)
        return math.pi / 2 - euler_ned[0]


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-headless-rendering", action="store_true")
    args = parser.parse_args()
    workdir = Path(tempfile.mkdtemp(prefix="ares_gz_diag_"))
    print(f"Diagnostics (logs in {workdir})")
    results: dict = {}
    all_ok = True

    # 1-4: open air at the mission start, nose north (Gazebo yaw = +90 deg).
    with Probe([0.0, 0.0, -7.0], args, workdir, "flight") as probe:
        print("1. Hover")
        trace = probe.run(3.0)
        drift = np.linalg.norm(trace[-1][1] - np.array([0.0, 0.0, 7.0]))
        all_ok &= check("hover", drift < 0.5, f"drift {drift:.3f} m after 3 s")
        results["hover_drift_m"] = float(drift)

        print("2. Command frame (command +x for 2 s, nose points north)")
        start = trace[-1][1]
        trace = probe.run(2.0, velocity_enu=(1.0, 0.0, 0.0))
        moved = trace[-1][1] - start
        frame = "world" if abs(moved[0]) > abs(moved[1]) else "heading"
        all_ok &= check("frame", np.linalg.norm(moved[:2]) > 0.5,
                        f"moved east {moved[0]:+.2f} m, north {moved[1]:+.2f} m -> commands are in the {frame} frame")
        results["frame"] = frame

        print("3. Velocity tracking rate (stop, then step to 1 m/s along the command axis)")
        probe.run(3.0)
        axis = 0 if frame == "world" else 1  # the ENU axis the +x command moves along
        trace = probe.run(3.0, velocity_enu=(1.0, 0.0, 0.0))
        times = np.array([sample[0] for sample in trace]) - trace[0][0] + DT
        speed = np.array([sample[2][axis] for sample in trace])
        usable = (speed > 0.05) & (speed < 0.85)
        gain = float(np.median(-np.log(1.0 - speed[usable]) / times[usable])) if np.any(usable) else float("nan")
        final = float(speed[-1])
        all_ok &= check("gain", np.isfinite(gain) and 0.3 < gain < 20 and abs(final - 1.0) < 0.15,
                        f"k = {gain:.3f} 1/s (nominal 2.7/1.52 = {2.7 / 1.52:.3f}), speed after 3 s {final:.3f} m/s")
        results["gain_per_s"] = gain

        print("4. External force (wind) via persistent wrench deltas")
        probe.run(3.0)
        speeds = []
        for force in (1.5, 3.0, 0.0):
            trace = probe.run(4.0, force_enu=(0.0, force, 0.0))  # push north
            speeds.append(float(np.mean([sample[2][1] for sample in trace[-10:]])))
        expected = [force / (1.52 * gain) for force in (1.5, 3.0, 0.0)] if np.isfinite(gain) else [np.nan] * 3
        ok = speeds[0] > 0.1 and abs(speeds[1] - 2 * speeds[0]) < 0.35 * speeds[1] + 0.05 and abs(speeds[2]) < 0.1
        all_ok &= check("wind force", ok, "steady drift north at 1.5/3.0/0 N: "
                        + ", ".join(f"{value:.2f}" for value in speeds) + " m/s (expected about "
                        + ", ".join(f"{value:.2f}" for value in expected) + ")")
        results["force_drift_mps"] = speeds

    # 5: inside the cluster at 14 m, just below the 15 m roofs: upward beams pass over the
    # roofs while downward beams hit walls, so a vertical flip is detectable.
    print("5. Lidar layout vs the MATLAB ray cast")
    with Probe([10.0, 12.0, -14.0], args, workdir, "lidar") as probe:
        probe.run(1.0, need_scan=True)
        scan = probe.session.latest_scan
        odometry = probe.session.odometry[-1]
        position = ENU_NED @ odometry.position_enu
        euler = gazebo_orientation_to_ned_euler(*odometry.orientation_wxyz)
        _, ranges_ref, hits_ref = virtual_lidar(probe.scene, position, euler)
        best = None
        for flip_v in (False, True):
            for flip_h in (False, True):
                cloud = gazebo_cloud_to_matlab_order(scan.xyz, flip_v, flip_h)
                hits = np.all(np.isfinite(cloud), axis=1)
                ranges = np.linalg.norm(np.nan_to_num(cloud.astype(float)), axis=1)
                agree = float(np.mean(hits == hits_ref))
                both = hits & hits_ref
                error = float(np.median(np.abs(ranges[both] - ranges_ref[both]))) if np.any(both) else float("inf")
                print(f"     flip_vertical={flip_v!s:5} flip_horizontal={flip_h!s:5}: "
                      f"hit agreement {agree:.3f}, median range error {error:.3f} m")
                if best is None or (agree, -error) > (best[0], -best[1]):
                    best = (agree, error, flip_v, flip_h)
        all_ok &= check("lidar", best[0] > 0.97 and best[1] < 0.05,
                        f"{scan.height}x{scan.width} scan; best layout agreement {best[0]:.3f}, "
                        f"median error {best[1]:.3f} m ({int(hits_ref.sum())} beams hit in the MATLAB ray cast)")
        results.update(lidar_flip_vertical=best[2], lidar_flip_horizontal=best[3],
                       lidar_hit_agreement=best[0], lidar_median_range_error_m=best[1],
                       lidar_shape=[scan.height, scan.width])

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    results["all_checks_passed"] = bool(all_ok)
    CALIBRATION_FILE.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\n{'All checks passed' if all_ok else 'SOME CHECKS FAILED'}; wrote {CALIBRATION_FILE}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
