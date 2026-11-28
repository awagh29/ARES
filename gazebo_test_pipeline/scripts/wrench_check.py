#!/usr/bin/env python3
"""Check that wind forces stay cheap and correct over a long run (Gazebo, no MATLAB).

The quad hovers for ``--seconds`` while a sideways force that changes every tick
is applied the way the runners apply drag (persistent-wrench deltas, with the
list cleared every ``WRENCH_MAX_ENTRIES`` entries). Prints wall time per 10 s of
simulation (must stay flat) and the hover offset the force causes (must match
a constant force of the same mean).

    python3 scripts/wrench_check.py --seconds 60
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ares_gz.gazebo import GazeboSession  # noqa: E402
from ares_gz.scene import scene_map  # noqa: E402
from ares_gz.world import world_sdf  # noqa: E402

DT = 0.01


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--constant", action="store_true", help="reference: constant force (one wrench entry)")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="ares_wrench_"))
    world = work / "w.sdf"
    world.write_text(world_sdf(scene_map(), [0.0, 0.0, -7.0]))
    session = GazeboSession(world, work / "logs", steps_per_tick=10)
    offsets, chunk_walls = [], []
    with session:
        t, chunk = 0.0, time.monotonic()
        for k in range(int(args.seconds / DT)):
            ripple = 0.0 if args.constant else 0.1 * np.sin(k * 0.7)  # > threshold every tick
            force = np.array([1.5 + ripple, 0.0, 0.0])
            session.enable_controller()
            session.send_velocity((0.0, 0.0, 0.0), 0.0)
            session.set_force_enu(force)
            t = round(t + DT, 9)
            session.step(t)
            if k % 100 == 99:
                offsets.append(float(session.odometry[-1].position_enu[0]))
            if k % 1000 == 999:
                chunk_walls.append(time.monotonic() - chunk)
                chunk = time.monotonic()
        clears = session.wrench_clears
    print("wall s per 10 s sim:", [round(w, 1) for w in chunk_walls])
    print("east offset (m) each second:", [round(o, 2) for o in offsets[:10]], f" clears: {clears}")
    flat = chunk_walls[-1] < 1.5 * min(chunk_walls)
    pushed = offsets[5] > 0.05
    print("PASS" if flat and pushed else "FAIL", "- speed flat" if flat else "- slowing down",
          "- force acts" if pushed else "- force missing")
    return 0 if flat and pushed else 1


if __name__ == "__main__":
    raise SystemExit(main())
