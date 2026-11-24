#!/usr/bin/env python3
"""Run the single-drone Gazebo test scenarios.

Examples (inside WSL, from the gazebo_test_pipeline folder):
  python3 run_tests.py --list
  python3 run_tests.py --scenario S1_intermittent --seed 513 --gui
  python3 run_tests.py --all                       # every scenario x every seed in the YAML
  python3 run_tests.py --scenario S3 S4 --seeds 1-3
  python3 run_tests.py --report results/single_drone
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ares_gz.paths import RESULTS_ROOT  # noqa: E402
from ares_gz.report import write_report  # noqa: E402
from ares_gz.run import RunConfig, run_scenario  # noqa: E402
from ares_gz.scenarios import DEFAULT_MATRIX, load_matrix  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="scenario YAML file")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--scenario", nargs="+", help="scenario ids or prefixes (e.g. S1 S4)")
    parser.add_argument("--all", action="store_true", help="run every scenario in the matrix")
    parser.add_argument("--seed", type=int, help="run a single seed")
    parser.add_argument("--seeds", type=parse_seeds, help="seed list, e.g. 1-10 or 1,4,513")
    parser.add_argument("--campaign", help="results sub-folder name (default from the YAML)")
    parser.add_argument("--gui", action="store_true", help="also open the Gazebo GUI to watch")
    parser.add_argument("--no-headless-rendering", action="store_true",
                        help="render the lidar through the desktop (WSLg) instead of headless EGL")
    parser.add_argument("--report", type=Path, help="only (re)build the report for a results folder")
    args = parser.parse_args()

    if args.report:
        print(write_report(args.report))
        return 0
    matrix = load_matrix(args.matrix)
    if args.list:
        for scenario in matrix.scenarios:
            print(f"{scenario.id:32s} lidar={scenario.lidar:12s} wind={scenario.wind.name:7s} {scenario.note}")
        return 0
    if not (args.all or args.scenario):
        parser.error("choose --scenario ..., --all, --list or --report")
    scenarios = matrix.select(None if args.all else args.scenario)
    seeds = [args.seed] if args.seed is not None else (args.seeds or matrix.seeds)
    campaign = args.campaign or matrix.campaign
    total = 0
    started = time.monotonic()
    for scenario in scenarios:
        deterministic = scenario.lidar in ("healthy", "blackout") and scenario.wind.is_calm
        # No randomness in these: two seeds are enough to show the result repeats.
        run_seeds = seeds[:2] if deterministic and args.seed is None else seeds
        for seed in run_seeds:
            config = RunConfig(scenario=scenario.id, lidar_mode=scenario.lidar, wind=scenario.wind, seed=seed,
                               overrides=matrix.mission, gui=args.gui,
                               headless_rendering=not args.no_headless_rendering, campaign=campaign)
            run_scenario(config)
            total += 1
    report = write_report(RESULTS_ROOT / campaign)
    print(f"\n{total} run(s) in {time.monotonic() - started:.0f} s. Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
