#!/usr/bin/env python3
"""Run the single-drone Gazebo test scenarios.

Examples (inside WSL, from the gazebo_test_pipeline folder):
  python3 run_tests.py --list
  python3 run_tests.py --scenario S1_intermittent --start W1 --seed 1 --gui
  python3 run_tests.py --all --workers 2 --video   # every scenario x start x seed, 2 at a time, with videos
  python3 run_tests.py --scenario S3 S4 --seeds 1-3
  python3 run_tests.py --all --controller map-correction
  python3 run_tests.py --report results/single_drone_map-temporal-cnn
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ares_gz.controllers import CONTROLLERS, make_controller  # noqa: E402
from ares_gz.paths import RESULTS_ROOT  # noqa: E402
from ares_gz.report import write_report  # noqa: E402
from ares_gz.scenarios import DEFAULT_MATRIX, load_matrix  # noqa: E402
from ares_gz.scene import scene_map  # noqa: E402

# Rough wall-time weights, used only to start the longest runs first.
WEIGHT = {"blackout": 5.0, "intermittent": 1.3, "healthy": 1.0, "custom": 1.3}


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def build_jobs(matrix, args) -> list[tuple[str, str, int]]:
    scenarios = matrix.select(None if args.all else args.scenario)
    starts = matrix.select_starts(args.start)
    seeds = [args.seed] if args.seed is not None else (args.seeds or matrix.seeds)
    jobs = []
    for scenario in scenarios:
        deterministic = scenario.lidar in ("healthy", "blackout") and scenario.wind.is_calm
        # No randomness in these: two seeds are enough to show the result repeats.
        run_seeds = seeds[:2] if deterministic and args.seed is None else seeds
        for start in starts:
            for seed in run_seeds:
                jobs.append((scenario.id, start.id, seed))
    lidar = {scenario.id: scenario.lidar for scenario in matrix.scenarios}
    return sorted(jobs, key=lambda job: -WEIGHT.get(lidar[job[0]], 1.0))


def build_controller(name: str, settings: dict, options: dict, worker: int):
    """One controller (and one MATLAB session) per worker, shared by all its runs."""
    if name == "map-correction":
        return make_controller(name, scene=scene_map()), None
    from ares_gz.controllers.map_temporal import MapTemporalController
    from ares_gz.matlab_bridge import MatlabBridge
    from ares_gz.model_artifact import resolve_model

    model_path, info = resolve_model(options["model"] or settings.get("model_path", ""))
    if worker == 0:
        print(f"Model: {info.get('requested')}"
              + (f" (reusing export {info['reused_export']})" if "reused_export" in info else "")
              + f"   guidance: {options.get('guidance') or settings.get('guidance', 'planner')}")
    export_python = os.environ.get("ARES_EXPORT_PYTHON", settings.get("export_python", "") or "")
    bridge = MatlabBridge(port=options["port"] + worker, launch=not options["manual_matlab"],
                          log_file=RESULTS_ROOT / f"matlab_bridge_worker{worker}.log")
    guidance = options.get("guidance") or settings.get("guidance", "planner")
    return MapTemporalController(bridge, model_path, export_python, guidance), bridge


def worker_main(worker: int, jobs, options: dict) -> int:
    """Run jobs (a list, or a multiprocessing queue ending with None). Returns runs completed."""
    # Each worker gets its own ROS 2 domain; Gazebo partitions are already unique per run.
    os.environ["ARES_ROS_DOMAIN"] = str(options["domain_base"] + worker)
    from ares_gz.run import RunConfig, run_scenario

    matrix = load_matrix(Path(options["matrix"]))
    scenarios = {scenario.id: scenario for scenario in matrix.scenarios}
    starts = {start.id: start for start in matrix.starts}
    altitude_down = float(matrix.mission.get("StartState", [0, 0, -7])[2])
    controller, bridge = build_controller(options["controller"], matrix.controller, options, worker)
    done = 0
    try:
        while True:
            job = jobs.get() if hasattr(jobs, "get") else (jobs.pop(0) if jobs else None)
            if job is None:
                break
            scenario_id, start_id, seed = job
            scenario, start = scenarios[scenario_id], starts[start_id]
            overrides = dict(matrix.mission, StartState=start.state(altitude_down))
            config = RunConfig(scenario=scenario.id, lidar_mode=scenario.lidar, wind=scenario.wind, seed=seed,
                               controller=options["controller"], overrides=overrides, gui=options["gui"],
                               headless_rendering=options["headless_rendering"], campaign=options["campaign"],
                               start_id=start.id)
            if options["workers"] > 1:
                print(f"[worker {worker}] starting {scenario_id} @ {start_id} seed {seed}", flush=True)
            summary = run_scenario(config, controller=controller)
            done += 1
            if options["video"] and summary.get("control_ticks"):
                from ares_gz.replay import render_replay
                try:
                    render_replay(Path(summary["directory"]), copy_to=RESULTS_ROOT / options["campaign"] / "videos")
                except Exception as error:  # a video problem must not stop the campaign
                    print(f"  video skipped: {error}", flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        if bridge is not None:
            bridge.close()
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="scenario YAML file")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--scenario", nargs="+", help="scenario ids or prefixes (e.g. S1 S4)")
    parser.add_argument("--all", action="store_true", help="run every scenario in the matrix")
    parser.add_argument("--seed", type=int, help="run a single seed")
    parser.add_argument("--seeds", type=parse_seeds, help="seed list, e.g. 1-10 or 1,4,513")
    parser.add_argument("--start", nargs="+", help="start point ids from the YAML (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="simulations at once; each has its own MATLAB + Gazebo (2-3 on a 16 GB PC)")
    parser.add_argument("--controller", choices=CONTROLLERS, help="override the YAML controller")
    parser.add_argument("--model", help="override controller.model_path (best.pt or model.mat)")
    parser.add_argument("--guidance", choices=("planner", "governor", "vfh"),
                        help="map-temporal guidance after the CNN (default from the YAML: planner)")
    parser.add_argument("--campaign", help="results sub-folder name (default from the YAML)")
    parser.add_argument("--gui", action="store_true", help="also open the Gazebo GUI to watch")
    parser.add_argument("--no-headless-rendering", action="store_true",
                        help="render the lidar through the desktop (WSLg) instead of headless EGL")
    parser.add_argument("--manual-matlab", action="store_true",
                        help="do not start MATLAB; wait for runGuidanceBridge to be started by hand")
    parser.add_argument("--port", type=int, default=50555, help="first MATLAB bridge port (one per worker)")
    parser.add_argument("--report", type=Path, help="only (re)build the report for a results folder")
    parser.add_argument("--video", action="store_true",
                        help="after each run, render smooth replay videos (needs ffmpeg)")
    args = parser.parse_args()

    if args.report:
        print(write_report(args.report))
        return 0
    matrix = load_matrix(args.matrix)
    if args.list:
        print(f"controller: {matrix.controller['name']}   seeds: {matrix.seeds}")
        for start in matrix.starts:
            print(f"start {start.id}: north {start.north:g} m, east {start.east:g} m  {start.note}")
        for scenario in matrix.scenarios:
            print(f"{scenario.id:32s} lidar={scenario.lidar:12s} wind={scenario.wind.name:7s} {scenario.note}")
        return 0
    if not (args.all or args.scenario):
        parser.error("choose --scenario ..., --all, --list or --report")
    controller_name = args.controller or matrix.controller["name"]
    guidance = args.guidance or matrix.controller.get("guidance", "planner")
    suffix = f"_{guidance}" if controller_name == "map-temporal-cnn" else ""
    campaign = args.campaign or f"{matrix.campaign}_{controller_name}{suffix}"
    jobs = build_jobs(matrix, args)
    workers = max(1, min(args.workers, len(jobs)))
    if args.gui and workers > 1:
        parser.error("--gui works with --workers 1 only")
    options = {"matrix": str(args.matrix), "controller": controller_name, "model": args.model, "guidance": guidance,
               "manual_matlab": args.manual_matlab, "port": args.port, "gui": args.gui,
               "headless_rendering": not args.no_headless_rendering, "campaign": campaign,
               "video": args.video, "workers": workers, "domain_base": 40 + (os.getpid() % 40)}
    print(f"{len(jobs)} run(s), {workers} worker(s), campaign '{campaign}'", flush=True)
    started = time.monotonic()
    if workers == 1:
        total = worker_main(0, list(jobs), options)
    else:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        for job in jobs:
            queue.put(job)
        for _ in range(workers):
            queue.put(None)
        processes = [context.Process(target=worker_main, args=(index, queue, options), name=f"worker{index}")
                     for index in range(workers)]
        for index, process in enumerate(processes):
            process.start()
            time.sleep(15 if index + 1 < workers else 0)  # stagger MATLAB start-ups
        for process in processes:
            process.join()
        total = len(jobs)
    report = write_report(RESULTS_ROOT / campaign)
    print(f"\n{total} run(s) in {(time.monotonic() - started) / 60:.1f} min. Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
