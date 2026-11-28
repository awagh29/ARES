#!/usr/bin/env python3
"""Three-drone search-and-rescue tests in Gazebo with the swarm hub in the loop.

  python3 run_swarm_tests.py --list
  python3 run_swarm_tests.py --scenario S0                 # one scenario (all seeds)
  python3 run_swarm_tests.py --all --video                 # every scenario + replay videos
  python3 run_swarm_tests.py --scenario S2 --comms lossy   # stress the radio links
  python3 run_swarm_tests.py --report results/swarm_map-temporal-cnn_planner

Each run starts one MATLAB guidance session per drone (they run in parallel).
Results: results/<campaign>/<scenario>/seed####_<id>/ (summary.json, hub_events.json,
trust.csv, drones/<UAV>/control.csv) and REPORT.md for the campaign.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ares_gz.paths import RESULTS_ROOT  # noqa: E402
from ares_gz.swarm.comms import PROFILES  # noqa: E402
from ares_gz.swarm.hub import HubConfig  # noqa: E402
from ares_gz.swarm.run_swarm import DroneSpec, SwarmConfig, build_controllers, run_swarm, victims_from  # noqa: E402
from ares_gz.wind import WindProfile  # noqa: E402

DEFAULT_MATRIX = Path(__file__).resolve().parent / "config" / "swarm_scenarios.yaml"


def load(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    data["wind"] = {name: replace(WindProfile(), name=name, **values) for name, values in data["wind_profiles"].items()}
    data["drone_specs"] = [DroneSpec(str(d["id"]), float(d["north"]), float(d["east"]), tuple(d["slot"]))
                           for d in data["drones"]]
    return data


def select(data: dict, wanted: list[str] | None) -> list[dict]:
    scenarios = data["scenarios"]
    if not wanted:
        return scenarios
    chosen = []
    for key in wanted:
        matches = [s for s in scenarios if s["id"] == key or s["id"].startswith(key)]
        if not matches:
            raise SystemExit(f"Unknown scenario {key!r}; use --list")
        chosen.extend(m for m in matches if m not in chosen)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--scenario", nargs="+", help="scenario ids or prefixes (e.g. S1 S4)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=lambda text: [int(x) for x in text.split(",")], help="e.g. 1,2,3")
    parser.add_argument("--comms", choices=sorted(PROFILES), help="override the YAML radio profile")
    parser.add_argument("--guidance", choices=("planner", "governor", "vfh"))
    parser.add_argument("--model", help="override controller.model_path")
    parser.add_argument("--campaign")
    parser.add_argument("--gui", action="store_true", help="also open the Gazebo GUI")
    parser.add_argument("--lidar-source", choices=("gazebo", "raycast"), default="gazebo",
                        help="raycast: same buildings + the other drones ray-cast in Python (no GPU LiDAR "
                             "rendering, several times faster); Gazebo still flies all physics")
    parser.add_argument("--video", action="store_true", help="render replay videos after each run")
    parser.add_argument("--manual-matlab", action="store_true")
    parser.add_argument("--port", type=int, default=50655, help="first MATLAB bridge port (one per drone)")
    parser.add_argument("--report", type=Path, help="only (re)build REPORT.md for a swarm results folder")
    args = parser.parse_args()

    if args.report:
        from ares_gz.swarm.report import write_report
        print(write_report(args.report))
        return 0
    data = load(args.matrix)
    if args.list:
        print(f"drones: {[(d.id, d.north, d.east, d.slot) for d in data['drone_specs']]}")
        print(f"victims: {[(v['id'], v['north'], v['east']) for v in data['victims']]}")
        for s in data["scenarios"]:
            print(f"  {s['id']:<30} lidar={s['lidar']:<12} wind={s['wind']:<7} faulty={s.get('faulty', 'all')} "
                  f"outages={s.get('outages', 'independent')}  {s.get('note', '')}")
        return 0
    if not (args.all or args.scenario):
        parser.error("choose --scenario ..., --all, --list or --report")

    controller = data["controller"]
    guidance = args.guidance or controller.get("guidance", "planner")
    campaign = args.campaign or f"{data.get('campaign', 'swarm')}_map-temporal-cnn_{guidance}"
    from ares_gz.model_artifact import resolve_model
    model_path, info = resolve_model(args.model or controller["model_path"])
    print(f"Model: {info.get('requested')}   guidance: {guidance}   campaign: {campaign}")
    export_python = os.environ.get("ARES_EXPORT_PYTHON", controller.get("export_python", "") or "")
    ids = [spec.id for spec in data["drone_specs"]]
    os.environ.setdefault("ARES_ROS_DOMAIN", str(60 + os.getpid() % 30))
    controllers, bridges = build_controllers(ids, model_path, export_python, guidance, args.port,
                                             args.manual_matlab, RESULTS_ROOT)
    hub_config = HubConfig(**{k: v for k, v in (data.get("hub") or {}).items()})
    comms = PROFILES[args.comms or data.get("comms", "nominal")]
    jobs = [(scenario, seed) for scenario in select(data, args.scenario) for seed in (args.seeds or data["seeds"])]
    print(f"{len(jobs)} swarm run(s)")
    started = time.monotonic()
    try:
        for scenario, seed in jobs:
            config = SwarmConfig(
                scenario=scenario["id"], lidar_mode=scenario["lidar"], wind=data["wind"][scenario["wind"]], seed=seed,
                drones=data["drone_specs"], victims=victims_from(data["victims"]),
                faulty=scenario.get("faulty", "all"), outage_sync=scenario.get("outages", "independent"),
                comms=comms, hub=hub_config, mission=dict(data.get("mission") or {}), campaign=campaign,
                lidar_source=args.lidar_source, gui=args.gui)
            summary = run_swarm(config, controllers)
            if args.video and summary.get("end_time_s"):
                from ares_gz.swarm.replay_swarm import render_swarm_replay
                try:
                    render_swarm_replay(Path(summary["directory"]), copy_to=RESULTS_ROOT / campaign / "videos")
                except Exception as error:  # a video problem must not stop the campaign
                    print(f"  video skipped: {error}", flush=True)
    finally:
        for bridge in bridges:
            bridge.close()
    from ares_gz.swarm.report import write_report
    print(write_report(RESULTS_ROOT / campaign))
    print(f"Done in {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
