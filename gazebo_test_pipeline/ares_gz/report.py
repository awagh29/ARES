"""Aggregate run summaries into a table, a markdown report and plots."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .scene import scene_map

COLUMNS = ["scenario", "start_id", "start_ned", "seed", "controller", "lidar_mode", "wind", "mission_success", "termination_reason",
           "arrival_time_s", "minimum_clearance_m", "collision", "near_collision", "final_goal_error_m",
           "max_route_deviation_m", "observed_outages", "planned_outages", "reconstruction_scans",
           "max_wind_mps", "max_drag_n", "controller_healthy", "outages_used_reconstruction", "error", "directory"]


def collect(campaign_dir: Path) -> list[dict]:
    summaries = [json.loads(path.read_text()) for path in sorted(Path(campaign_dir).rglob("summary.json"))]
    return [summary for summary in summaries if "scenario" in summary]


def group_key(summary: dict) -> str:
    start = summary.get("start_id")
    return summary["scenario"] if not start or start == "default" else f"{summary['scenario']} @ {start}"


def _fmt(value, digits=2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(campaign_dir: Path) -> Path:
    campaign_dir = Path(campaign_dir)
    summaries = collect(campaign_dir)
    if not summaries:
        raise SystemExit(f"No summary.json files under {campaign_dir}")
    with (campaign_dir / "all_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    groups: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        groups[group_key(summary)].append(summary)
    controllers = sorted({summary.get("controller", "map-correction") for summary in summaries})
    lines = [f"# Gazebo test report: `{campaign_dir.name}`", "",
             f"Controller(s): {', '.join(controllers)}. Pass = goal reached at rest, no collision, "
             f"surface clearance >= {summaries[0]['required_clearance_m']} m, altitude in bounds, the "
             "controller never failed, and (CNN) every outage scan used the reconstruction.", "",
             "| Scenario | Runs | Passed | Collisions | Min clearance (m) | Mean arrival (s) | Max route deviation (m) | Failure reasons |",
             "|---|---|---|---|---|---|---|---|"]
    for scenario in sorted(groups):
        runs = groups[scenario]
        passed = sum(1 for run in runs if run["mission_success"])
        collisions = sum(1 for run in runs if run.get("collision"))
        clearances = [run["minimum_clearance_m"] for run in runs if run.get("minimum_clearance_m") is not None]
        arrivals = [run["arrival_time_s"] for run in runs if run["mission_success"]]
        deviations = [run["max_route_deviation_m"] for run in runs if run.get("max_route_deviation_m") is not None]
        reasons = sorted({run["termination_reason"] for run in runs if not run["mission_success"]})
        lines.append(f"| {scenario} | {len(runs)} | {passed}/{len(runs)} | {collisions} | "
                     f"{_fmt(min(clearances) if clearances else None, 3)} | "
                     f"{_fmt(float(np.mean(arrivals)) if arrivals else None)} | "
                     f"{_fmt(max(deviations) if deviations else None)} | {', '.join(reasons) or '-'} |")
    lines += ["", "Per-run results: `all_runs.csv`. Plots: `trajectories.png`, `clearance.png`. "
              "Replay videos (if rendered): `videos/`.", ""]
    report = campaign_dir / "REPORT.md"
    report.write_text("\n".join(lines))
    try:
        _plots(campaign_dir, groups)
    except Exception as error:  # plotting must never hide the table
        print(f"(plots skipped: {error})")
    return report


def _plots(campaign_dir: Path, groups: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scene = scene_map()
    names = sorted(groups)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for axis, name in zip(axes.flat, names):
        for lo, hi in zip(scene.box_min, scene.box_max):
            axis.add_patch(Rectangle((lo[1], lo[0]), hi[1] - lo[1], hi[0] - lo[0], color="0.6"))
        for run in groups[name]:
            path = Path(run["directory"]) / "odometry_ned.csv"
            if not path.exists():
                continue
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            color = "tab:green" if run["mission_success"] else "tab:red"
            axis.plot(data[:, 2], data[:, 1], color=color, linewidth=0.8, alpha=0.8)
        axis.set_title(name, fontsize=9)
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        axis.set_aspect("equal")
        axis.set_xlim(-5, 40)
        axis.set_ylim(-5, 35)
    for axis in list(axes.flat)[len(names):]:
        axis.axis("off")
    figure.suptitle("Flown paths (green = pass, red = fail)")
    figure.tight_layout()
    figure.savefig(campaign_dir / "trajectories.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4))
    data = [[run["minimum_clearance_m"] for run in groups[name] if run.get("minimum_clearance_m") is not None]
            for name in names]
    axis.boxplot(data, tick_labels=names) if data else None
    required = next(iter(groups.values()))[0]["required_clearance_m"]
    axis.axhline(required, color="tab:red", linestyle="--", label=f"required {required} m")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Minimum surface clearance (m)")
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(campaign_dir / "clearance.png", dpi=120)
    plt.close(figure)
