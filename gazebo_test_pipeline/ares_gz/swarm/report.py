"""Swarm campaign report: REPORT.md, all_runs.csv and a map plot per run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

COLUMNS = ["scenario", "seed", "mission_success", "termination_reason", "end_time_s", "min_pairwise_distance_m",
           "separation_invariant_held", "coverage", "victims_found", "victims_confirmed", "victims_total",
           "worst_active_clearance_m", "flag_latency_s", "quarantine_latency_s", "comms", "wind", "lidar_mode",
           "faulty", "outage_sync", "directory"]


def _row(summary: dict) -> dict:
    drones = summary.get("drones", {})
    active = [d for d in drones.values() if d.get("final_mode") not in ("QUARANTINED", "LOST")]
    clearances = [d.get("min_building_clearance_m") for d in active if d.get("min_building_clearance_m") is not None]
    detection = summary.get("failure_detection") or {}
    flags = [v["flagged_after_s"] for v in detection.values() if v.get("flagged_after_s") is not None]
    quarantines = [v["quarantined_after_s"] for v in detection.values() if v.get("quarantined_after_s") is not None]
    row = {key: summary.get(key) for key in COLUMNS}
    row.update(worst_active_clearance_m=min(clearances) if clearances else None,
               flag_latency_s=max(flags) if flags else None,
               quarantine_latency_s=max(quarantines) if quarantines else None,
               faulty=",".join(summary["faulty"]) if isinstance(summary.get("faulty"), list) else summary.get("faulty"))
    return row


def plot_run(run_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    from ..scene import scene_map
    from .mapping import FREE_BELOW, OCCUPIED_ABOVE, Grid

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 8))
    map_file = run_dir / "shared_map_logodds.npy"
    if map_file.exists():
        grid = Grid()
        value = np.load(map_file)
        image = np.full(value.shape, 0.5)
        image[value < FREE_BELOW] = 1.0
        image[value > OCCUPIED_ABOVE] = 0.0
        ax.imshow(image, origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0.35,
                  extent=(grid.east[0] - 0.25, grid.east[-1] + 0.25, grid.north[0] - 0.25, grid.north[-1] + 0.25))
    scene = scene_map()
    for lo, hi in zip(scene.box_min, scene.box_max):
        ax.add_patch(plt.Rectangle((lo[1], lo[0]), hi[1] - lo[1], hi[0] - lo[0], color="0.45"))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for k, drone in enumerate(manifest["drones"]):
        path = run_dir / "drones" / drone["id"] / "control.csv"
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        n = np.array([float(r["n"]) for r in rows])
        e = np.array([float(r["e"]) for r in rows])
        ok = np.array([r["lidar_available"] == "1" for r in rows])
        ax.plot(e, n, color=colors[k % 4], lw=1.8, label=f"{drone['id']}")
        ax.scatter(e[~ok][::20], n[~ok][::20], s=6, color="red", zorder=3)
        ax.plot(drone["east"], drone["north"], "o", color=colors[k % 4])
        ax.plot(drone["slot"][1], drone["slot"][0], "s", mfc="none", color=colors[k % 4], ms=10)
    for victim in manifest["victims"]:
        ax.plot(victim["east"], victim["north"], "*", color="crimson", ms=16)
        ax.annotate(victim["id"], (victim["east"] + 0.6, victim["north"] + 0.6))
    ax.set_xlim(-5, 40)
    ax.set_ylim(-5, 40)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")
    verdict = "PASS" if summary.get("mission_success") else "FAIL"
    ax.set_title(f"{summary['scenario']} seed {summary['seed']}: {verdict}\n"
                 f"min separation {summary['min_pairwise_distance_m']:.2f} m, coverage {summary['coverage']:.0%}, "
                 f"victims {summary['victims_confirmed']}/{summary['victims_total']} (red dots: LiDAR down)",
                 fontsize=10)
    ax.legend(loc="lower right")
    out = run_dir / "swarm_map.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(campaign_dir: Path) -> Path:
    campaign_dir = Path(campaign_dir).resolve()
    summaries = []
    for path in sorted(campaign_dir.rglob("summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("schema", "").startswith("ares-gz-swarm"):
            summaries.append(summary)
            if not (path.parent / "swarm_map.png").exists():
                plot_run(path.parent)
    rows = [_row(s) for s in summaries]
    with (campaign_dir / "all_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, digits=2):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [f"# Swarm Gazebo report: `{campaign_dir.name}`", "",
             "Pass = no crash, no two drones ever closer than the separation rule, every trusted drone keeps 1.0 m "
             "from buildings, all trusted drones reach their rendezvous slot, and every victim is confirmed by a "
             "drone whose LiDAR was working.", "",
             "| Scenario | Seed | Result | Min separation (m) | Coverage | Victims confirmed | First flag / quarantine (s after first outage) "
             "| Worst trusted clearance (m) | End (s) | Notes |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['seed']} | {'PASS' if row['mission_success'] else 'FAIL'} "
            f"| {fmt(row['min_pairwise_distance_m'])} | {fmt((row['coverage'] or 0) * 100, 0)}% "
            f"| {row['victims_confirmed']}/{row['victims_total']} "
            f"| {fmt(row['flag_latency_s'], 1)} / {fmt(row['quarantine_latency_s'], 1)} "
            f"| {fmt(row['worst_active_clearance_m'])} | {fmt(row['end_time_s'], 1)} | {row['termination_reason']} |")
    lines += ["", "Per-run detail: `summary.json`, `hub_events.json` (every hub decision with its reason), "
              "`trust.csv` (trust evidence at 10 Hz), `swarm_map.png`, `drones/<UAV>/control.csv`."]
    report = campaign_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    return report
