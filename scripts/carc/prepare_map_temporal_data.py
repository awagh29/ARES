#!/usr/bin/env python3
"""Prepare portable NPZ inputs from a clone, including raw-only MATLAB 7.3 data.

Existing checked exports are reused. Rebuilt inputs and their NEW checksum audit
go under run_root/exports/<batch>; tracked data/reports are never rewritten.
Dependencies: numpy; h5py only when raw telemetry must be converted. No MATLAB.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

DEFAULT_BATCH = "c3eaab07-d882-4825-98a5-84c278cf8935"
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check_exports(episode_root, audit_path, expected_runs=80):
    """Ignore historical absolute paths in reports; resolve IDs in this checkout."""
    episode_root, audit_path = Path(episode_root), Path(audit_path)
    if not audit_path.is_file():
        return False
    report = read_json(audit_path)
    require(report.get("allExportChecksPassed") is True and report.get("partialTestExport") is False,
            f"Export audit did not pass: {audit_path}")
    entries = report.get("files", [])
    require(len(entries) == expected_runs and len({e["runId"] for e in entries}) == expected_runs,
            f"Expected {expected_runs} unique recorded exports: {audit_path}")
    for item in entries:
        run_id = item["runId"]
        require(Path(run_id).name == run_id and run_id not in (".", ".."), "Invalid run ID.")
        path = episode_root/run_id/"episode.npz"
        if not path.is_file():
            return False
        require(sha256(path) == item["npzSha256"], f"NPZ checksum mismatch (or an undownloaded Git LFS pointer): {path}")
    return True


def numeric(group, key):
    """Reverse HDF5 dimensions to MATLAB's native sample-by-channel matrices."""
    value = np.asarray(group[key][()])
    require(value.dtype.kind in "biuf", f"Expected numeric telemetry: {group.name}/{key}")
    return value.transpose(tuple(reversed(range(value.ndim))))


def read_signal(signals, name, width=None):
    group = signals[name]
    times = numeric(group, "time_s").reshape(-1).astype(np.float64)
    values = numeric(group, "values").astype(np.float64)
    require(len(times) and np.isfinite(times).all() and (np.diff(times) >= 0).all(), f"Invalid timeline: {name}")
    require(values.ndim == 2 and values.shape[0] == len(times) and np.isfinite(values).all()
            and (width is None or values.shape[1] == width), f"Invalid signal values/width: {name}")
    return times, values


def sample_signal(signals, name, query, width, method="linear"):
    """Match buildLidarEpisode.m: duplicate-last, no extrapolation, angle unwrap."""
    times, values = read_signal(signals, name, width)
    unique, reversed_index = np.unique(times[::-1], return_index=True)
    values = values[len(times)-1-reversed_index]
    require(unique[0]-query[0] <= 1e-9 and query[-1]-unique[-1] <= 1e-9,
            f"Signal does not cover scan timeline (no extrapolation): {name}")
    clipped = query.clip(unique[0], unique[-1])
    if method == "previous":
        return values[np.searchsorted(unique, clipped, side="right")-1]
    if method == "angular":
        values = np.unwrap(values, axis=0)
    output = np.column_stack([np.interp(clipped, unique, values[:, column]) for column in range(width)])
    return np.arctan2(np.sin(output), np.cos(output)) if method == "angular" else output


def convert_raw(path, manifest, row):
    import h5py
    path = Path(path)
    with path.open("rb") as stream:
        require(not stream.read(64).startswith(b"version https://git-lfs.github.com/spec"),
                f"Raw MAT is a Git LFS pointer. Run git lfs pull first: {path}")
    run_id = row["RunId"]
    require(manifest["runId"] == run_id and path.parent.name == run_id, "Raw manifest/folder identity mismatch.")
    maximum = float(manifest["static"]["maximum_sensor_range_m"])
    require(maximum == 7. and manifest["static"]["lidar_grid_size"] == [15, 716], "Wrong raw LiDAR grid/range.")
    with h5py.File(path, "r") as source:
        telemetry = source["telemetry"]
        if "experiment/runIndex" in telemetry:
            require(int(numeric(telemetry, "experiment/runIndex").item()) == int(row["RunIndex"]), "Raw telemetry/index run number differs.")
        require(numeric(telemetry, "lidarMetadata/gridSize").reshape(-1).tolist() == [15., 716.], "Wrong raw grid size.")
        require(float(numeric(telemetry, "lidarMetadata/maximumRangeM").item()) == maximum, "Raw range metadata differs.")
        signals = telemetry["signals"]
        times, ranges = read_signal(signals, "nominal_observation", 10740)
        count = len(times)
        require(count == int(row["ScanCount"]) and count >= 2 and np.allclose(np.diff(times), .1, atol=1e-6, rtol=0), "Raw scan count/cadence mismatch.")
        valid_time, valid = read_signal(signals, "nominal_lidar_valid_return_mask", 10740)
        require(np.array_equal(valid_time, times) and np.isin(valid, [0, 1]).all(), "Misaligned raw return mask.")
        valid = valid.astype(bool)
        require(((ranges >= 0) & (ranges <= maximum)).all() and (ranges[valid] > 0).all()
                and (ranges[~valid] == maximum).all(), "Invalid clean raw ranges.")
        if "nominal_lidar_ranges_m" in signals:
            range_time, redundant = read_signal(signals, "nominal_lidar_ranges_m", 10740)
            require(np.array_equal(range_time, times) and np.array_equal(redundant, ranges), "Raw nominal range aliases differ.")
        require((read_signal(signals, "lidar_enabled", 1)[1] == 1).all(), "Source contains a LiDAR outage.")
        for name, width, expected in (("observation_availability_mask", 10740, 1),
                                      ("missing_observation_mask", 10740, 0), ("lidar_failure_active", 1, 0)):
            if name in signals:
                mask_time, mask = read_signal(signals, name, width)
                require(np.array_equal(mask_time, times) and (mask == expected).all(), f"Unhealthy raw mask: {name}")
        arrays = dict(time_s=times, sample_index=np.arange(1, count+1, dtype=np.uint32),
            lidar_ranges_m=ranges, lidar_valid_return_mask=valid,
            lidar_availability_mask=np.ones_like(valid), lidar_missing_mask=np.zeros_like(valid),
            lidar_enabled=np.ones(count, bool))
        for name, method in (("position_ned_m", "linear"), ("velocity_ned_mps", "linear"),
                             ("attitude_ned_rpy_rad", "angular"), ("angular_velocity_body_radps", "linear")):
            arrays[name] = sample_signal(signals, name, times, 3, method)
        arrays["reference_desired_position_ned_m"] = sample_signal(signals, "desired_position_ned_m", times, 3, "previous")
        goal = numeric(telemetry, "mission/finalGoalPositionNed_m").reshape(-1)
        require(goal.shape == (3,) and np.isfinite(goal).all() and np.array_equal(goal, np.asarray(manifest["configuration"]["mission"]["waypointsNed"])[-1]), "Raw mission goal mismatch.")
        arrays["goal_position_ned_m"] = np.tile(goal, (count, 1))
        commands = ["command_roll_rad", "command_pitch_rad", "command_yaw_rate_radps", "command_thrust_n"]
        arrays["action_current"] = np.column_stack([sample_signal(signals, key, times, 1, "previous")[:, 0] for key in commands])
        arrays["action_previous"] = np.vstack((np.full((1, 4), np.nan), arrays["action_current"][:-1]))
        arrays["previous_action_available"] = np.r_[False, np.ones(count-1, bool)]
        if "obstacle_avoidance_status" in signals:
            arrays["obstacle_avoidance_status"] = sample_signal(signals, "obstacle_avoidance_status", times, 1, "previous")[:, 0]
    metadata = dict(lidar_grid_size=[15, 716], lidar_channel_order="elevation-index-fastest-then-azimuth-index",
        coordinate_frame="NED", source_telemetry_sha256=sha256(path),
        state_interpolation="linear; angles unwrapped then wrapped; duplicate timestamps keep last; no extrapolation",
        exporter="carc-raw-to-python-v1", schema_version="ares-python-lidar-episode-v1")
    arrays.update(run_id=np.asarray(run_id), schema_version=np.asarray("ares-python-lidar-episode-v1"),
                  metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    return arrays


def save_episode(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == set(arrays) and all(np.array_equal(z[k], v, equal_nan=True) if v.dtype.kind in "fc"
                    else np.array_equal(z[k], v) for k, v in arrays.items()), f"Existing rebuilt export differs; use a fresh run root: {path}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.link(temporary, path)  # atomically install, refusing to overwrite existing data
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(repo_root, run_root, batch_id=DEFAULT_BATCH):
    repo_root, run_root = Path(repo_root).resolve(), Path(run_root).resolve()
    require(Path(batch_id).name == batch_id and batch_id not in (".", ".."), "Invalid batch folder name.")
    batch = repo_root/"data"/"batches"/batch_id
    destination = run_root/"exports"/batch_id
    for episode_root, audit in ((destination/"lidar_episode_v1", destination/"python_export_qa.json"),
                                (repo_root/"data/processed/lidar_episode_v1", batch/"python_export_qa.json")):
        if check_exports(episode_root, audit):
            print("Reusing verified recorded exports:", episode_root, flush=True)
            return episode_root, audit
    require((batch/"dataset_qa.json").is_file() and read_json(batch/"dataset_qa.json").get("allDataChecksPassed") is True,
            "Missing/passing MATLAB dataset_qa.json is required. Include data/batches in the clone.")
    with (batch/"index.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 80 and all(r["Status"] == "complete" for r in rows)
            and len({r["RunId"] for r in rows}) == 80, "Need the 80-run finalized batch index.")
    report = dict(schemaVersion="ares-carc-raw-export-qa-v1", allExportChecksPassed=False,
        partialTestExport=False, batchPath=str(batch), sourceIndexSha256=sha256(batch/"index.csv"),
        sourceDatasetQaSha256=sha256(batch/"dataset_qa.json"), exporterSha256=sha256(__file__), files=[])
    for i, row in enumerate(rows):
        run_id = row["RunId"]
        require(Path(run_id).name == run_id and run_id not in (".", ".."), "Invalid run ID.")
        raw = repo_root/"data/raw"/run_id
        candidates = sorted(raw.glob("*_telemetry.mat"))
        require(len(candidates) == 1, f"Expected one raw telemetry MAT for {run_id}: {candidates}")
        manifest_path = raw/"manifest.json"
        manifest = read_json(manifest_path)
        require(manifest["experiment"]["batchId"] == batch_id, "Raw manifest belongs to another batch.")
        arrays = convert_raw(candidates[0], manifest, row)
        path = destination/"lidar_episode_v1"/run_id/"episode.npz"
        save_episode(path, arrays)
        report["files"].append(dict(runId=run_id, scanCount=len(arrays["time_s"]), npzSha256=sha256(path),
                                    sourceTelemetrySha256=sha256(candidates[0]), sourceManifestSha256=sha256(manifest_path)))
        print(f"Prepared recorded flight {i+1}/80: {run_id}", flush=True)
    report.update(allExportChecksPassed=True, exportedRuns=80, totalScans=sum(e["scanCount"] for e in report["files"]))
    destination.mkdir(parents=True, exist_ok=True)
    audit = destination/"python_export_qa.json"
    temporary = audit.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    os.replace(temporary, audit)
    require(check_exports(destination/"lidar_episode_v1", audit), "Prepared export verification failed.")
    return destination/"lidar_episode_v1", audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("ARES_RUN_ROOT", ROOT)))
    parser.add_argument("--batch-id", default=DEFAULT_BATCH)
    args = parser.parse_args()
    episode_root, audit = prepare(args.repo_root, args.run_root, args.batch_id)
    print("Episode root:", episode_root)
    print("Export audit:", audit)


if __name__ == "__main__":
    main()
