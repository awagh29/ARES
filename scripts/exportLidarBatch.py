#!/usr/bin/env python3
"""Export finalized ARES MATLAB episodes to immutable, Python-ready NPZ files.

Dependencies: numpy, h5py. Usage: python exportLidarBatch.py --batch PATH
The MATLAB finalizer must have produced dataset_qa.json with
allDataChecksPassed=true. MATLAB R2026a string fields use MCOS storage, so run
identity is established by that finalizer and checked against each raw manifest.
Numeric v7.3 arrays are transposed back into MATLAB's sample-by-channel order.

Quick load:
    import numpy as np
    with np.load("episode.npz", allow_pickle=False) as episode:
        time_s = episode["time_s"]
        ranges = episode["lidar_ranges_m"]  # scans x 10740
        valid = episode["lidar_valid_return_mask"]
        position = episode["position_ned_m"]  # scans x 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import tempfile

import h5py
import numpy as np


SCHEMA = "ares-python-lidar-episode-v1"
VECTOR_FIELDS = {
    "time_s": "time_s",
    "sample_index": "sampleIndex",
    "lidar_enabled": "lidar/enabled",
    "previous_action_available": "action/previousAvailable",
}
MATRIX_FIELDS = {
    "lidar_ranges_m": ("lidar/ranges_m", 10740),
    "lidar_valid_return_mask": ("lidar/validReturnMask", 10740),
    "lidar_availability_mask": ("lidar/availabilityMask", 10740),
    "lidar_missing_mask": ("lidar/missingMask", 10740),
    "position_ned_m": ("state/positionNed_m", 3),
    "velocity_ned_mps": ("state/velocityNed_mps", 3),
    "attitude_ned_rpy_rad": ("state/attitudeNedRpy_rad", 3),
    "angular_velocity_body_radps": ("state/angularVelocityBody_radps", 3),
    "action_current": ("action/current", 4),
    "action_previous": ("action/previous", 4),
    "goal_position_ned_m": ("goal/positionNed_m", 3),
    "reference_desired_position_ned_m": ("reference/desiredPositionNed_m", 3),
}
BOOLEAN_FIELDS = {
    "lidar_enabled", "previous_action_available", "lidar_valid_return_mask",
    "lidar_availability_mask", "lidar_missing_mask",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_numeric(episode: h5py.Group, relative: str) -> np.ndarray:
    item = episode[relative]
    require(isinstance(item, h5py.Dataset), f"Not a numeric dataset: {relative}")
    result = np.asarray(item[()])
    require(result.dtype.kind in "biuf", f"Unsupported numeric dtype: {relative}")
    # MATLAB v7.3/HDF5 dimension ordering is reversed. Do not squeeze matrices.
    return np.ascontiguousarray(result.transpose(tuple(reversed(range(result.ndim)))))


def load_episode(path: Path, row: dict[str, str], manifest: dict) -> dict[str, np.ndarray]:
    run_id = row["RunId"]
    require(str(manifest.get("runId", "")) == run_id,
            f"Index/manifest run ID mismatch: {run_id}")
    require(path.parent.name == run_id, f"Episode folder/run ID mismatch: {path}")
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as source:
        episode = source["episode"]
        for key, relative in VECTOR_FIELDS.items():
            value = read_numeric(episode, relative)
            require(value.ndim <= 2 and (value.ndim < 2 or 1 in value.shape),
                    f"Expected vector: {relative}: {value.shape}")
            arrays[key] = value.reshape(-1)
        count = arrays["time_s"].size
        require(count >= 2, f"Insufficient scans: {run_id}")
        require(count == int(row["ScanCount"]), f"Index scan-count mismatch: {run_id}")
        for key, (relative, width) in MATRIX_FIELDS.items():
            value = read_numeric(episode, relative)
            require(value.shape == (count, width),
                    f"Unexpected {relative} shape {value.shape}; expected {(count, width)}")
            arrays[key] = value
        if "obstacleAvoidanceStatus" in episode:
            arrays["obstacle_avoidance_status"] = read_numeric(
                episode, "obstacleAvoidanceStatus").reshape(-1)

    for key, value in arrays.items():
        require(value.shape[0] == count, f"Unaligned samples: {key}")
        if key in BOOLEAN_FIELDS:
            require(np.isin(value, (0, 1)).all(), f"Nonbinary values: {key}")
            arrays[key] = value.astype(np.bool_, copy=False)
        elif key != "action_previous":
            require(np.isfinite(value).all(), f"Nonfinite values: {key}")

    require(np.all(np.diff(arrays["time_s"]) > 0), f"Nonincreasing scan time: {run_id}")
    require(np.allclose(np.diff(arrays["time_s"]), 0.1, rtol=0, atol=1e-6),
            f"Unexpected scan cadence: {run_id}")
    require(np.array_equal(arrays["sample_index"], np.arange(1, count + 1)),
            f"Unexpected sample indices: {run_id}")
    require(arrays["lidar_enabled"].all(), f"LiDAR outage: {run_id}")
    require(arrays["lidar_availability_mask"].all()
            and not arrays["lidar_missing_mask"].any(), f"Missing observations: {run_id}")
    valid = arrays["lidar_valid_return_mask"]
    ranges = arrays["lidar_ranges_m"]
    maximum_range = float(manifest["static"]["maximum_sensor_range_m"])
    require(np.all((ranges >= 0) & (ranges <= maximum_range)),
            f"Out-of-range LiDAR values: {run_id}")
    require(np.all(ranges[valid] > 0) and np.all(ranges[~valid] == maximum_range),
            f"Invalid no-return representation: {run_id}")
    available = arrays["previous_action_available"]
    require(not available[0] and available[1:].all(),
            f"Unexpected previous-action availability: {run_id}")
    previous = arrays["action_previous"]
    require(np.isnan(previous[0]).all() and np.isfinite(previous[1:]).all(),
            f"Unexpected previous-action NaN values: {run_id}")
    require(np.array_equal(previous[1:], arrays["action_current"][:-1]),
            f"Previous actions do not match preceding scans: {run_id}")
    final_goal = np.asarray(manifest["configuration"]["mission"]["waypointsNed"])[-1]
    require(np.array_equal(arrays["goal_position_ned_m"],
                           np.broadcast_to(final_goal, (count, 3))),
            f"Episode goal differs from final mission goal: {run_id}")
    metadata = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "source_episode_path": str(path),
        "source_episode_sha256": sha256(path),
        "run_identity_validation": "MATLAB dataset_qa and matching index/raw manifest",
        "coordinate_frame": "NED",
        "action_components": ["roll_rad", "pitch_rad", "yaw_rate_radps", "thrust_n"],
        "attitude_components": ["roll", "pitch", "yaw"],
        "angular_velocity_frame": "body",
        "lidar_grid_size": [15, 716],
        "lidar_channel_order": "elevation-index-fastest-then-azimuth-index",
        "maximum_lidar_range_m": maximum_range,
        "previous_action_definition": "action at previous LiDAR scan; first unavailable",
        "goal_definition": "fixed final mission waypoint",
        "reference_definition": "moving obstacle-avoidance controller reference",
    }
    arrays.update({
        "run_id": np.asarray(run_id),
        "schema_version": np.asarray(SCHEMA),
        "source_episode_path": np.asarray(str(path)),
        "source_episode_sha256": np.asarray(metadata["source_episode_sha256"]),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    })
    return arrays


def arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    if first.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def verify_npz(path: Path, expected: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), f"Existing NPZ field mismatch: {path}")
        for key, value in expected.items():
            require(arrays_equal(saved[key], value), f"NPZ read-back mismatch: {path}: {key}")


def install_exclusive(temporary: Path, destination: Path) -> None:
    # A hard link atomically installs a fully written file and refuses to replace
    # an existing name. Both files are on the same filesystem by construction.
    os.link(temporary, destination)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    if path.exists():
        verify_npz(path, arrays)
        return False
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=".episode-", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        verify_npz(temporary, arrays)
        try:
            install_exclusive(temporary, path)
        except FileExistsError:
            verify_npz(path, arrays)
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_immutable_text(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        require(path.read_bytes() == encoded, f"Existing export report differs: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".export-", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            install_exclusive(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == encoded, f"Concurrent export report differs: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_batch(batch: Path, allow_partial_test: bool = False) -> dict:
    batch = batch.resolve(strict=True)
    qa_path = batch / "dataset_qa.json"
    require(qa_path.is_file(), f"Missing MATLAB finalizer report: {qa_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8-sig"))
    require(qa.get("allDataChecksPassed") is True,
            "MATLAB finalizer data checks have not passed; export refused.")
    index_path = batch / "index.csv"
    with index_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    require("NpzPath" not in fields, "Source index already contains NpzPath.")
    completed = [row for row in rows if row.get("Status") == "complete"]
    require(completed, "Batch has no completed runs.")
    if not allow_partial_test:
        require(len(rows) == len(completed) == 80, "Expected 80 completed runs.")
        require(sum(row["Axis"] == "North" for row in rows) == 40
                and sum(row["Axis"] == "East" for row in rows) == 40,
                "Expected 40 completed runs on each axis.")
    ids = [row["RunId"] for row in completed]
    require(len(set(ids)) == len(ids) and all(ids), "Run IDs must be unique and nonempty.")
    require(len(set(row["RunIndex"] for row in rows)) == len(rows), "Duplicate run indices.")
    report = {
        "schemaVersion": "ares-python-export-qa-v1",
        "batchPath": str(batch),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "h5py": h5py.__version__},
        "exporterSha256": sha256(Path(__file__)),
        "sourceIndexSha256": sha256(index_path),
        "sourceDatasetQaSha256": sha256(qa_path),
        "partialTestExport": allow_partial_test,
        "completedRuns": len(completed),
        "exportedRuns": 0,
        "totalScans": 0,
        "allExportChecksPassed": False,
        "files": [],
    }
    created = 0
    for row in rows:
        row["NpzPath"] = ""
        if row["Status"] != "complete":
            continue
        episode_path = Path(row["EpisodePath"]).resolve(strict=True)
        raw_path = Path(row["RawTelemetryPath"]).resolve(strict=True)
        manifest_path = raw_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        arrays = load_episode(episode_path, row, manifest)
        destination = episode_path.parent / "episode.npz"
        created += int(save_npz(destination, arrays))
        row["NpzPath"] = str(destination)
        report["files"].append({
            "runId": row["RunId"], "npzPath": str(destination),
            "npzSha256": sha256(destination),
            "sourceEpisodeSha256": str(arrays["source_episode_sha256"].item()),
            "scanCount": int(arrays["time_s"].size),
        })
        report["exportedRuns"] += 1
        report["totalScans"] += int(arrays["time_s"].size)
        print(f"EXPORTED {row['RunIndex']} {row['RunId']} scans={arrays['time_s'].size}",
              flush=True)
        del arrays
    report["allExportChecksPassed"] = True
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields + ["NpzPath"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    save_immutable_text(batch / "python_index.csv", text.getvalue())
    save_immutable_text(batch / "python_export_qa.json",
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"PYTHON_EXPORT_COMPLETE runs={report['exportedRuns']} "
          f"scans={report['totalScans']} created={created} verified_existing="
          f"{report['exportedRuns'] - created}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path,
                        help="Finalized data/batches/UUID folder")
    parser.add_argument("--allow-partial-test", action="store_true",
                        help="Permit fewer than 80 runs for explicit testing; finalizer QA is still required")
    args = parser.parse_args()
    export_batch(args.batch, args.allow_partial_test)


if __name__ == "__main__":
    main()
