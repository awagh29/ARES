#!/usr/bin/env python3
"""Read the NPZ episodes produced by exportLidarBatch.py in Colab or locally.

Dependencies: numpy, pandas. No MATLAB, h5py, GPU, or training framework needed.
Use an existing repository checkout; stored Windows paths are provenance only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_BATCH_ID = "c3eaab07-d882-4825-98a5-84c278cf8935"
SCHEMA = "ares-python-lidar-episode-v1"
GRID_SIZE = (15, 716)
MAX_RANGE_M = 7.0
MATRIX_WIDTHS = {
    "lidar_ranges_m": 10740,
    "lidar_valid_return_mask": 10740,
    "lidar_availability_mask": 10740,
    "lidar_missing_mask": 10740,
    "position_ned_m": 3,
    "velocity_ned_mps": 3,
    "attitude_ned_rpy_rad": 3,
    "angular_velocity_body_radps": 3,
    "action_current": 4,
    "action_previous": 4,
    "goal_position_ned_m": 3,
    "reference_desired_position_ned_m": 3,
}
VECTOR_FIELDS = ("time_s", "sample_index", "lidar_enabled", "previous_action_available")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeDataset:
    """Keep a small run index in memory and load one episode at a time.

    ``runs`` includes every instrumentally valid run, including unsafe flights.
    Paths are reconstructed inside repo_root using RunId, so Linux/Colab never
    tries to open the original A:\\Projects\\ares paths. Input files are read only.
    """

    def __init__(self, repo_root: str | Path, batch_id: str = DEFAULT_BATCH_ID,
                 *, verify_hashes: bool = True, expected_runs: int = 80):
        self.repo_root = Path(repo_root).expanduser().resolve()
        _require(Path(batch_id).name == batch_id and batch_id not in (".", ".."),
                 "batch_id must be a single batch folder name.")
        self.batch_path = self.repo_root / "data" / "batches" / batch_id
        self.verify_hashes = verify_hashes
        index_path = self.batch_path / "python_index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Missing {index_path}. Set repo_root to your existing checkout and "
                "ensure the batch data, not only the notebook, is available there.")
        self.qa = json.loads((self.batch_path / "dataset_qa.json").read_text(encoding="utf-8-sig"))
        self.export_qa = json.loads(
            (self.batch_path / "python_export_qa.json").read_text(encoding="utf-8-sig"))
        _require(self.qa.get("allDataChecksPassed") is True,
                 "The MATLAB dataset audit has not passed.")
        _require(self.export_qa.get("allExportChecksPassed") is True
                 and self.export_qa.get("partialTestExport") is False,
                 "A verified, complete Python export is required.")
        self.runs = pd.read_csv(index_path, dtype={"RunId": str, "Axis": str, "Status": str})
        required = {"RunIndex", "RunId", "Axis", "Status", "ScanCount",
                    "MissionSuccess", "CollisionOccurred", "SafetyChecksPassed"}
        _require(required.issubset(self.runs.columns), "The run index lacks required columns.")
        _require(len(self.runs) == expected_runs and self.runs["Status"].eq("complete").all(),
                 f"Expected {expected_runs} completed runs.")
        _require(self.runs["RunId"].notna().all() and self.runs["RunId"].is_unique
                 and self.runs["RunId"].str.fullmatch(r"[a-zA-Z0-9-]+").all(),
                 "Run IDs must be unique, nonempty folder identifiers.")
        _require(self.runs["RunIndex"].is_unique
                 and set(self.runs["RunIndex"]) == set(range(1, expected_runs + 1)),
                 "Run indices must uniquely cover the planned run numbers.")
        counts = self.runs["Axis"].value_counts().to_dict()
        _require(counts == {"North": expected_runs // 2, "East": expected_runs // 2},
                 "Expected equal North-axis and East-axis groups.")
        _require(np.isfinite(self.runs["ScanCount"]).all()
                 and self.runs["ScanCount"].ge(2).all()
                 and self.runs["ScanCount"].eq(self.runs["ScanCount"].astype(int)).all(),
                 "Every run must have an integer scan count of at least two.")
        for name in ("MissionSuccess", "CollisionOccurred", "SafetyChecksPassed"):
            _require(self.runs[name].isin([0, 1]).all(), f"Invalid outcome column: {name}")
            self.runs[name] = self.runs[name].astype(bool)
        self.runs = self.runs.sort_values("RunIndex").reset_index(drop=True)
        reports = self.export_qa.get("files", [])
        self._reports = {item["runId"]: item for item in reports}
        _require(len(reports) == len(self._reports) == expected_runs
                 and set(self._reports) == set(self.runs["RunId"]),
                 "The export report and index disagree on run IDs.")
        for row in self.runs.itertuples():
            _require(int(self._reports[row.RunId]["scanCount"]) == row.ScanCount,
                     f"Index/export report scan-count mismatch: {row.RunId}")
        self.runs["NpzPath"] = [
            str(self.repo_root / "data" / "processed" / "lidar_episode_v1" / run_id / "episode.npz")
            for run_id in self.runs["RunId"]
        ]
        missing = [path for path in self.runs["NpzPath"] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} episode files are missing from this checkout. First: {missing[0]}. "
                "Make the exported data available with its original folder layout; "
                "this loader does not rerun simulations or modify the index.")
        self._rows = self.runs.set_index("RunId")
        total = int(self.runs["ScanCount"].sum())
        _require(total == self.qa.get("totalScans") == self.export_qa.get("totalScans"),
                 "Dataset reports disagree on the total scan count.")

    def load_episode(self, run_id: str, fields: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """Return ordinary NumPy arrays with the NPZ handle already closed.

        With fields=None, load all arrays for this ONE run. Pass a list of fields
        to avoid decompressing unused large arrays. IDs and time are always kept.
        Previous-action NaNs are intentional; masks and native values are retained.
        """
        if run_id not in self._rows.index:
            raise KeyError(f"Unknown run ID: {run_id}")
        _require(not isinstance(fields, str), "Pass fields as a list or tuple of field names.")
        row = self._rows.loc[run_id]
        path = Path(row["NpzPath"])
        with path.open("rb") as stream:
            header = stream.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer. Retrieve the actual data file in the checkout.")
        if self.verify_hashes:
            _require(_sha256(path) == self._reports[run_id]["npzSha256"],
                     f"Episode checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = set(MATRIX_WIDTHS) | set(VECTOR_FIELDS) | {"run_id", "schema_version", "metadata_json"}
            _require(required.issubset(archive.files), f"Episode schema is incomplete: {run_id}")
            selected = set(archive.files) if fields is None else set(fields)
            selected |= {"run_id", "schema_version", "metadata_json", "time_s"}
            _require(selected.issubset(archive.files), f"Unknown episode fields: {selected - set(archive.files)}")
            arrays = {key: archive[key] for key in sorted(selected)}
        _require(str(arrays["run_id"].item()) == run_id
                 and str(arrays["schema_version"].item()) == SCHEMA,
                 f"Episode ID/schema mismatch: {run_id}")
        count = int(row["ScanCount"])
        time = arrays["time_s"]
        _require(time.shape == (count,) and np.isfinite(time).all()
                 and np.allclose(np.diff(time), 0.1, rtol=0, atol=1e-6),
                 f"Invalid 10 Hz timeline: {run_id}")
        for name, width in MATRIX_WIDTHS.items():
            if name in arrays:
                _require(arrays[name].shape == (count, width), f"Invalid shape: {run_id}/{name}")
        for name in VECTOR_FIELDS:
            if name in arrays:
                _require(arrays[name].shape == (count,), f"Invalid vector: {run_id}/{name}")
        return arrays

    def iter_episodes(self, run_ids: Iterable[str] | None = None,
                      fields: Iterable[str] | None = None):
        """Stream runs; do not turn this iterator into a list of all LiDAR arrays."""
        selected = self.runs["RunId"] if run_ids is None else run_ids
        requested = None if fields is None else tuple(fields)
        for run_id in selected:
            yield run_id, self.load_episode(run_id, requested)

    def load_scan_table(self) -> pd.DataFrame:
        """Load the compact EDA table, which contains summaries rather than all beams."""
        scans = pd.read_csv(self.batch_path / "all_scans.csv", dtype={"RunId": str})
        _require({"RunId", "SampleIndex", "TimeS", "Axis"}.issubset(scans.columns),
                 "The scan table lacks its run/sample/time columns.")
        counts = scans.groupby("RunId").size().to_dict()
        _require(counts == self._rows["ScanCount"].to_dict(),
                 "The scan table does not match the run index.")
        for run_id, group in scans.groupby("RunId", sort=False):
            time = group["TimeS"].to_numpy()
            _require(np.array_equal(group["SampleIndex"], np.arange(1, len(group) + 1))
                     and np.isfinite(time).all()
                     and np.allclose(np.diff(time), 0.1, rtol=0, atol=1e-6)
                     and group["Axis"].eq(self._rows.loc[run_id, "Axis"]).all(),
                     f"Scan indices/timestamps/axis are inconsistent: {run_id}")
        return scans


def make_run_splits(runs: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Seeded 70/10/10/10 train/validation/calibration/test splits within each axis.

    For the 80-run batch this gives 56/8/8/8 runs, with both axes in each set.
    Split before windows, scalers, synthetic missingness, or hyperparameter tuning.
    Safety outcomes are retained and are not used to select or shuffle the runs.
    """
    _require(runs["RunId"].notna().all() and runs["RunId"].is_unique,
             "Split whole, uniquely identified runs.")
    _require(set(runs["Axis"]) == {"North", "East"}, "Expected North and East groups.")
    names = ("train", "validation", "calibration", "test")
    fractions = np.asarray([0.7, 0.1, 0.1, 0.1])
    rng = np.random.default_rng(seed)
    assignment = {}
    for axis in ("North", "East"):
        ids = np.asarray(sorted(runs.loc[runs["Axis"].eq(axis), "RunId"]), dtype=str)
        ids = rng.permutation(ids)
        exact = len(ids) * fractions
        counts = np.floor(exact).astype(int)
        remainder = len(ids) - int(counts.sum())
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1
        _require((counts > 0).all(), "Each axis needs enough runs for four nonempty splits.")
        start = 0
        for name, count in zip(names, counts):
            assignment.update({run_id: name for run_id in ids[start:start + count]})
            start += int(count)
    split = runs[["RunId", "Axis"]].copy()
    split["Split"] = split["RunId"].map(assignment)
    return split


def lidar_grid(episode: dict[str, np.ndarray], scan_index: int = 0,
               *, mask_no_returns: bool = True) -> np.ndarray:
    """A [15,716] elevation-by-azimuth scan; original ranges remain unchanged.

    MATLAB elevation-fastest channel ordering needs order='F'. NaNs in this
    returned plotting copy mean no return, not a sensor outage or missing beam.
    """
    grid = episode["lidar_ranges_m"][scan_index].reshape(GRID_SIZE, order="F").copy()
    if mask_no_returns:
        valid = episode["lidar_valid_return_mask"][scan_index].reshape(GRID_SIZE, order="F")
        grid[~valid] = np.nan
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    args = parser.parse_args()
    dataset = EpisodeDataset(args.repo_root, args.batch_id)
    print(f"Loaded index: {len(dataset.runs)} runs, {dataset.runs.ScanCount.sum():,} scans")
    print(f"Safety checks passed: {dataset.runs.SafetyChecksPassed.sum()}; "
          f"sampled collisions: {dataset.runs.CollisionOccurred.sum()}")
    splits = make_run_splits(dataset.runs)
    print(pd.crosstab(splits["Split"], splits["Axis"]).to_string())
    first = dataset.load_episode(dataset.runs.iloc[0]["RunId"])
    print(f"First episode LiDAR shape: {first['lidar_ranges_m'].shape}")
    print("Ready: import EpisodeDataset, make_run_splits, and lidar_grid in your notebook.")


if __name__ == "__main__":
    main()
