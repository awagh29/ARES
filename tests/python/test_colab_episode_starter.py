"""Run with: python -m unittest discover -s tests/python -v"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from colab_episode_starter import EpisodeDataset, SCHEMA, lidar_grid, make_run_splits


class TestColabEpisodeStarter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = self.root / "data/batches/fixture"
        self.batch.mkdir(parents=True)
        self.rows, reports, scans = [], [], []
        for number, axis in enumerate(["North", "North", "East", "East"], 1):
            run_id = f"run-{number:03d}"
            path = self.root / "data/processed/lidar_episode_v1" / run_id / "episode.npz"
            path.parent.mkdir(parents=True)
            ranges = np.full((3, 10740), 7.0)
            ranges[:, 0] = 2.0
            valid = np.zeros_like(ranges, dtype=bool)
            valid[:, 0] = True
            current = np.arange(12.0).reshape(3, 4)
            arrays = {
                "run_id": np.asarray(run_id), "schema_version": np.asarray(SCHEMA),
                "metadata_json": np.asarray("{}"), "time_s": np.array([0, 0.1, 0.2]),
                "sample_index": np.array([1, 2, 3], dtype=np.uint32),
                "lidar_enabled": np.ones(3, dtype=bool),
                "previous_action_available": np.array([False, True, True]),
                "lidar_ranges_m": ranges, "lidar_valid_return_mask": valid,
                "lidar_availability_mask": np.ones_like(valid),
                "lidar_missing_mask": np.zeros_like(valid),
                "action_current": current,
                "action_previous": np.vstack([np.full((1, 4), np.nan), current[:-1]]),
            }
            for name in ("position_ned_m", "velocity_ned_mps", "attitude_ned_rpy_rad",
                         "angular_velocity_body_radps", "goal_position_ned_m",
                         "reference_desired_position_ned_m"):
                arrays[name] = np.zeros((3, 3))
            np.savez_compressed(path, **arrays)
            reports.append({"runId": run_id, "scanCount": 3,
                            "npzSha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            self.rows.append({"RunIndex": number, "RunId": run_id, "Axis": axis,
                              "Status": "complete", "ScanCount": 3,
                              "MissionSuccess": 0, "SafetyChecksPassed": 0,
                              "CollisionOccurred": number == 1,
                              "NpzPath": rf"Z:\old-computer\{run_id}\episode.npz"})
            for sample in range(3):
                scans.append({"RunId": run_id, "Axis": axis, "SampleIndex": sample + 1,
                              "TimeS": sample * 0.1, "NearestValidReturnM": np.nan})
        pd.DataFrame(self.rows).to_csv(self.batch / "python_index.csv", index=False)
        pd.DataFrame(scans).to_csv(self.batch / "all_scans.csv", index=False)
        (self.batch / "dataset_qa.json").write_text(json.dumps(
            {"allDataChecksPassed": True, "totalScans": 12}))
        (self.batch / "python_export_qa.json").write_text(json.dumps(
            {"allExportChecksPassed": True, "partialTestExport": False,
             "totalScans": 12, "files": reports}))

    def dataset(self):
        return EpisodeDataset(self.root, "fixture", expected_runs=4)

    def test_rebases_windows_paths_and_keeps_unsafe_runs(self):
        source = (self.batch / "python_index.csv").read_bytes()
        data = self.dataset()
        episode = data.load_episode("run-001")
        self.assertEqual(len(data.runs), 4)
        self.assertTrue(data.runs.iloc[0].CollisionOccurred)
        self.assertEqual(episode["lidar_ranges_m"].shape, (3, 10740))
        self.assertEqual(episode["lidar_ranges_m"].dtype, np.float64)
        self.assertTrue(np.isnan(episode["action_previous"][0]).all())
        self.assertEqual((self.batch / "python_index.csv").read_bytes(), source)

    def test_selected_fields_and_streaming(self):
        data = self.dataset()
        items = data.iter_episodes(["run-001", "run-003"], fields=["position_ned_m"])
        self.assertIs(iter(items), items)
        run_id, episode = next(items)
        self.assertEqual(run_id, "run-001")
        self.assertNotIn("lidar_ranges_m", episode)
        self.assertIn("time_s", episode)
        self.assertEqual(next(items)[0], "run-003")
        with self.assertRaises(ValueError):
            data.load_episode("run-001", fields="position_ned_m")

    def test_detects_corruption_and_lfs_pointer(self):
        data = self.dataset()
        path = Path(data.runs.iloc[0].NpzPath)
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            data.load_episode("run-001")
        path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            data.load_episode("run-001")

    def test_missing_data_and_failed_audit(self):
        data = self.dataset()
        Path(data.runs.iloc[0].NpzPath).unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset()
        (self.batch / "dataset_qa.json").write_text(json.dumps({"allDataChecksPassed": False}))
        with self.assertRaisesRegex(ValueError, "audit"):
            self.dataset()

    def test_scan_table_preserves_nan_and_rejects_duplicate_samples(self):
        data = self.dataset()
        scans = data.load_scan_table()
        self.assertEqual(len(scans), 12)
        self.assertTrue(scans.NearestValidReturnM.isna().all())
        scans.loc[1, "SampleIndex"] = 1
        scans.to_csv(self.batch / "all_scans.csv", index=False)
        with self.assertRaisesRegex(ValueError, "indices/timestamps"):
            data.load_scan_table()

    def test_matlab_grid_order_and_mask_do_not_change_source(self):
        ranges = np.arange(10740.0).reshape(1, -1)
        valid = np.ones_like(ranges, dtype=bool)
        valid[0, 16] = False
        episode = {"lidar_ranges_m": ranges, "lidar_valid_return_mask": valid}
        grid = lidar_grid(episode)
        self.assertEqual(grid.shape, (15, 716))
        self.assertEqual(grid[14, 0], 14)
        self.assertEqual(grid[0, 1], 15)
        self.assertTrue(np.isnan(grid[1, 1]))
        self.assertEqual(ranges[0, 16], 16)

    def test_splits_are_disjoint_balanced_and_reproducible(self):
        runs = pd.DataFrame({"RunId": [f"run-{i:03d}" for i in range(80)],
                             "Axis": ["North"] * 40 + ["East"] * 40})
        split = make_run_splits(runs, seed=42)
        expected = {"train": 56, "validation": 8, "calibration": 8, "test": 8}
        self.assertEqual(split.Split.value_counts().to_dict(), expected)
        self.assertEqual(len(set(split.RunId)), 80)
        for _, group in split.groupby("Split"):
            self.assertEqual(group.Axis.value_counts()["North"], group.Axis.value_counts()["East"])
        pd.testing.assert_frame_equal(split, make_run_splits(runs, seed=42))
        shuffled = make_run_splits(runs.sample(frac=1, random_state=9), seed=42)
        self.assertEqual(split.set_index("RunId").Split.to_dict(),
                         shuffled.set_index("RunId").Split.to_dict())
        self.assertFalse(split.Split.equals(make_run_splits(runs, seed=43).Split))


if __name__ == "__main__":
    unittest.main()
