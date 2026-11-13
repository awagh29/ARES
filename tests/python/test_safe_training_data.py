"""Data causality, whole-flight isolation and physical rolled-pose sensing.

Synthetic fixtures below test mechanics only. The production builder and
integration checks use actual verified NPZ episodes, never invented flights.
"""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from ares_safe_training.data import (
    Corpus, Trajectory, FeatureNormalizer, build_dataset, make_pretraining_windows,
    split_run_records, FEATURE_NAMES, STATE_FEATURE_NAMES, DEFAULT_BATCH_ID,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from ares_safe_training.geometry import TorchLidarScene


def trajectory(index=1, split="train", count=180, no_returns=False):
    times = np.arange(count) * .1
    position = np.column_stack((times * .2 + index, np.full(count, index*.5), np.full(count, -7.))).astype(np.float32)
    velocity = np.column_stack((np.full(count, .2), np.zeros((count, 2)))).astype(np.float32)
    rpy, omega = np.zeros((count, 3), np.float32), np.zeros((count, 3), np.float32)
    clean = np.column_stack((np.tile((2 + times*.01)[:, None], (1, 7)), np.full((count, 7), .25))).astype(np.float32)
    if no_returns:
        clean[:, :7], clean[:, 7:] = 7, 0
    return Trajectory(f"fixture-{index}", index, "North" if index % 2 else "East", split,
        position, velocity, rpy, omega, clean, np.full((count, 4), 1e6, np.float32), times,
        np.concatenate((position[0], np.zeros(10))).astype(np.float32),
        np.array([30, 35, -7], np.float32), f"fixture-sha-{index}", True, True, False)


def corpus_fixture(no_returns=False):
    items = [trajectory(1, no_returns=no_returns), trajectory(2),
             trajectory(3, "validation"), trajectory(4, "test")]
    return Corpus(items, {}, {split: [item.run_id for item in items if item.split == split]
        for split in ("train", "validation", "test")}, [], {}, "fixture", 42)


def scene_fixture(minimum=None, maximum=None):
    return {"box_min_ned_m": [[4, -1, -15]] if minimum is None else minimum,
        "box_max_ned_m": [[6, 1, 0]] if maximum is None else maximum,
        "mounting_translation_body_m": [0, 0, -.4],
        "sensor_to_body_rotation": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "maximum_range_m": 7, "azimuth_limits_deg": [-179, 179], "azimuth_count": 716,
        "elevation_limits_deg": [-15, 15], "elevation_count": 15,
        "sector_edges_body_deg": np.linspace(-60, 60, 8).tolist()}


class TestWholeTrajectorySplits(unittest.TestCase):
    def setUp(self):
        self.records = [{"run_id": f"run-{axis}-{index}", "run_index": offset+index,
            "axis": axis, "axis_start_m": index*.7, "mission_success": index % 2 == 0,
            "contains_returns": index < 25} for axis, offset in (("North", 0), ("East", 40))
            for index in range(1, 41)]

    def test_exact_three_way_axis_quotas_and_start_ranges(self):
        assignment, strata = split_run_records(self.records)
        self.assertEqual(len(assignment), 80)
        for split, expected in (("train", 56), ("validation", 12), ("test", 12)):
            chosen = [item for item in self.records if assignment[item["run_id"]] == split]
            self.assertEqual(len(chosen), expected)
            self.assertEqual(sum(item["axis"] == "North" for item in chosen), expected//2)
            for axis in ("North", "East"):
                self.assertEqual(len({strata[item["run_id"]] for item in chosen if item["axis"] == axis}), 5)

    def test_deterministic_and_independent_of_outcomes_or_return_labels(self):
        first = split_run_records(self.records, seed=17)
        changed = copy.deepcopy(self.records)
        for item in changed:
            item["mission_success"] = not item["mission_success"]
            item["contains_returns"] = not item["contains_returns"]
        self.assertEqual(first, split_run_records(changed, seed=17))
        self.assertNotEqual(first, split_run_records(self.records, seed=18))
        self.assertEqual(first, split_run_records(list(reversed(self.records)), seed=17))


class TestCausalRealLabelWindows(unittest.TestCase):
    def test_cold_padding_and_encoder_allowlist(self):
        corpus = corpus_fixture()
        result = make_pretraining_windows(corpus, "train", profiles=("healthy",))
        self.assertEqual(result["raw_history"].shape, (360, 50, 31))
        self.assertEqual(result["current_state"].shape, (360, 34))
        self.assertTrue(np.isnan(result["raw_history"][0, :49]).all())
        self.assertFalse(result["observed"][0, :49].any())
        self.assertTrue(result["observed"][0, -1].all())
        self.assertFalse(result["history_ready"][0])
        self.assertTrue(result["history_ready"][49])
        np.testing.assert_array_equal(result["current_state"][0, :3], corpus.trajectories[0].position[0])
        self.assertEqual(tuple(result["metadata"]["feature_names"]), FEATURE_NAMES)
        self.assertEqual(tuple(result["metadata"]["state_feature_names"]), STATE_FEATURE_NAMES)
        self.assertFalse(any("goal" in name or "command" in name or "scene" in name for name in STATE_FEATURE_NAMES))
        self.assertLess(np.nanmax(np.abs(result["current_state"])), 1000)  # commands are a 1e6 sentinel

    def test_augmentation_preserves_labels_and_pairs_partial_sector_channels(self):
        corpus = corpus_fixture()
        original = corpus.trajectories[0].clean_lidar.copy()
        result = make_pretraining_windows(corpus, "train", profiles=("mixed", "healthy"))
        for run in (1, 2):
            selected = result["run_index"] == run
            np.testing.assert_array_equal(result["target"][selected],
                corpus.trajectories[run-1].clean_lidar[result["row_index"][selected]])
        np.testing.assert_array_equal(corpus.trajectories[0].clean_lidar, original)
        endpoint_mask = result["observed"][:, -1]
        self.assertTrue(endpoint_mask[:, 14:].all())
        np.testing.assert_array_equal(endpoint_mask[:, :7], endpoint_mask[:, 7:14])
        np.testing.assert_array_equal(result["target_mask"], ~endpoint_mask[:, :14])
        self.assertTrue(result["clean_label_mask"].all())
        self.assertTrue(np.any(result["target_mask"].all(1)))
        self.assertTrue(np.any(result["target_mask"].any(1) & ~result["target_mask"].all(1)))
        for schedule in result["metadata"]["schedules"]:
            for interval in schedule["intervals"]:
                low, high = (5., 7.) if interval["mode"] == "long" else (1.5, 3.)
                self.assertGreaterEqual(interval["duration_s"] + 1e-9, low)
                self.assertLessEqual(interval["duration_s"], high + 1e-9)

    def test_future_measurements_cannot_change_earlier_inputs(self):
        corpus = corpus_fixture()
        first = make_pretraining_windows(corpus, "train", profiles=("mixed",), run_indices=[1])
        changed = copy.deepcopy(corpus)
        changed.trajectories[0].clean_lidar[81:, :7] = 6
        changed.trajectories[0].position[81:] += 900
        second = make_pretraining_windows(changed, "train", profiles=("mixed",), run_indices=[1])
        past = first["row_index"] <= 80
        np.testing.assert_array_equal(first["raw_history"][past], second["raw_history"][past])
        np.testing.assert_array_equal(first["observed"][past], second["observed"][past])
        np.testing.assert_array_equal(first["current_state"][past], second["current_state"][past])

    def test_healthy_no_return_is_observed_not_a_sensor_outage(self):
        corpus = corpus_fixture(no_returns=True)
        result = make_pretraining_windows(corpus, "train", profiles=("healthy",), run_indices=[1])
        self.assertTrue(result["observed"][:, -1, :14].all())
        self.assertFalse(result["target_mask"].any())
        self.assertTrue((result["target"][:, :7] == 7).all())
        self.assertTrue((result["target"][:, 7:] == 0).all())
        self.assertFalse(result["contains_returns"].any())

    def test_nested_calibration_and_heldout_rows_do_not_fit_normalizer(self):
        corpus = corpus_fixture()
        first = FeatureNormalizer.fit(corpus, profiles=("healthy",), run_indices=[1])
        changed = copy.deepcopy(corpus)
        for item in changed.trajectories[1:]:
            item.position[:] = 1e8
            item.velocity[:] = -1e8
        second = FeatureNormalizer.fit(changed, profiles=("healthy",), run_indices=[1])
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.fit_row_count, 180)
        self.assertEqual(first.training_run_ids, ("fixture-1",))
        with self.assertRaises(ValueError):
            FeatureNormalizer.fit(corpus, run_indices=[3])
        with self.assertRaises(ValueError):
            make_pretraining_windows(corpus, "train", run_indices=[3])

    def test_normalizer_roundtrip_and_finite_masked_tensors(self):
        corpus = corpus_fixture()
        normalizer = FeatureNormalizer.fit(corpus)
        restored = FeatureNormalizer.from_dict(json.loads(json.dumps(normalizer.to_dict())))
        result = make_pretraining_windows(corpus, "train", run_indices=[1])
        history = restored.transform_history_numpy(result["raw_history"], result["observed"])
        current = restored.transform_state_numpy(result["current_state"], result["current_state_observed"])
        self.assertTrue(np.isfinite(history).all())
        self.assertTrue(np.isfinite(current).all())
        self.assertTrue((history[~result["observed"]] == 0).all())
        np.testing.assert_allclose(restored.history_std[:7], 7)
        np.testing.assert_allclose(restored.history_std[7:14], 1)
        np.testing.assert_allclose(restored.inverse_targets_numpy(
            restored.transform_targets_numpy(result["target"])), result["target"], atol=1e-6)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable; data-only tests remain usable.")
class TestPhysicalRolledLidar(unittest.TestCase):
    def state(self, batch=1):
        state = torch.zeros(batch, 13)
        state[:, 2] = -7
        return state

    def test_exact_forward_beam_counts_and_nearest_occlusion(self):
        scene = TorchLidarScene(scene_fixture([[2, -1, -15], [4, -1, -15]], [[3, 1, 0], [6, 1, 0]]))
        self.assertEqual(len(scene.rays_body), 3600)
        np.testing.assert_array_equal(scene.sector_beam_counts.numpy(), [510, 525, 510, 510, 510, 525, 510])
        value = scene.features(self.state())
        expected = 2/np.cos(np.deg2rad(179/715))
        self.assertAlmostEqual(float(value[0, 3]), expected, places=4)
        shifted = self.state(); shifted[:, 0] = .5
        self.assertLess(float(scene.features(shifted)[0, 3]), float(value[0, 3])-.49)

    def test_sensor_left_maps_to_negative_body_bearing(self):
        scene = TorchLidarScene(scene_fixture([[3.5, -5.5, -15]], [[4.5, -4.5, 0]]))
        value = scene.features(self.state())
        self.assertLess(float(value[0, 0]), 7)
        self.assertGreater(float(value[0, 7]), 0)
        self.assertEqual(float(value[0, 6]), 7)
        self.assertEqual(float(value[0, 13]), 0)

    def test_hit_distance_gradients_and_chunking_agree(self):
        first = TorchLidarScene(scene_fixture(), batch_chunk=1, ray_chunk=101)
        second = TorchLidarScene(scene_fixture(), batch_chunk=16, ray_chunk=512)
        state = self.state(2); state[1, 0] = .1
        state.requires_grad_()
        one, two = first.features(state), second.features(state)
        torch.testing.assert_close(one, two, atol=1e-6, rtol=1e-6)
        two[:, 3].sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue((state.grad[:, 0] < -.9).all())

    def test_no_returns_have_finite_zero_gradient_and_known_encoding(self):
        scene = TorchLidarScene(scene_fixture([[10, -1, -15]], [[12, 1, 0]]))
        state = self.state(); state.requires_grad_()
        value = scene.features(state)
        torch.testing.assert_close(value[:, :7], torch.full((1, 7), 7.))
        torch.testing.assert_close(value[:, 7:], torch.zeros(1, 7))
        value.sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_signed_three_dimensional_clearance(self):
        scene = TorchLidarScene(scene_fixture())
        positions = torch.tensor([[[0., 0., -7.], [5., 0., -7.], [5., 0., -17.]]], requires_grad=True)
        value = scene.clearance(positions, vehicle_radius=.1)
        torch.testing.assert_close(value, torch.tensor([[3.9, -1.1, 1.9]]), atol=1e-6, rtol=0)
        value.sum().backward()
        self.assertTrue(torch.isfinite(positions.grad).all())


@unittest.skipUnless((ROOT / "data" / "batches" / DEFAULT_BATCH_ID / "python_index.csv").is_file(),
                     "Actual 80-run export is not installed in this checkout.")
class TestActualCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_dataset(ROOT)

    def test_all_actual_trajectories_retained_and_heldout_folds_have_returns(self):
        corpus = self.corpus
        self.assertEqual(len(corpus.trajectories), 80)
        self.assertEqual(sum(len(item.times) for item in corpus.trajectories), 19185)
        self.assertEqual([len(corpus.splits[name]) for name in ("train", "validation", "test")], [56, 12, 12])
        self.assertEqual(sum(item.collision for item in corpus.trajectories), 2)
        self.assertEqual(sum(not item.safety_pass for item in corpus.trajectories), 33)
        self.assertEqual(sum(int(item.contains_returns.sum()) for item in corpus.trajectories), 6919)
        for name in ("train", "validation", "test"):
            self.assertGreater(corpus.diagnostics[name]["scans_with_returns"], 0)
        self.assertFalse(set(corpus.splits["train"]) & set(corpus.splits["test"]))
        self.assertEqual(np.asarray(corpus.scene["box_min_ned_m"]).shape, (6, 3))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable.")
    def test_raycaster_matches_recorded_targets_at_actual_poses(self):
        scene = TorchLidarScene(self.corpus.scene)
        for run_index, row in ((1, 50), (20, 75), (41, 25), (55, 25)):
            item = self.corpus.trajectories[run_index-1]
            # F is irrelevant to optics; no unknown interior thrust is inferred.
            state = torch.tensor(np.r_[item.state12[row], 0.], dtype=torch.float32)[None]
            actual = scene.features(state)[0].detach().numpy()
            np.testing.assert_allclose(actual[:7], item.clean_lidar[row, :7], atol=2e-4, rtol=0)
            np.testing.assert_allclose(actual[7:], item.clean_lidar[row, 7:], atol=.003, rtol=0)

    def test_compact_cache_is_new_and_retains_split_and_source_provenance(self):
        from ares_safe_training.data import _write_cache
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.npz"
            _write_cache(self.corpus, path)
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(archive["metadata_json"].item())
                self.assertEqual(metadata["splits"], self.corpus.splits)
                self.assertEqual(len(archive["trajectory_offsets"]), 81)
                self.assertEqual(archive["clean_lidar"].shape, (19185, 14))
                self.assertEqual(metadata["runs"][0]["source_npz_sha256"], self.corpus.records[0]["source_npz_sha256"])
            with self.assertRaises(ValueError):
                _write_cache(self.corpus, path)


if __name__ == "__main__":
    unittest.main()
