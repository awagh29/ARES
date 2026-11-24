"""Known-map geometry, causal isolation, fault generalisation and runnable training.

These checks execute definitions embedded in the delivered notebook. They do not
fit the production corpus or modify the single-episode baseline.
"""
from contextlib import redirect_stdout, contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/train_sparse_reconstruction_map_temporal.ipynb"
BASELINE = ROOT / "notebooks/train_sparse_reconstruction_single_episode_baseline.ipynb"


def namespace():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {}
    for index in (1, 8, 9, 10, 11, 13, 17, 20, 22):
        source = "".join(nb["cells"][index]["source"])
        with redirect_stdout(io.StringIO()):
            exec(compile(source, f"<map-temporal:{index}>", "exec"), ns)
    return ns


def tiny_config(ns, root):
    config = copy.deepcopy(ns["CONFIG"])
    config["paths"].update(repo_root=str(root), output_root=str(root/"runs"), cache_root=str(root/"cache"))
    config["model"].update(widths=[2, 4], kernel=3, history_hidden=8, state_hidden=8,
                           history_channels=2, obstacle_hidden=4)
    config["data"].update(history_steps=4, cache_trajectories=2)
    config["synthetic"].update(trajectory_count=3, speed_mps=[50., 50.])
    config["training"].update(epochs=1, batch_size=4, save_every_n_epochs=1)
    config["scheduler"]["t_max_epochs"] = 1
    config["plots"]["dpi"] = 40
    config["runtime"].update(device="cpu", pin_memory=False, fused_adamw=False, cudnn_benchmark=False)
    config["evaluation"]["test_profiles"] = ["assigned", "cold_blackout", "long_blackout", "blackout_burst"]
    return config


def make_fixture(ns, root):
    """Small healthy exports, using the ACTUAL ARES map/mount and independent rays."""
    config = tiny_config(ns, root)
    real_manifest = json.loads((ROOT/"data/raw/f9ab72ef-e1d7-4479-8faa-03bb5b36776c/manifest.json").read_text(encoding="utf-8-sig"))
    scene = ns["scene_from_manifest"](real_manifest, config["model"])
    boxes = np.asarray(scene["obstacles_ned_center_dimensions_m"], np.float32)
    files = []
    for ordinal in range(3):
        run_id, count = f"fixture-{ordinal}", 7
        raw = root/"data/raw"/run_id
        raw.mkdir(parents=True)
        manifest = copy.deepcopy(real_manifest)
        manifest["runId"], manifest["experiment"]["axis"] = run_id, "fixture"
        (raw/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        states = np.zeros((count, 9), np.float32)
        states[:, 0], states[:, 1], states[:, 2] = np.arange(count)*.1+ordinal, 6., -3.-ordinal
        states[:, 3] = 1.
        with torch.inference_mode():
            ranges, hits = ns["raycast_known_map"](torch.from_numpy(states), torch.tensor(boxes)[None].expand(count, -1, -1),
                ns["body_ray_directions"](config["model"]), torch.tensor([0., 0., -.4]))
        # Save using the real MATLAB flattened ray ordering (elevation-fastest).
        ranges = ranges.reshape(count, 15, 716).transpose(1, 2).reshape(count, -1).numpy()
        hits = hits.reshape(count, 15, 716).transpose(1, 2).reshape(count, -1).numpy()
        path = root/"data/processed/lidar_episode_v1"/run_id/"episode.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, run_id=np.asarray(run_id), schema_version=np.asarray("ares-python-lidar-episode-v1"),
            metadata_json=np.asarray(json.dumps(dict(lidar_grid_size=[15, 716], lidar_channel_order="elevation-index-fastest-then-azimuth-index"))),
            sample_index=np.arange(1, count+1), time_s=np.arange(count)*.1,
            position_ned_m=states[:, :3], velocity_ned_mps=states[:, 3:6], attitude_ned_rpy_rad=states[:, 6:9],
            goal_position_ned_m=np.tile([30., 35., -5.], (count, 1)), lidar_ranges_m=ranges,
            lidar_valid_return_mask=hits, lidar_availability_mask=np.ones_like(hits), lidar_missing_mask=np.zeros_like(hits),
            lidar_enabled=np.ones(count, bool))
        files.append(dict(runId=run_id, scanCount=count, npzSha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    audit = root/"data/batches"/config["data"]["batch_id"]/"python_export_qa.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps(dict(allExportChecksPassed=True, partialTestExport=False, files=files)))
    with redirect_stdout(io.StringIO()):
        corpus = ns["build_dataset"](root, config)
    return config, corpus


@contextmanager
def fixture_workspace(ns):
    """Always release memory maps BEFORE Windows removes the temporary cache."""
    with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as directory:
        original, created = ns["CausalTrajectoryDataset"], []
        def tracked(*args, **kwargs):
            dataset = original(*args, **kwargs)
            created.append(dataset)
            return dataset
        ns["CausalTrajectoryDataset"] = tracked
        try:
            config, corpus = make_fixture(ns, Path(directory))
            yield Path(directory), config, corpus
        finally:
            for dataset in created:
                dataset.close()
            ns["CausalTrajectoryDataset"] = original


class MapTemporalNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = namespace()

    def setUp(self):
        torch.manual_seed(42)

    def test_flow_and_separate_copy(self):
        original, delivered = [json.loads(p.read_text(encoding="utf-8")) for p in (BASELINE, NOTEBOOK)]
        self.assertEqual([(c["id"], c["cell_type"]) for c in original["cells"]],
                         [(c["id"], c["cell_type"]) for c in delivered["cells"]])
        for cell in delivered["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<delivered>", "exec")
                # Notebook execution state belongs to the user; rerunning cells
                # must not invalidate the source/flow checks or erase outputs.
        self.assertEqual(original["cells"][9]["source"], delivered["cells"][9]["source"])
        self.ns["validate_config"](self.ns["CONFIG"])
        self.assertEqual(self.ns["CONFIG"]["map"]["altitude_bounds_m"], [1., 10.])
        self.assertEqual(self.ns["CONFIG"]["map"]["north_bounds_m"], [-5., 40.])

    def test_analytic_rays_parallel_inside_mount_and_translation(self):
        ns = self.ns
        states = torch.zeros(2, 9)
        boxes = torch.tensor([[[4., 0., 0., 2., 2., 2.]], [[4., 0., 0., 2., 2., 2.]]])
        rays = torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
        ranges, valid = ns["raycast_known_map"](states, boxes, rays, torch.zeros(3))
        torch.testing.assert_close(ranges, torch.tensor([[3., 7., 7.], [3., 7., 7.]]))
        self.assertEqual(valid[0].tolist(), [True, False, False])
        shifted = states.clone(); shifted[:, :3] += torch.tensor([12., -20., -5.])
        shifted_boxes = boxes.clone(); shifted_boxes[..., :3] += shifted[:, None, :3]
        actual, _ = ns["raycast_known_map"](shifted, shifted_boxes, rays, torch.zeros(3))
        torch.testing.assert_close(actual, ranges)
        states[:, 0] = 4.
        inside, _ = ns["raycast_known_map"](states, boxes, rays, torch.zeros(3))
        torch.testing.assert_close(inside, torch.ones(2, 3))
        mounted, _ = ns["raycast_known_map"](torch.zeros(2, 9), boxes, rays, torch.tensor([1., 0., 0.]))
        self.assertEqual(mounted[0, 0].item(), 2.)

    def test_actual_recorded_scan_geometry_and_order(self):
        ns = self.ns
        scene, records, _ = ns["discover_sources"](ROOT, ns["CONFIG"])
        record = next(r for r in records if r["trajectory_id"] == "f9ab72ef-e1d7-4479-8faa-03bb5b36776c")
        states, _, _, target, valid = ns["load_recorded_trajectory"](record, ns["CONFIG"])
        select = np.linspace(0, len(states)-1, 12, dtype=int)
        with torch.inference_mode():
            predicted, hits = ns["raycast_known_map"](torch.from_numpy(states[select]),
                torch.tensor(scene["obstacles_ned_center_dimensions_m"])[None].expand(len(select), -1, -1),
                ns["body_ray_directions"](ns["CONFIG"]["model"]), torch.tensor([0., 0., -.4]))
        error = np.abs(predicted.numpy().reshape(-1, 15, 716)-target[select])
        # Allows boundary/mesh numerical tolerance; catches swapped axes, rays and mount.
        self.assertLess(float(np.mean(error)), .005)
        self.assertGreater(float(valid[select].mean()), 0)
        self.assertLess(float(np.mean(hits.numpy().reshape(-1, 15, 716) != valid[select])), .005)

    def test_paths_cover_free_volume_and_connect_random_endpoints(self):
        config = copy.deepcopy(self.ns["CONFIG"])
        manifest = json.loads((ROOT/"data/raw/f9ab72ef-e1d7-4479-8faa-03bb5b36776c/manifest.json").read_text(encoding="utf-8-sig"))
        boxes = np.array(self.ns["scene_from_manifest"](manifest, config["model"])["obstacles_ned_center_dimensions_m"])
        starts, goals = [], []
        for index in range(20):
            states, goal, times = self.ns["synthetic_trajectory"](index, boxes, config)
            self.assertTrue(self.ns["free_positions"](states[:, :3], boxes, config["map"]).all())
            np.testing.assert_allclose(states[-1, :3], goal[-1], atol=1e-5)
            np.testing.assert_allclose(np.diff(times), .1, atol=1e-10)
            self.assertGreaterEqual(np.linalg.norm(states[0, :3]-goal[-1]), 5.-1e-5)
            starts.append(states[0, :3]); goals.append(goal[-1])
        self.assertGreater(np.ptp(np.array(goals)[:, 2]), 7.)
        self.assertGreater(np.ptp(np.array(starts)[:, 0]), 35.)

    def test_faults_include_long_cold_and_repeated_outages(self):
        config = copy.deepcopy(self.ns["CONFIG"])
        cold, age = self.ns["fault_schedule"](140, config, np.random.default_rng(4), "cold_blackout")
        self.assertFalse(cold.any())
        self.assertGreater(age[-1], config["data"]["history_steps"]*.1)
        self.assertAlmostEqual(float(age[-1]), 14., places=4)
        config["masking"].update(partial_probability=0, beam_dropout_fraction=[0., 0.],
            blackout_start_probability=.5, blackout_duration_frames=[1, 3])
        repeated, _ = self.ns["fault_schedule"](100, config, np.random.default_rng(11), "mixed")
        fully_lost = ~repeated.any((1, 2))
        self.assertGreater(np.sum(fully_lost & ~np.r_[False, fully_lost[:-1]]), 3)
        healthy, healthy_age = self.ns["fault_schedule"](10, config, np.random.default_rng(9), "healthy")
        self.assertTrue(healthy.all()); self.assertFalse(healthy_age.any())

    def test_cache_whole_flight_splits_causal_history_and_fixed_validation(self):
        with fixture_workspace(self.ns) as (directory, config, corpus):
            folds = [set(corpus.splits[name]) for name in ("train", "validation", "test")]
            self.assertFalse(folds[0] & folds[1] or folds[1] & folds[2] or folds[0] & folds[2])
            train, val, _ = self.ns["build_train_val_tensors"](corpus, config)
            train.force_profile = "healthy"
            config_no_noise = train.config
            config_no_noise["masking"]["history_reset_probability"] = 0.
            for key in config_no_noise["augmentation"]:
                config_no_noise["augmentation"][key] = 0.
            first, third, fourth = train[0], train[2], train[3]
            self.assertEqual(int(first["present"].sum()), 1)
            torch.testing.assert_close(third["history"][1:], fourth["history"][:-1])
            next_start = train[int(train.offsets[1])]
            self.assertEqual(int(next_start["present"].sum()), 1)
            # Modify FUTURE cached observations/states and ensure an earlier window is identical.
            timeline = train._load(0)
            before = train[2]
            timeline["history"][3:] = 123.
            timeline["states"][3:] = -345.
            after = train[2]
            for key in ("ranges", "states", "history", "goals", "present"):
                torch.testing.assert_close(before[key], after[key])
            val_first = val[0]
            val.close()
            for key, expected in val_first.items():
                torch.testing.assert_close(expected, val[0][key])
            with self.assertRaisesRegex(ValueError, "frozen"):
                val.set_epoch(1)
            with redirect_stdout(io.StringIO()):
                cached = self.ns["build_dataset"](Path(directory), config)
            self.assertEqual(cached.source_metadata["fingerprint"], corpus.source_metadata["fingerprint"])
            path = Path(corpus.records[0]["folder"])/"ranges.npy"
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.ns["build_dataset"](Path(directory), config)

    def test_hidden_input_isolation_measurement_preservation_history_gradients(self):
        with fixture_workspace(self.ns) as (directory, config, corpus):
            data = self.ns["CausalTrajectoryDataset"](corpus, "validation", config, force_profile="cold_blackout")
            item = data[3]
            batch = {k: v.unsqueeze(0) for k, v in item.items()}
            model = self.ns["SparseRangeReconstructor"](config["model"]).eval()
            # Exercise learned branches rather than only the zero-initialized prior.
            torch.nn.init.normal_(model.head.weight, std=.01)
            torch.nn.init.normal_(model.film[-1].weight, std=.05)
            torch.nn.init.constant_(model.head.bias, -.1)
            torch.nn.init.constant_(model.film[-1].bias, .2)
            with torch.no_grad():
                expected = model(**self.ns["model_inputs"](batch))
                batch["ranges"].fill_(float("nan")); batch["valid"].fill_(True)
                batch["target"].fill_(-999.)
                actual = model(**self.ns["model_inputs"](batch))
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual).all())
            shifted = {k: v.clone() for k, v in batch.items()}
            shift = torch.tensor([50., -30., 0.])
            shifted["states"][..., :3] += shift
            shifted["goals"] += shift
            shifted["obstacles"][..., :3] += shift
            with torch.no_grad():
                translated = model(**self.ns["model_inputs"](shifted))
            torch.testing.assert_close(translated, actual, rtol=1e-3, atol=2e-4)
            model.train()
            prediction = model(**self.ns["model_inputs"](batch))
            prediction.sum().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.history_gru.parameters()), 0)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.obstacle_encoder.parameters()), 0)
            batch["available"].fill_(True); batch["ranges"].fill_(2.25)
            with torch.no_grad():
                measured = model(**self.ns["model_inputs"](batch))
            torch.testing.assert_close(measured, torch.full_like(measured, 2.25))

    def test_training_randomisation_and_independent_uav_sessions_match_dataset(self):
        with fixture_workspace(self.ns) as (directory, config, corpus):
            for key in config["augmentation"]:
                config["augmentation"][key] = 0.
            data = self.ns["CausalTrajectoryDataset"](corpus, "validation", config, force_profile="cold_blackout")
            model = self.ns["SparseRangeReconstructor"](config["model"]).eval()
            torch.nn.init.normal_(model.head.weight, std=.01)
            torch.nn.init.normal_(model.film[-1].weight, std=.03)
            runtime = self.ns["MultiUavReconstruction"](model, config)
            untouched = self.ns["MultiUavReconstruction"](model, config)
            for t in range(4):
                item = data[t]
                args = (t*.1, item["ranges"].numpy(), item["valid"].numpy(), item["available"].numpy(),
                        item["states"][-1].numpy(), item["goals"][-1].numpy(), item["obstacles"].numpy())
                output = runtime.step("uav_1", *args)
                independent = untouched.step("uav_1", *args)
                runtime.step("uav_2", *args)
                with torch.inference_mode():
                    expected = model(**self.ns["model_inputs"]({k: v.unsqueeze(0) for k, v in item.items()}))[0].numpy()
                np.testing.assert_allclose(output, expected, atol=1e-5)
                np.testing.assert_array_equal(output, independent)
            runtime.reset("uav_1")
            self.assertNotIn("uav_1", runtime.buffers)
            self.assertIn("uav_2", runtime.buffers)
            train = self.ns["CausalTrajectoryDataset"](corpus, "train", config)
            before = train.trajectory_seed(0)
            train.set_epoch(1)
            self.assertNotEqual(before, train.trajectory_seed(0))

    def test_one_epoch_smoke_artifacts_and_holdout(self):
        with fixture_workspace(self.ns) as (directory, config, corpus):
            train, val, test = self.ns["build_train_val_tensors"](corpus, config)
            val.force_profile = "cold_blackout"
            with redirect_stdout(io.StringIO()):
                model, history, run_dir = self.ns["train_sparse_reconstruction"](
                    self.ns["SparseRangeReconstructor"](config["model"]), train, val, test_data=test,
                    config=config, split_ids=corpus.splits, source_metadata=corpus.source_metadata)
            self.assertEqual(len(history["epoch"]), 1)
            summary = json.loads((run_dir/"summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(np.isfinite(history["val_blackout_rmse"]).all())
            for filename in ("source.json", "splits.json", "metrics.csv", "validation_outages.npz", "best.pt", "last.pt",
                             "epoch_0001.pt", "plots/loss_curves.png", "plots/validation_scan.png", "holdout_report.json"):
                self.assertTrue((run_dir/filename).is_file(), filename)
            holdout = json.loads((run_dir/"holdout_report.json").read_text())
            self.assertGreater(holdout["profiles"]["cold_blackout"]["groups"]["cold_start_blackout"]["samples"], 0)
            checkpoint = torch.load(run_dir/"best.pt", weights_only=True, map_location="cpu")
            restored = self.ns["SparseRangeReconstructor"](checkpoint["config"]["model"])
            restored.load_state_dict(checkpoint["model_state_dict"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for mixed-precision check")
    def test_default_architecture_cuda_forward_backward(self):
        config = copy.deepcopy(self.ns["CONFIG"])
        config["runtime"]["amp_dtype"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        device = self.ns["runtime_device"](config)
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        ns = dict(self.ns); ns["CONFIG"] = config
        with redirect_stdout(io.StringIO()):
            exec("".join(nb["cells"][12]["source"]), ns)
        model = self.ns["SparseRangeReconstructor"](config["model"]).to(device)
        horizon = config["data"]["history_steps"]
        sample = dict(ranges=torch.zeros(2, 15, 716, device=device),
            valid=torch.zeros(2, 15, 716, dtype=torch.bool, device=device),
            available=torch.zeros(2, 15, 716, dtype=torch.bool, device=device),
            states=torch.zeros(2, horizon, 9, device=device), goals=torch.zeros(2, horizon, 3, device=device),
            history=torch.zeros(2, horizon, 3, 3, 90, device=device),
            present=torch.ones(2, horizon, dtype=torch.bool, device=device), ages=torch.ones(2, horizon, device=device),
            obstacles=torch.tensor([[[0., 4., -7.5, 3., 3., 15.]]], device=device).expand(2, -1, -1))
        sample["states"][:, :, 2] = -5.
        with self.ns["autocast_context"](config, device):
            output = model(**sample)
        loss = (output-3.).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        del model, sample
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
