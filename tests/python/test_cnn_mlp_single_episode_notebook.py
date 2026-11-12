"""Single-path loading, chronological isolation and persisted GPU training."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/train_cnn_mlp_baseline_single_episode.ipynb"


def notebook_namespace():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = dict(np=np, torch=torch, nn=nn, F=F, plt=plt, DataLoader=DataLoader)
    wanted = {"cnn-required-config", "454965e7", "6bda8cc7", "be00db75",
              "13cdd8cb", "90e51251", "6b13d3b0"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"<single-episode:{cell['id']}>", "exec")
        if cell["id"] == "cnn-required-config":
            # The delivered notebook intentionally requires the user to supply a path.
            source = source.split("\nvalidate_config(CONFIG)\n")[0]
        if cell["id"] in wanted:
            exec(source, ns)
    return ns


def fixture(root, count=10):
    episode_id = "fixture-episode"
    raw = root / "data/raw" / episode_id
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text(json.dumps({
        "runId": episode_id, "static": {"maximum_sensor_range_m": 7.0},
    }), encoding="utf-8")
    path = root / "data/processed/lidar_episode_v1" / episode_id / "episode.npz"
    path.parent.mkdir(parents=True)
    ranges = np.full((count, 10740), 7, np.float32)
    ranges[:, :30] = np.arange(count)[:, None] * .01 + np.arange(30)[None, :] * .001 + .1
    arrays = {
        "run_id": np.asarray(episode_id),
        "schema_version": np.asarray("ares-python-lidar-episode-v1"),
        "metadata_json": np.asarray(json.dumps({
            "lidar_grid_size": [15, 716],
            "lidar_channel_order": "elevation-index-fastest-then-azimuth-index",
        })),
        "time_s": np.arange(count) * .1,
        "sample_index": np.arange(1, count + 1),
        "position_ned_m": np.repeat(np.arange(count)[:, None], 3, axis=1).astype(np.float32),
        "velocity_ned_mps": np.ones((count, 3), np.float32),
        "attitude_ned_rpy_rad": np.zeros((count, 3), np.float32),
        "lidar_ranges_m": ranges,
        "lidar_valid_return_mask": ranges < 7,
        "lidar_availability_mask": np.ones_like(ranges, dtype=bool),
        "lidar_missing_mask": np.zeros_like(ranges, dtype=bool),
        "lidar_enabled": np.ones(count, dtype=bool),
    }
    np.savez_compressed(path, **arrays)
    report = root / "data/batches/fixture/python_export_qa.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"allExportChecksPassed": True, "files": [{
        "runId": episode_id, "scanCount": count,
        "npzSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }]}), encoding="utf-8")
    return raw, path, arrays


class TestSingleEpisodeNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = notebook_namespace()

    def load(self, root, path, **kwargs):
        return self.ns["build_dataset"](
            root, path, .8, .2, split_policy="chronological", **kwargs,
        )

    def test_same_flow_and_required_episode_path(self):
        original = json.loads((ROOT / "notebooks/train_cnn_mlp_baseline.ipynb").read_text(encoding="utf-8"))
        single = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual([c["id"] for c in original["cells"]], [c["id"] for c in single["cells"]])
        for cell_id in ("454965e7", "6bda8cc7", "be00db75", "cnn-results-history", "cnn-results-scan"):
            before = next(c for c in original["cells"] if c["id"] == cell_id)
            after = next(c for c in single["cells"] if c["id"] == cell_id)
            self.assertEqual(before["source"], after["source"])
        config = copy.deepcopy(self.ns["CONFIG"])
        with self.assertRaisesRegex(ValueError, "episode_path"):
            self.ns["validate_config"](config)
        config["data"]["episode_path"] = "fixture-episode"
        self.ns["validate_config"](config)
        self.assertNotIn("batch_id", config["data"])
        self.assertNotIn("expected_episodes", config["data"])

    def test_one_episode_chronology_alignment_and_path_forms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, _, arrays = fixture(root)
            for path in (str(raw), "data/raw/fixture-episode", "fixture-episode"):
                corpus = self.load(root, path, verify_hashes=True)
                self.assertEqual(corpus.split_ranges, {"train": [0, 8], "validation": [8, 10]})
                self.assertTrue(set(corpus.splits["train"]).isdisjoint(corpus.splits["validation"]))
                self.assertEqual(corpus.diagnostics["train"]["scan_count"], 8)
                self.assertLess(corpus.diagnostics["train"]["last_time_s"],
                                corpus.diagnostics["validation"]["first_time_s"])
                train_x, train_y, val_x, val_y = self.ns["build_train_val_tensors"](corpus)
                self.assertEqual(train_x.shape, (8, 9))
                self.assertEqual(val_y.shape, (2, 15, 716))
                self.assertEqual(train_y.dtype, torch.float32)
                self.assertEqual(val_x.device.type, "cpu")
                np.testing.assert_array_equal(val_x[0, :3], arrays["position_ned_m"][8])
                np.testing.assert_array_equal(val_y[0], arrays["lidar_ranges_m"][8].reshape(15, 716, order="F"))
                self.assertTrue(corpus.source_metadata["checksum_verified"])

    def test_checksum_missing_export_and_wrong_path_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, path, _ = fixture(root)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.load(root, str(raw), verify_hashes=True)
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Python export"):
                self.load(root, str(raw), verify_hashes=True)
            with self.assertRaisesRegex(ValueError, "data/raw"):
                self.load(root, "../outside", verify_hashes=False)

    def test_rejects_unhealthy_lidar(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, path, arrays = fixture(root)
            arrays["lidar_missing_mask"][1, 1] = True
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "healthy LiDAR"):
                self.load(root, str(raw), verify_hashes=False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for the training check.")
    def test_training_artifacts_and_scan_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, _, _ = fixture(root)
            corpus = self.load(root, str(raw), verify_hashes=True)
            train_x, train_y, val_x, val_y = self.ns["build_train_val_tensors"](corpus)
            config = copy.deepcopy(self.ns["CONFIG"])
            config["paths"].update(repo_root=str(root), output_root=str(root / "runs"))
            config["data"]["episode_path"] = str(raw)
            config["training"].update(epochs=2, batch_size=3, save_every_n_epochs=2)
            config["scheduler"]["t_max_epochs"] = 2
            config["model"].update(encoder_hidden=[8, 16], latent_channels=4,
                                   decoder_channels=[8, 8, 4, 4, 15])
            config["plots"]["dpi"] = 60
            config["runtime"]["amp_dtype"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            train = TensorDataset(train_x, train_y)
            val = TensorDataset(val_x, val_y)
            with redirect_stdout(io.StringIO()):
                model, history, run_dir = self.ns["train_cnn_mlp_baseline"](
                    self.ns["PoseToRangeCNN"](config["model"]), train, val,
                    config=config, split_ids=corpus.splits,
                    source_metadata=corpus.source_metadata,
                )
            self.assertEqual(len(history["epoch"]), 2)
            self.assertAlmostEqual(history["next_learning_rate"][-1], config["scheduler"]["min_lr"])
            for name in ("source.json", "best.pt", "last.pt", "epoch_0002.pt", "config.json",
                         "metrics.csv", "metrics.json", "summary.json", "training.log",
                         "plots/loss_curves.png", "plots/learning_rate.png", "plots/validation_scan.png"):
                self.assertGreater((run_dir / name).stat().st_size, 0, name)
            self.assertEqual(json.loads((run_dir / "splits.json").read_text()), corpus.splits)
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["episode_id"], "fixture-episode")
            self.assertEqual(summary["status"], "completed")
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["source_metadata"]["npz_sha256"], corpus.source_metadata["npz_sha256"])
            self.assertEqual(checkpoint["split_ids"], corpus.splits)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), checkpoint["model_state_dict"][key])
            swapped = copy.deepcopy(corpus.splits)
            swapped["train"].reverse()
            with self.assertRaisesRegex(ValueError, "chronological"):
                self.ns["train_cnn_mlp_baseline"](
                    model, train, val, config=config, split_ids=swapped,
                    source_metadata=corpus.source_metadata,
                )


if __name__ == "__main__":
    unittest.main()
