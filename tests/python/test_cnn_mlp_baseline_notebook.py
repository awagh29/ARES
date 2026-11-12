"""Check configuration and persisted runs without training on recorded flights."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
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
sys.path.insert(0, str(ROOT / "scripts"))


def notebook_namespace():
    notebook = json.loads((ROOT / "notebooks/train_cnn_mlp_baseline.ipynb").read_text(encoding="utf-8"))
    namespace = dict(np=np, torch=torch, nn=nn, F=F, plt=plt, DataLoader=DataLoader)
    wanted = {"cnn-required-config", "454965e7", "6bda8cc7", "be00db75", "90e51251"}
    with redirect_stdout(io.StringIO()):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"<notebook:{cell['id']}>", "exec")
                if cell["id"] in wanted:
                    exec(source, namespace)
    return namespace


class TestCNNNotebookConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = notebook_namespace()

    def test_explicit_settings_and_missing_values(self):
        config = self.ns["CONFIG"]
        self.assertEqual(config["training"]["batch_size"], 16)
        self.assertEqual(config["scheduler"]["name"], "cosine")
        for section, key, value in (
            ("training", "learning_rate", None),
            ("paths", "output_root", " "),
            ("training", "epochs", 1.5),
            ("model", "kernel_sizes", [4, 5, 3, 3, 3]),
            ("scheduler", "t_max_epochs", 2),
        ):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                self.ns["validate_config"](changed)
        changed = copy.deepcopy(config)
        del changed["scheduler"]["min_lr"]
        with self.assertRaisesRegex(ValueError, "Missing configuration"):
            self.ns["validate_config"](changed)

    def test_configured_model_and_periodic_encoding(self):
        raw = torch.zeros(2, 9)
        shifted = raw.clone()
        shifted[:, 6:] += 2 * torch.pi
        encode = self.ns["encode_sensor_state"]
        torch.testing.assert_close(encode(raw), encode(shifted), rtol=0, atol=1e-6)
        with torch.inference_mode():
            result = self.ns["PoseToRangeCNN"](self.ns["CONFIG"]["model"])(encode(raw))
        self.assertEqual(result.shape, (2, 15, 716))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for training checks.")
class TestCNNNotebookRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = notebook_namespace()

    def config(self, output):
        config = copy.deepcopy(self.ns["CONFIG"])
        config["paths"]["output_root"] = str(output)
        config["data"]["expected_episodes"] = 2
        config["training"].update(epochs=2, batch_size=2, save_every_n_epochs=1)
        config["scheduler"]["t_max_epochs"] = 2
        config["model"].update(encoder_hidden=[8, 16], latent_channels=4,
                               decoder_channels=[8, 8, 4, 4, 15])
        config["plots"]["dpi"] = 60
        config["runtime"]["amp_dtype"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return config

    def run_training(self, config, train, val, model=None):
        if model is None:
            model = self.ns["PoseToRangeCNN"](config["model"])
        with redirect_stdout(io.StringIO()):
            return self.ns["train_cnn_mlp_baseline"](
                model, train, val, config=config,
                split_ids={"train": ["fixture-train"], "validation": ["fixture-val"]},
            )

    def test_saved_runs_schedule_best_weights_and_unique_directories(self):
        with tempfile.TemporaryDirectory() as output:
            config = self.config(output)
            states = torch.randn(8, 9)
            scans = torch.rand(8, 15, 716) * 7
            train = TensorDataset(states[:5], scans[:5])
            val = TensorDataset(states[5:], scans[5:])
            model, history, run_dir = self.run_training(config, train, val)
            for name in ("config.json", "splits.json", "summary.json", "training.log",
                         "metrics.csv", "metrics.json", "best.pt", "last.pt", "epoch_0002.pt",
                         "plots/loss_curves.png", "plots/learning_rate.png", "plots/validation_scan.png"):
                self.assertGreater((run_dir / name).stat().st_size, 0, name)
            self.assertEqual(len(history["epoch"]), 2)
            self.assertEqual(history["learning_rate"][0], config["training"]["learning_rate"])
            self.assertLess(history["learning_rate"][1], history["learning_rate"][0])
            self.assertAlmostEqual(history["next_learning_rate"][-1], config["scheduler"]["min_lr"])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertAlmostEqual(summary["best_val_mse"], min(history["val_mse"]))
            for name in ("best.pt", "last.pt"):
                checkpoint = torch.load(run_dir / name, map_location="cpu", weights_only=True)
                for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                            "scaler_state_dict", "config", "torch_rng_state", "loader_rng_state"):
                    self.assertIn(key, checkpoint)
            best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), best["model_state_dict"][key])
            self.assertFalse(model.training)
            self.assertEqual(json.loads((run_dir / "metrics.json").read_text()), history)
            self.assertEqual(len((run_dir / "metrics.csv").read_text().splitlines()), 3)
            # Re-render from persisted metrics; no training data needed.
            self.ns["plot_run_history"](run_dir, show=False)
            _, _, second_dir = self.run_training(config, train, val)
            self.assertNotEqual(run_dir, second_dir)
            self.assertTrue((run_dir / "best.pt").is_file())

    def test_early_stop_and_failed_run_metadata(self):
        with tempfile.TemporaryDirectory() as output:
            config = self.config(output)
            config["training"].update(epochs=3, patience=1)
            config["scheduler"]["t_max_epochs"] = 3

            class ConstantModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model_config = copy.deepcopy(config["model"])
                    self.anchor = nn.Parameter(torch.zeros(()))

                def forward(self, states):
                    return (self.anchor * 0).expand(len(states), 15, 716)

            dataset = TensorDataset(torch.zeros(3, 9), torch.ones(3, 15, 716))
            _, history, run_dir = self.run_training(config, dataset, dataset, ConstantModel())
            self.assertEqual(history["stale_epochs"], [0, 1])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "early_stopped")
            self.assertEqual(summary["best_epoch"], 1)
            before = set(Path(output).iterdir())
            invalid = TensorDataset(torch.zeros(3, 9), torch.full((3, 15, 716), float("nan")))
            with self.assertRaises(FloatingPointError):
                self.run_training(config, dataset, invalid)
            failed_dir, = set(Path(output).iterdir()) - before
            failed = json.loads((failed_dir / "summary.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error_type"], "FloatingPointError")
            self.assertIn("FloatingPointError", (failed_dir / "training.log").read_text())


if __name__ == "__main__":
    unittest.main()
