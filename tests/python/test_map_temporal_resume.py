"""Exact CPU continuation against uninterrupted training, plus legacy recovery."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts/carc"), str(ROOT/"tests/python")]
import checkpoint_resume as resume
import run_map_temporal as launcher
from test_map_temporal_sparse_notebook import namespace, fixture_workspace


class StopAt:
    checkpoint_seconds = float("inf")
    reason = "test interruption"
    def __init__(self, stage=None, epoch=1, batch=2):
        self.stage, self.epoch, self.batch = stage, epoch, batch
    def should_stop(self, **position):
        return position.get("stage") == self.stage and position.get("epoch") == self.epoch and position.get("next_batch") == self.batch


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = namespace()
        # Report rendering is covered by the notebook integration suite.
        cls.ns["plot_run_history"] = lambda *a, **k: None
        cls.ns["plot_validation_example"] = lambda *a, **k: None

    def fit(self, corpus, config, *, checkpoint=None, control=None):
        train, val, _ = self.ns["build_train_val_tensors"](corpus, config)
        val.force_profile = "cold_blackout"
        self.ns["seed_training"](config["run"]["seed"])
        model = self.ns["SparseRangeReconstructor"](config["model"])
        with redirect_stdout(io.StringIO()):
            return self.ns["train_sparse_reconstruction"](model, train, val, config=config,
                split_ids=corpus.splits, source_metadata=corpus.source_metadata,
                resume_checkpoint=checkpoint, control=control or StopAt())

    def test_mid_epoch_resume_matches_uninterrupted_weights_optimizer_and_lr(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            config["training"]["epochs"] = 2
            config["scheduler"]["t_max_epochs"] = 4
            _, full_history, full_dir = self.fit(corpus, config)
            _, paused_history, paused_dir = self.fit(corpus, config, control=StopAt("after_batch", 1, 2))
            self.assertEqual(paused_history["epoch"], [])
            initial = resume.load_checkpoint(paused_dir/"last.pt")
            self.assertEqual(initial["resume_state"]["next_batch"], 2)
            self.assertEqual(initial["epoch"], 0)
            # Simulate an extra metric row written before a killed checkpoint commit.
            (paused_dir/"metrics.csv").write_text("stale,row\n999,999\n")
            (paused_dir/"summary.json").write_text("{killed JSON write")
            _, history, same_dir = self.fit(corpus, config, checkpoint=paused_dir/"last.pt")
            self.assertEqual(same_dir, paused_dir)
            self.assertEqual(history["epoch"], [1, 2])
            self.assertEqual(history["learning_rate"], full_history["learning_rate"])
            self.assertEqual(history["val_loss"], full_history["val_loss"])
            full, continued = [resume.load_checkpoint(folder/"last.pt") for folder in (full_dir, paused_dir)]
            for key, value in full["model_state_dict"].items():
                torch.testing.assert_close(value, continued["model_state_dict"][key], rtol=0, atol=0)
            for key, state in full["optimizer_state_dict"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, continued["optimizer_state_dict"]["state"][key][name], rtol=0, atol=0)
            self.assertEqual(full["scheduler_state_dict"], continued["scheduler_state_dict"])
            self.assertEqual(len((paused_dir/"metrics.csv").read_text().splitlines()), 3)
            self.assertEqual(json.loads((paused_dir/"summary.json").read_text())["status"], "completed")

    def test_legacy_last_and_best_checkpoint_continuation(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            config["training"]["epochs"] = 2
            config["scheduler"]["t_max_epochs"] = 4
            _, expected, full_dir = self.fit(corpus, config)
            _, _, paused_dir = self.fit(corpus, config, control=StopAt("epoch_end", 1, 0))
            saved = resume.load_checkpoint(paused_dir/"last.pt")
            saved["torch_rng_state"] = saved["rng_state"]["torch"]
            saved["cuda_rng_states"] = saved["rng_state"]["cuda"]
            for key in ("resume_state", "checkpoint_version", "rng_state", "best_model_state_dict", "validation_fault_manifest"):
                saved.pop(key, None)
            legacy = paused_dir/"legacy.pt"
            torch.save(saved, legacy)
            _, history, _ = self.fit(corpus, config, checkpoint=legacy)
            self.assertEqual(history["epoch"], [1, 2])
            self.assertEqual(history["val_loss"], expected["val_loss"])
            self.assertEqual(history["learning_rate"], expected["learning_rate"])
            # Explicit best.pt is a usable full training-state checkpoint too.
            selected_path, best = resume.resolve_checkpoint(paused_dir/"best.pt", root)
            self.assertEqual(selected_path.name, "best.pt")
            self.assertIn("optimizer_state_dict", best)

    def test_validation_interruption_can_resume_without_repeating_updates(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            _, _, baseline = self.fit(corpus, config)
            train_count = sum(r["count"] for r in corpus.records if r["trajectory_id"] in corpus.splits["train"])
            class BeforeValidation(StopAt):
                def should_stop(self, **position):
                    return position.get("stage") == "before_validation"
            _, _, interrupted = self.fit(corpus, config, control=BeforeValidation())
            saved = resume.load_checkpoint(interrupted/"last.pt")
            self.assertGreater(saved["resume_state"]["next_batch"], 0)
            _, history, _ = self.fit(corpus, config, checkpoint=interrupted/"last.pt")
            a, b = [resume.load_checkpoint(p/"last.pt") for p in (baseline, interrupted)]
            for key in a["model_state_dict"]:
                torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
            self.assertEqual(len(history["epoch"]), 1)

    def test_completed_early_stop_does_not_restart_optimizer_updates(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            config["training"].update(epochs=3, early_stopping=True, patience=1)
            config["scheduler"]["t_max_epochs"] = 4
            evaluate = self.ns["evaluate_reconstruction"]
            def constant_validation(*args, **kwargs):
                metrics, report = evaluate(*args, **kwargs)
                metrics["val_loss"] = 1.
                return metrics, report
            with patch.dict(self.ns, {"evaluate_reconstruction": constant_validation}):
                _, history, folder = self.fit(corpus, config)
                self.assertEqual(history["epoch"], [1, 2])
                original = resume.load_checkpoint(folder/"last.pt")
                _, continued, _ = self.fit(corpus, config, checkpoint=folder/"last.pt")
            self.assertEqual(continued["epoch"], [1, 2])
            self.assertEqual(resume.load_checkpoint(folder/"last.pt")["epoch"], original["epoch"])

    def test_checkpoint_selection_corrupt_last_and_atomic_temp(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            _, _, folder = self.fit(corpus, config)
            (folder/"last.pt").write_bytes(b"killed/corrupt")
            (folder/"last.pt.tmp").write_bytes(b"unfinished")
            with redirect_stdout(io.StringIO()):
                path, saved = resume.resolve_checkpoint("auto", Path(config["paths"]["output_root"]))
            self.assertEqual(saved["epoch"], 1)
            self.assertNotEqual(path.name, "last.pt.tmp")
            with self.assertRaises(Exception):
                resume.resolve_checkpoint(folder/"last.pt", root)

    def test_configuration_data_guard_and_concurrent_lock(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            _, _, folder = self.fit(corpus, config)
            saved = resume.load_checkpoint(folder/"last.pt")
            altered = copy.deepcopy(config)
            altered["training"]["batch_size"] *= 2
            with self.assertRaisesRegex(ValueError, "settings differ"):
                resume.validate_resume(saved, altered, corpus.splits, corpus.source_metadata)
            changed_source = dict(corpus.source_metadata, fingerprint="different")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                resume.validate_resume(saved, config, corpus.splits, changed_source)
            with resume.run_lock(folder):
                with self.assertRaisesRegex(RuntimeError, "Another trainer"):
                    with resume.run_lock(folder): pass
            with resume.run_lock(folder): pass  # released, without deleting lock files

    def test_slurm_deadline_and_rng_roundtrip(self):
        import random
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), patch("checkpoint_resume.subprocess.run") as command:
            command.return_value.stdout = "1-02:03:04\n"
            self.assertEqual(resume.slurm_remaining_seconds(), 93784)
        with patch("checkpoint_resume.slurm_remaining_seconds", return_value=120):
            control = resume.TrainingControl(walltime_margin_minutes=5)
            self.assertTrue(control.should_stop())
        state = resume.capture_rng()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        resume.restore_rng({"rng_state": state})
        actual = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2]); torch.testing.assert_close(expected[2], actual[2])

    def test_actual_notebook_resume_through_cluster_launcher(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            config["masking"]["evaluation_profiles"] = ["cold_blackout", "healthy", "mixed", "blackout_burst", "long_blackout"]
            train, val, test = self.ns["build_train_val_tensors"](corpus, config)
            self.ns["seed_training"](config["run"]["seed"])
            with redirect_stdout(io.StringIO()):
                _, _, folder = self.ns["train_sparse_reconstruction"](
                    self.ns["SparseRangeReconstructor"](config["model"]), train, val,
                    config=config, split_ids=corpus.splits, source_metadata=corpus.source_metadata,
                    control=StopAt("after_batch", 1, 2))
            notebook = root/"notebooks/train_sparse_reconstruction_map_temporal.ipynb"
            notebook.parent.mkdir()
            shutil.copyfile(ROOT/"notebooks"/notebook.name, notebook)
            helper = root/"scripts/carc/checkpoint_resume.py"
            helper.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/"scripts/carc/checkpoint_resume.py", helper)
            args = launcher.parse_args(["--repo-root", str(root), "--run-root", str(root),
                                       "--resume", str(folder), "--allow-local"])
            with patch.dict(os.environ, {}), redirect_stdout(io.StringIO()):
                launch = launcher.run_notebook(args)
            status = json.loads((launch/"status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(Path(status["training_run_dir"]), folder)
            saved = resume.load_checkpoint(folder/"last.pt")
            self.assertEqual(saved["epoch"], 1)
            self.assertTrue((folder/"holdout_report.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
