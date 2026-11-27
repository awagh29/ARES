"""Safe optimizer-preserving batch changes and measured CUDA sizing."""
from contextlib import redirect_stdout
import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts/carc"), str(ROOT/"tests/python")]
import batch_tuning
import checkpoint_resume as resume
import run_map_temporal as launcher
from test_map_temporal_sparse_notebook import namespace, fixture_workspace
import test_map_temporal_resume as resume_tests
from test_map_temporal_resume import StopAt


class BatchTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resume_tests.ResumeTests.setUpClass()
        cls.fitter = resume_tests.ResumeTests()
        cls.ns = resume_tests.ResumeTests.ns

    def assert_same_training_state(self, a, b):
        for key, value in a["model_state_dict"].items():
            torch.testing.assert_close(value, b["model_state_dict"][key], rtol=0, atol=0)
        for key, state in a["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, b["optimizer_state_dict"]["state"][key][name], rtol=0, atol=0)
        self.assertEqual(a["scheduler_state_dict"], b["scheduler_state_dict"])
        self.assertEqual(a["scaler_state_dict"], b["scaler_state_dict"])

    def test_changed_batch_starts_at_saved_epoch_without_resetting_state(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            config["training"]["epochs"] = 2
            config["scheduler"]["t_max_epochs"] = 4
            _, _, folder = self.fitter.fit(corpus, config, control=StopAt("epoch_end", 1, 0))
            original = resume.load_checkpoint(folder/"last.pt")
            _, history, same = self.fitter.fit(corpus, config, checkpoint=folder/"last.pt",
                control=StopAt("before_batch", 2, 0), batch_size_request=dict(batch_size=8, source="test"))
            self.assertEqual(same, folder)
            changed = resume.load_checkpoint(folder/"last.pt")
            self.assert_same_training_state(original, changed)
            self.assertEqual(changed["epoch"], 1)
            self.assertEqual(changed["config"]["training"]["batch_size"], 8)
            self.assertEqual(changed["config"]["training"]["batch_policy"], "packed")
            self.assertEqual(changed["batch_size_changes"][0]["epoch"], 2)
            self.assertEqual(changed["history"], original["history"])
            config = changed["config"]
            _, history, _ = self.fitter.fit(corpus, config, checkpoint=folder/"last.pt")
            self.assertEqual(history["epoch"], [1, 2])

    def test_partial_epoch_finishes_with_old_batch_and_persists_pending_request(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            config["training"]["epochs"] = 2
            config["scheduler"]["t_max_epochs"] = 4
            _, _, reference = self.fitter.fit(corpus, config, control=StopAt("epoch_end", 1, 0))
            reference_state = resume.load_checkpoint(reference/"last.pt")
            _, _, folder = self.fitter.fit(corpus, config, control=StopAt("after_batch", 1, 1))
            self.fitter.fit(corpus, config, checkpoint=folder/"last.pt", control=StopAt("epoch_end", 1, 0),
                            batch_size_request=dict(batch_size=8, source="test"))
            pending = resume.load_checkpoint(folder/"last.pt")
            self.assert_same_training_state(reference_state, pending)
            self.assertEqual(pending["config"]["training"]["batch_size"], config["training"]["batch_size"])
            self.assertEqual(pending["pending_batch_size"]["batch_size"], 8)
            self.fitter.fit(corpus, config, checkpoint=folder/"last.pt", control=StopAt("before_batch", 2, 0))
            applied = resume.load_checkpoint(folder/"last.pt")
            self.assertEqual(applied["config"]["training"]["batch_size"], 8)
            self.assertIsNone(applied["pending_batch_size"])
            self.assert_same_training_state(pending, applied)

    def test_packed_batches_cover_each_frame_once_and_reset_flight_history(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            data, _, _ = self.ns["build_train_val_tensors"](corpus, config)
            sampler = self.ns["TrajectoryBatchSampler"](data, 8, 42, pack=True)
            batches = list(sampler)
            self.assertEqual(len(batches), math.ceil(len(data)/8))
            self.assertTrue(all(len(b) == 8 for b in batches[:-1]))
            self.assertEqual(sorted(i for b in batches for i in b), list(range(len(data))))
            for offset in data.offsets[:-1]:
                self.assertEqual(int(data[int(offset)]["present"].sum()), 1)

    def test_memory_search_and_command_options(self):
        selected, trials = batch_tuning.search_batch_size(lambda b: dict(fits=b <= 150), start=16, maximum=512)
        self.assertTrue(142 <= selected <= 150)
        self.assertTrue(any(not t["fits"] for t in trials))
        small, _ = batch_tuning.search_batch_size(lambda b: dict(fits=b <= 3), start=16, maximum=64)
        self.assertTrue(1 <= small <= 3)
        with self.assertRaises(RuntimeError):
            batch_tuning.search_batch_size(lambda b: dict(fits=False), start=16, maximum=32)
        options = launcher.parse_args(["--resume", "--auto-batch-size", "--gpu-memory-fraction", ".90"])
        self.assertEqual(options.resume, "auto")
        self.assertTrue(options.auto_batch_size)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["--batch-size", "64", "--auto-batch-size"])

    def test_launcher_hands_tuned_request_to_partial_epoch_checkpoint(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            config["masking"]["evaluation_profiles"] = ["cold_blackout", "healthy", "mixed", "blackout_burst", "long_blackout"]
            train, val, _ = self.ns["build_train_val_tensors"](corpus, config)
            self.ns["seed_training"](42)
            with redirect_stdout(io.StringIO()):
                _, _, folder = self.ns["train_sparse_reconstruction"](
                    self.ns["SparseRangeReconstructor"](config["model"]), train, val, config=config,
                    split_ids=corpus.splits, source_metadata=corpus.source_metadata, control=StopAt("after_batch", 1, 1))
            notebook = root/"notebooks/train_sparse_reconstruction_map_temporal.ipynb"
            notebook.parent.mkdir()
            shutil.copyfile(ROOT/"notebooks"/notebook.name, notebook)
            helper = root/"scripts/carc/checkpoint_resume.py"
            helper.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/"scripts/carc/checkpoint_resume.py", helper)
            args = launcher.parse_args(["--repo-root", str(root), "--run-root", str(root),
                "--resume", str(folder), "--auto-batch-size", "--allow-local"])
            with patch.dict(os.environ, {}), patch("batch_tuning.tune_batch_size", return_value=(8, {})) as probe, redirect_stdout(io.StringIO()):
                launch = launcher.run_notebook(args)
            probe.assert_called_once()
            self.assertEqual(json.loads((launch/"status.json").read_text())["selected_batch_size"], 8)
            saved = resume.load_checkpoint(folder/"last.pt")
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(saved["pending_batch_size"]["batch_size"], 8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for real scratch-step sizing")
    def test_real_cuda_probe_preserves_checkpoint_weights_optimizer_and_rng(self):
        with fixture_workspace(self.ns) as (root, config, corpus):
            _, _, folder = self.fitter.fit(corpus, config)
            saved = resume.load_checkpoint(folder/"last.pt")
            original = copy.deepcopy(saved)
            config = copy.deepcopy(config)
            config["runtime"].update(device="cuda", amp_dtype="bf16" if torch.cuda.is_bf16_supported() else "fp16")
            # A CPU checkpoint has no enabled FP16 scaler; probe a new scratch
            # optimizer here while separately verifying the saved state is untouched.
            data, _, _ = self.ns["build_train_val_tensors"](corpus, config)
            expected_rng = torch.get_rng_state().clone()
            with redirect_stdout(io.StringIO()):
                selected, report = batch_tuning.tune_batch_size(self.ns, data, config,
                    saved if torch.cuda.is_bf16_supported() else None,
                    root/"batch_tuning.json", maximum=16, fraction=.5, steps=1)
            self.assertEqual(selected, 16)
            self.assertEqual(report["status"], "completed")
            self.assertTrue(all(t["fits"] for t in report["trials"]))
            torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assert_same_training_state(original, saved)
            self.assertTrue((root/"batch_tuning.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
