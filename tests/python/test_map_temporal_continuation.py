"""Continuation chains use committed progress and stop at completion or failure."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/carc"))
import slurm_continuation as continuation


class SlurmContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/"tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/"Map Temporal saved run"
        self.run.mkdir()
        (self.run/"last.pt").touch()
        self.script = self.root/"run_ares_map_temporal.slurm"
        self.script.touch()
        self.status = dict(status="paused", training_run_dir=str(self.run), job_id="123",
            repo_root=str(self.root), target_epochs=800, resume_progress=[17, 42])
        self.summary = dict(status="paused", completed_epochs=27, next_batch=45,
                            stop_reason="time budget reached")
        self.environment = patch.dict(os.environ, dict(SLURM_JOB_ID="123",
            SLURM_JOB_ACCOUNT="project", SLURM_JOB_QOS="normal"))
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def submit(self):
        (self.root/"status.json").write_text(json.dumps(self.status), encoding="utf-8")
        (self.run/"summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            return continuation.submit_continuation(self.root, self.script)

    def test_pause_queues_same_run_and_target_once_with_success_dependency(self):
        with patch.object(continuation.subprocess, "run") as command:
            command.return_value.stdout = "456;discovery\n"
            self.assertEqual(self.submit(), "456;discovery")
            args = command.call_args.args[0]
            self.assertEqual(args[-3:], [str(self.script.resolve()), str(self.run.resolve()), "800"])
            self.assertIn("--dependency=afterok:123", args)
            self.assertIn("--kill-on-invalid-dep=yes", args)
            self.assertIn("--account=project", args)
            self.assertIn("--qos=normal", args)
            self.assertTrue(command.call_args.kwargs["check"])
            self.status = json.loads((self.root/"status.json").read_text())
            self.assertEqual(self.status["continuation"]["status"], "submitted")
            self.assertEqual(self.submit(), "456;discovery")
            command.assert_called_once()

    def test_finished_checked_and_failed_launches_do_not_queue(self):
        with patch.object(continuation.subprocess, "run") as command:
            for status in ("completed", "checked", "failed", "interrupted", "running"):
                with self.subTest(status=status):
                    self.status["status"] = status
                    self.assertIsNone(self.submit())
            command.assert_not_called()

    def test_reaching_target_does_not_queue(self):
        self.summary.update(completed_epochs=800, next_batch=0)
        with patch.object(continuation.subprocess, "run") as command:
            self.assertIsNone(self.submit())
            command.assert_not_called()

    def test_cancellation_or_manual_stop_does_not_queue(self):
        with patch.object(continuation.subprocess, "run") as command:
            for reason in ("signal 15", "signal 2", "requested stop"):
                with self.subTest(reason=reason):
                    self.summary["stop_reason"] = reason
                    self.assertIsNone(self.submit())
            command.assert_not_called()

    def test_slurm_warning_can_queue(self):
        self.summary["stop_reason"] = "signal 10"
        with patch.object(continuation.signal, "SIGUSR1", 10, create=True), \
                patch.object(continuation.subprocess, "run") as command:
            command.return_value.stdout = "456\n"
            self.assertEqual(self.submit(), "456")

    def test_no_progress_or_uncommitted_pause_does_not_queue(self):
        with patch.object(continuation.subprocess, "run") as command:
            self.summary.update(completed_epochs=17, next_batch=42)
            with self.assertRaisesRegex(RuntimeError, "no checkpoint progress"):
                self.submit()
            self.summary.update(completed_epochs=27, status="failed")
            with self.assertRaisesRegex(RuntimeError, "does not confirm"):
                self.submit()
            command.assert_not_called()

    def test_missing_checkpoint_does_not_queue(self):
        (self.run/"last.pt").unlink()
        with patch.object(continuation.subprocess, "run") as command:
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint missing"):
                self.submit()
            command.assert_not_called()

    def test_only_original_job_can_queue(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "999"}), \
                patch.object(continuation.subprocess, "run") as command:
            with self.assertRaisesRegex(RuntimeError, "original active Slurm job"):
                self.submit()
            command.assert_not_called()

    def test_submission_failure_is_recorded_and_propagated(self):
        failure = subprocess.CalledProcessError(1, "sbatch", stderr="submission limit reached")
        with patch.object(continuation.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "submission limit reached"):
                self.submit()
        saved = json.loads((self.root/"status.json").read_text())
        self.assertEqual(saved["status"], "paused")
        self.assertEqual(saved["continuation"]["status"], "failed")
        self.assertTrue((self.run/"last.pt").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
