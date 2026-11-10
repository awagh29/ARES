"""Integration invariants for actual reconstruction, constrained control and recovery."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("scipy") is not None
if AVAILABLE:
    import torch
    from ares_safe_training.barrier import BarrierConfig, BarrierSafetyFilter
    from ares_safe_training.config import PipelineConfig
    from ares_safe_training.control import nominal_acceleration
    from ares_safe_training.data import FeatureNormalizer
    from ares_safe_training.dynamics import GuidanceDynamics
    from ares_safe_training.models import AugmentedSAITS, ModelConfig
    from ares_safe_training.pipeline import finalize_deployment, run_pipeline
    from ares_safe_training.training import (TensorNormalizer, _restore_stage, immutable_signature,
        model_state_sha256, reconstruction_loss, restore_rng, rng_state, run_rollout, save_checkpoint)
    from test_safe_dynamics import PARAMETERS


@unittest.skipUnless(AVAILABLE, "PyTorch and SciPy required.")
class TestTrainingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.config = PipelineConfig(device="cpu", amp=False, model=ModelConfig(
            d_model=8, n_heads=2, d_ff=16, state_layers=1, dropout=0))
        self.model = AugmentedSAITS(self.config.model).eval()
        self.history = torch.randn(2, 50, 31)
        self.hm = torch.rand(2, 50, 31) > .3
        self.current = torch.randn(2, 34)
        self.cm = torch.rand(2, 34) > .3

    def test_masked_values_cannot_leak_into_reconstruction(self):
        with torch.no_grad():
            a = self.model(self.history, self.hm, self.current, self.cm)["physical"]
            b = self.model(self.history.masked_fill(~self.hm, 12345), self.hm,
                           self.current.masked_fill(~self.cm, -12345), self.cm)["physical"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(bool((a[:, :7] > 0).all() and (a[:, :7] < 7).all()))
        self.assertTrue(bool((a[:, 7:] > 0).all() and (a[:, 7:] < 1).all()))

    def test_both_branches_receive_supervised_gradients(self):
        out = self.model(self.history, self.hm, self.current, self.cm)
        reconstruction_loss(out, torch.zeros(2, 14), torch.ones(2, 14, dtype=torch.bool), self.config).backward()
        for branch in (self.model.temporal, self.model.state):
            size = sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None)
            self.assertGreater(size, 0)

    def test_numpy_and_online_tensor_normalizers_agree(self):
        normalizer = FeatureNormalizer(np.array([1., 2., -7.]), np.array([3., 4., 1.]),
            np.arange(17, dtype=np.float32), np.arange(1, 18, dtype=np.float32),
            ("test-train-only",), 100, 42, ("mixed", "healthy"))
        tensor = TensorNormalizer(normalizer, torch.device("cpu"))
        torch.testing.assert_close(tensor.history(self.history, self.hm), torch.from_numpy(
            normalizer.transform_history_numpy(self.history.numpy(), self.hm.numpy())))
        torch.testing.assert_close(tensor.state(self.current, self.cm), torch.from_numpy(
            normalizer.transform_state_numpy(self.current.numpy(), self.cm.numpy())))

    def test_downstream_only_gradient_crosses_qp_allocation_and_actual_plant(self):
        # An admissible numerical fixture isolates the control path: no
        # reconstruction loss contributes to this backward pass.
        physical = self.model(self.history[:1], self.hm[:1], self.current[:1], self.cm[:1])["physical"]
        state = torch.zeros(1, 13)
        state[:, 2], state[:, 12] = -7, .981
        reference = {"p": state[:, :3] + torch.tensor([[.05, 0., 0.]]),
                     "v": torch.zeros(1, 3), "a": torch.zeros(1, 3),
                     "rho": torch.tensor([2.5]), "rhodot": torch.zeros(1), "rhoddot": torch.zeros(1)}
        self.config.nominal_repulsion_gain = .01
        nominal = nominal_acceleration(state, physical, reference, self.config)
        barrier = BarrierSafetyFilter(BarrierConfig(.2, (.01,) * 7))
        filtered = barrier(state[:, :3], state[:, 3:6], state[:, 6:9], physical,
            reference, torch.zeros(1, 2), nominal_acceleration=nominal[:, :2])
        self.assertTrue(bool(filtered["feasible"].all()))
        dynamics = GuidanceDynamics(PARAMETERS)
        acceleration = torch.cat((filtered["acceleration"].float(), torch.zeros(1, 1)), 1)
        command = dynamics.acceleration_to_commands(acceleration, state[:, 8], torch.zeros(1))
        future = dynamics(state, command, .1)
        (future[:, 0] - .2).square().sum().backward()
        gradient = sum(float(p.grad.abs().sum()) for p in self.model.parameters() if p.grad is not None)
        self.assertGreater(gradient, 1e-12)

    def test_checkpoint_restores_optimizer_and_rng_and_rejects_changed_signature(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        out = self.model(self.history, self.hm, self.current, self.cm)
        out["normalized"].sum().backward(); optimizer.step()
        before = model_state_sha256(self.model.state_dict())
        rng = rng_state()
        expected = torch.rand(5)
        payload = {"schema_version": "ares-augmented-saits-barrier-training-v1", "stage": "reconstruction",
            "signature": "test", "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, payload)
            with torch.no_grad():
                next(self.model.parameters()).add_(1)
            _restore_stage(path, "reconstruction", "test", self.model, optimizer, scheduler, scaler)
            self.assertEqual(model_state_sha256(self.model.state_dict()), before)
            torch.testing.assert_close(torch.rand(5), expected)
            with self.assertRaisesRegex(ValueError, "changed"):
                _restore_stage(path, "reconstruction", "other", self.model, optimizer, scheduler, scaler)

    def test_finalize_recovers_after_test_was_sealed_without_retesting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "stage2").mkdir()
            checkpoint = root / "stage2" / "best.pt"; checkpoint.write_bytes(b"selected-test-fixture")
            seal = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            def publish(_checkpoint, destination):
                destination.mkdir()
                (destination / "deployment.pt").write_bytes(b"fixture")
                (destination / "deployment.json").write_text(json.dumps({
                    "provenance": {"source_checkpoint_sha256": seal["checkpoint_sha256"]},
                    "artifacts": {"deployment.pt": {"sha256": hashlib.sha256(b"fixture").hexdigest()}}}))
            with mock.patch("export_augmented_saits_barrier.export_checkpoint", side_effect=publish) as export:
                result = finalize_deployment(root, seal)
                finalize_deployment(root, seal)
                self.assertEqual(export.call_count, 1)
                self.assertEqual(json.loads((root / "summary.json").read_text()), result)
                (root / "deployment" / "deployment.pt").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    finalize_deployment(root, seal)

    def test_configuration_rejects_nan_and_fractional_epoch_counts(self):
        for change in ({"tracking_weight": float("nan")}, {"pretrain_epochs": 1.5},
                       {"control_dt": .2}, {"maximum_yaw_rate_radps": 1.}):
            with self.assertRaises(ValueError):
                PipelineConfig.from_dict(dict(self.config.to_dict(), **change))

    def test_inadmissible_initial_barrier_executes_zero_control_steps(self):
        normalizer = FeatureNormalizer(np.zeros(3), np.ones(3), np.zeros(17), np.ones(17),
                                      ("test-train-only",), 10, 42, ("mixed", "healthy"))
        trajectory = SimpleNamespace(initial_state=np.array([0., 0., -7.] + [0.] * 10, np.float32),
            goal=np.array([5., 0., -7.], np.float32), run_id="inadmissible-fixture", run_index=1,
            safety_pass=True, collision=False)
        scene = SimpleNamespace(features=lambda state: state.new_tensor([[.5] * 7 + [1.] * 7]).expand(len(state), -1),
            clearance=lambda positions: positions.new_full(positions.shape[:-1], 10.))
        barrier = BarrierSafetyFilter(BarrierConfig(.2, (.05,) * 7))
        result = run_rollout(self.model, barrier, GuidanceDynamics(PARAMETERS), scene,
                             [trajectory], normalizer, self.config, steps=1)
        record = result["records"][0]
        self.assertTrue(record["initial_truth_clearance_ok"])
        self.assertFalse(record["initial_cbf_valid"])
        self.assertFalse(record["initial_state_safe"])
        self.assertEqual(record["steps_simulated"], 0)
        self.assertEqual(record["termination"], "invalid-initial-state")
        scene.clearance = lambda positions: positions.new_full(positions.shape[:-1], -1.)
        record = run_rollout(self.model, barrier, GuidanceDynamics(PARAMETERS), scene,
                            [trajectory], normalizer, self.config, steps=1)["records"][0]
        self.assertTrue(record["collision"])
        self.assertEqual(record["steps_simulated"], 0)

    def test_sealed_test_cannot_resume_fitting_even_when_epoch_limits_increase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "stage2").mkdir()
            checkpoint = root / "stage2" / "best.pt"; checkpoint.write_bytes(b"immutable-selected")
            reservation = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            (root / "test_evaluation_intent.json").write_text(json.dumps(reservation))
            (root / "test_evaluation.json").write_text(json.dumps(reservation))
            (root / "experiment.json").write_text(json.dumps({"signature": "test", "sessions": []}))
            with mock.patch("ares_safe_training.pipeline.build_dataset"), \
                 mock.patch("ares_safe_training.pipeline.experiment_metadata", return_value={}), \
                 mock.patch("ares_safe_training.pipeline.immutable_signature", return_value="test"), \
                 mock.patch("ares_safe_training.pipeline.train_stage_one") as pretrain, \
                 mock.patch("ares_safe_training.pipeline.train_stage_two") as finetune, \
                 mock.patch("ares_safe_training.pipeline.finalize_deployment", return_value=reservation) as finish:
                self.config.finetune_epochs += 10
                result = run_pipeline(ROOT, root, "unused", self.config, resume=True)
                self.assertEqual(result, reservation)
                self.assertEqual(finish.call_count, 1)
                with self.assertRaisesRegex(ValueError, "fitting is closed"):
                    run_pipeline(ROOT, root, "unused", self.config, stage="finetune", resume=True)
                pretrain.assert_not_called(); finetune.assert_not_called()
                self.assertEqual(checkpoint.read_bytes(), b"immutable-selected")

    def test_interrupted_test_reservation_rejects_replaced_weights_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "stage2").mkdir()
            (root / "stage2" / "best.pt").write_bytes(b"replaced-after-test-started")
            (root / "test_evaluation_intent.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
            (root / "experiment.json").write_text(json.dumps({"signature": "test", "sessions": []}))
            with mock.patch("ares_safe_training.pipeline.build_dataset"), \
                 mock.patch("ares_safe_training.pipeline.experiment_metadata", return_value={}), \
                 mock.patch("ares_safe_training.pipeline.immutable_signature", return_value="test"), \
                 mock.patch("ares_safe_training.pipeline.train_stage_one") as fit:
                with self.assertRaisesRegex(ValueError, "different weights"):
                    run_pipeline(ROOT, root, "unused", self.config, resume=True)
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
