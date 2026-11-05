"""CPU checks for the endpoint SAITS model and reproducible training.

The existing NumPy-only suite remains usable when PyTorch is unavailable.
These tests never install dependencies, contact services, or use a GPU.
"""
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

TEST_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIRECTORY))
sys.path.insert(0, str(TEST_DIRECTORY.parents[1] / "scripts"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from lidar_reconstruction_data import introduce_lidar_perturbations, split_train_val_test
    from saits_endpoint_model import SAITSEndpointModel
    from train_saits_t4 import FeatureTransform, TrainConfig, evaluate_saits, train_saits
    from test_lidar_reconstruction_data import compact_frame


def small_config(epochs=2, **changes):
    settings = dict(
        epochs=epochs, batch_size=8, device="cpu", amp=False, seed=17,
        d_model=16, n_heads=2, d_ff=32, n_layers=1, dropout=0.0,
        learning_rate=1e-3, weight_decay=1e-4, patience=100,
        min_delta=0.0, checkpoint_every=1, log_every=1,
    )
    settings.update(changes)
    return TrainConfig(**settings)


def prepare_small_experiment(*, no_returns=False):
    source = compact_frame(302, no_returns=no_returns)
    partitions = split_train_val_test(source)
    return tuple(introduce_lidar_perturbations(
        part, seed=42 + index, n_outages=1, duration_s=(0.5, 1.0),
        warmup_s=0.5, min_healthy_gap_s=0.2, window_size=10,
        min_scored_steps=3,
    ) for index, part in enumerate(partitions))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional; install it to test SAITS training.")
class TestSaitsEndpointModel(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        torch.manual_seed(11)
        self.model = SAITSEndpointModel(
            n_steps=10, n_features=31, n_targets=14,
            d_model=16, n_heads=2, d_ff=32, n_layers=1, dropout=0.0,
        )

    def test_both_stages_and_gate_receive_finite_gradients(self):
        x = torch.randn(3, 10, 31)
        observed = torch.ones_like(x, dtype=torch.bool)
        observed[:, -3:, :14] = False
        x = x.masked_fill(~observed, 0.0)
        output = self.model(x, observed)
        for key in ("prediction", "stage1", "stage2"):
            self.assertEqual(tuple(output[key].shape), (3, 14))
            self.assertTrue(torch.isfinite(output[key]).all().item())
        target = torch.linspace(-0.2, 1.0, 42).reshape(3, 14)
        loss = (output["prediction"] - target).square().mean()
        stage_gradients = torch.autograd.grad(
            loss, (output["stage1"], output["stage2"]), retain_graph=True,
        )
        for gradient in stage_gradients:
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        loss.backward()
        gate_gradients = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(parameter.grad, msg=f"No gradient for {name}")
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), msg=name)
            if "gate" in name.lower():
                gate_gradients.append(parameter.grad)
        self.assertTrue(gate_gradients, "The stage-combination gate must be trainable.")
        self.assertGreater(sum(float(value.abs().sum()) for value in gate_gradients), 0.0)

    def test_diagonal_attention_and_fast_path_agree(self):
        self.model.eval()
        x = torch.randn(2, 10, 31)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, -1, :14] = False
        x = x.masked_fill(~mask, 0.0)
        with torch.no_grad():
            fast = self.model(x, mask)
            inspected = self.model(x, mask, return_attention=True)
        attention = inspected["attention"]
        self.assertEqual(tuple(attention.shape), (2, 10, 10))
        torch.testing.assert_close(torch.diagonal(attention, dim1=-2, dim2=-1),
                                   torch.zeros(2, 10), rtol=0, atol=1e-7)
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 10),
                                   rtol=1e-5, atol=1e-6)
        self.assertTrue((attention >= 0).all().item())
        torch.testing.assert_close(fast["prediction"], inspected["prediction"],
                                   rtol=1e-5, atol=1e-6)

    def test_all_missing_features_are_finite_and_one_step_is_rejected(self):
        output = self.model(torch.zeros(2, 10, 31), torch.zeros(2, 10, 31, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output["prediction"]).all().item())
        with self.assertRaises(ValueError):
            SAITSEndpointModel(n_steps=1, n_features=31, n_targets=14,
                               d_model=16, n_heads=2, d_ff=32, n_layers=1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional; install it to test SAITS training.")
class TestFeatureTransform(unittest.TestCase):
    def test_statistics_use_unique_training_rows_and_preserve_missingness(self):
        prepared = prepare_small_experiment()[0]
        prepared["X"] = prepared["X"].copy()
        rows = np.arange(len(prepared["X"]), dtype=float)
        prepared["X"].iloc[:, 14] = rows
        prepared["X"].iloc[:, 15] = 7.0
        transform = FeatureTransform.fit(prepared)
        state = transform.to_dict()
        self.assertAlmostEqual(state["context_mean"][0], rows.mean())
        self.assertAlmostEqual(state["context_std"][0], rows.std(ddof=0))
        self.assertEqual(state["context_mean"][1], 7.0)
        self.assertEqual(state["context_std"][1], 1.0)
        raw = prepared["X"].to_numpy(copy=True)
        transformed = transform.transform_inputs(raw)
        self.assertEqual(transformed.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(np.isnan(transformed), np.isnan(raw))
        np.testing.assert_allclose(transformed[:, 14].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(transformed[:, 14].std(ddof=0), 1.0, atol=1e-6)
        np.testing.assert_array_equal(transformed[:, 15], np.zeros(len(raw), dtype=np.float32))
        np.testing.assert_allclose(transformed[:, :7], raw[:, :7] / 7, equal_nan=True)
        np.testing.assert_allclose(transformed[:, 7:14], raw[:, 7:14], equal_nan=True)
        # An extreme held-out value must not refit the learned training statistics.
        held_out = raw[:1].copy()
        held_out[:, 14] = 1e9
        transform.transform_inputs(held_out)
        self.assertEqual(transform.to_dict(), state)

    def test_target_units_round_trip_and_transform_round_trip(self):
        prepared = prepare_small_experiment()[0]
        transform = FeatureTransform.fit(prepared)
        targets = prepared["y"].to_numpy(copy=True)
        normalized = transform.transform_targets(targets)
        self.assertEqual(normalized.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(normalized[:, :7], targets[:, :7] / 7)
        np.testing.assert_allclose(normalized[:, 7:], targets[:, 7:])
        np.testing.assert_allclose(transform.inverse_targets(normalized), targets,
                                   rtol=1e-6, atol=1e-6)
        restored = FeatureTransform.from_dict(transform.to_dict())
        np.testing.assert_array_equal(restored.transform_inputs(prepared["X"].to_numpy()),
                                      transform.transform_inputs(prepared["X"].to_numpy()))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional; install it to test SAITS training.")
class TestSaitsTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = prepare_small_experiment()

    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def test_two_epochs_evaluate_and_restore_the_best_checkpoint(self):
        train, validation, test = self.prepared
        result = train_saits(train, validation, output_dir=self.output, config=small_config())
        self.assertEqual(str(result.device), "cpu")
        self.assertEqual(len(result.history), 2)
        self.assertTrue(1 <= result.best_epoch <= 2)
        self.assertTrue(np.isfinite(result.best_metric))
        self.assertTrue((self.output / "last.pt").is_file())
        checkpoint = torch.load(self.output / "best.pt", map_location="cpu", weights_only=True)
        for name, parameter in result.model.state_dict().items():
            torch.testing.assert_close(parameter.cpu(), checkpoint["model_state_dict"][name],
                                       rtol=0, atol=0)
        metrics = evaluate_saits(result, test, split_name="test")
        for name in ("distance_mae_m", "return_fraction_mae",
                     "persistence_distance_mae_m", "persistence_return_fraction_mae"):
            self.assertTrue(np.isfinite(metrics[name]), msg=name)
            self.assertGreaterEqual(metrics[name], 0.0)
        self.assertEqual(metrics["scored_timestep_count"],
                         int(test["score_mask"].any(axis=1).sum()))
        self.assertFalse(metrics["target_no_return_only"])
        self.assertNotIn("test_prepared", inspect.signature(train_saits).parameters)
        self.assertNotIn("test", inspect.signature(train_saits).parameters)

    def test_no_return_only_evaluation_is_identified(self):
        train, validation, _ = self.prepared
        result = train_saits(train, validation, output_dir=self.output, config=small_config(epochs=1))
        no_return_test = prepare_small_experiment(no_returns=True)[2]
        metrics = evaluate_saits(result, no_return_test)
        self.assertTrue(metrics["target_no_return_only"])
        self.assertEqual(metrics["persistence_distance_mae_m"], 0.0)
        self.assertEqual(metrics["persistence_return_fraction_mae"], 0.0)

    def test_resume_two_to_four_epochs_matches_uninterrupted_training(self):
        train, validation, _ = self.prepared
        continuous = train_saits(train, validation, output_dir=self.output / "continuous",
                                 config=small_config(epochs=4))
        train_saits(train, validation, output_dir=self.output / "resumed",
                    config=small_config(epochs=2))
        resumed = train_saits(train, validation, output_dir=self.output / "resumed",
                              config=small_config(epochs=4), resume=True)
        self.assertEqual(len(resumed.history), 4)
        self.assertEqual(resumed.best_epoch, continuous.best_epoch)
        self.assertEqual(resumed.best_metric, continuous.best_metric)
        for name, parameter in continuous.model.state_dict().items():
            torch.testing.assert_close(parameter, resumed.model.state_dict()[name], rtol=0, atol=0)
        loss_columns = [name for name in continuous.history if "loss" in name.lower()]
        self.assertTrue(loss_columns, "Training history must expose the optimized losses.")
        pd.testing.assert_frame_equal(continuous.history[loss_columns], resumed.history[loss_columns],
                                      check_exact=True)

    def test_resume_rejects_changed_inputs_or_model_hyperparameters(self):
        train, validation, _ = self.prepared
        train_saits(train, validation, output_dir=self.output, config=small_config(epochs=1))
        altered = dict(train)
        altered["X"] = train["X"].copy()
        altered["X"].iloc[0, 14] += 0.25
        with self.assertRaises(ValueError):
            train_saits(altered, validation, output_dir=self.output,
                        config=small_config(epochs=2), resume=True)
        with self.assertRaises(ValueError):
            train_saits(train, validation, output_dir=self.output,
                        config=small_config(epochs=2, learning_rate=2e-3), resume=True)
        altered_mask = dict(train)
        altered_mask["target_mask"] = train["target_mask"].copy()
        endpoint = np.flatnonzero(train["score_mask"].any(axis=1).to_numpy())[0]
        altered_mask["target_mask"].iloc[endpoint, 0] = False
        with self.assertRaisesRegex(ValueError, "Resume data/outages changed"):
            train_saits(altered_mask, validation, output_dir=self.output,
                        config=small_config(epochs=2), resume=True)

    def test_resume_restores_dropout_random_state(self):
        train, validation, _ = self.prepared
        config = small_config(epochs=3, dropout=0.2)
        continuous = train_saits(train, validation, output_dir=self.output / "continuous", config=config)
        train_saits(train, validation, output_dir=self.output / "resumed",
                    config=small_config(epochs=1, dropout=0.2))
        resumed = train_saits(train, validation, output_dir=self.output / "resumed",
                              config=config, resume=True)
        for name, parameter in continuous.model.state_dict().items():
            torch.testing.assert_close(parameter, resumed.model.state_dict()[name], rtol=0, atol=0)
        np.testing.assert_array_equal(continuous.history.train_loss, resumed.history.train_loss)

    def test_nonfinite_clean_labels_are_rejected(self):
        train, validation, _ = self.prepared
        invalid = dict(train)
        invalid["y"] = train["y"].copy()
        invalid["y"].iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            train_saits(invalid, validation, output_dir=self.output, config=small_config(epochs=1))


if __name__ == "__main__":
    unittest.main()
