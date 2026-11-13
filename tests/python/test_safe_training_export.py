"""Deployment integrity and numerical fixtures, using tiny test-only weights.

These synthetic checkpoints exercise packaging; they are not trained models or
evidence about reconstruction, flight performance, safety, or MATLAB parity.
"""
from dataclasses import asdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
DEPENDENCIES_AVAILABLE = (importlib.util.find_spec("torch") is not None
                          and importlib.util.find_spec("scipy") is not None)
if DEPENDENCIES_AVAILABLE:
    import torch
    from scipy.io import loadmat
    from ares_safe_training.barrier import BarrierConfig, BarrierSafetyFilter
    from ares_safe_training.config import PipelineConfig
    from ares_safe_training.data import FeatureNormalizer
    from ares_safe_training.dynamics import CALIBRATION_SCHEMA, acceleration_to_commands
    from ares_safe_training.models import AugmentedSAITS, ModelConfig
    import export_augmented_saits_barrier as exporter


def checkpoint_fixture():
    """Explicit numerical-test calibration and random weights; never production defaults."""
    torch.manual_seed(13)
    model_config = ModelConfig(d_model=8, n_heads=2, d_ff=16, temporal_layers=1, state_layers=1, dropout=0)
    model = AugmentedSAITS(model_config).eval()
    barrier_config = BarrierConfig(disturbance_bound_mps2=.2, range_error_bound_m=(.05,) * 7)
    barrier = BarrierSafetyFilter(barrier_config).eval()
    normalizer = FeatureNormalizer(np.array([1., 2., -7.]), np.array([3., 4., 1.]),
        np.zeros(17), np.ones(17), ("fixture-train-only",), 123, 42, ("mixed", "healthy"))
    calibration = {"schemaVersion": CALIBRATION_SCHEMA, "calibrationComplete": True,
        "parameters": {"mass_kg": .1, "gravity_mps2": 9.81, "roll_kp": 3402.97, "roll_kd": 116.67,
            "pitch_kp": 3402.97, "pitch_kd": 116.67, "yaw_rate_gain": 1950., "thrust_gain": 3900.,
            "roll_limit_rad": math.radians(25), "pitch_limit_rad": math.radians(25),
            "yaw_rate_limit_radps": .5},
        "tracking_error": {"bound_dimension": "horizontal-XY", "xy_bound_mps2": .2}, "calibration_sha256": "a" * 64,
        "software_design_envelope": {"policy_step_s": .1, "truth_sample_step_s": .001,
            "acceleration_lower_mps2": [-1., -1., -4.], "acceleration_upper_mps2": [1., 1., 4.],
            "acceleration_slew_mps3": [1., 1., 1.], "yaw_rate_limit_radps": .5,
            "roll_limit_rad": math.radians(25), "pitch_limit_rad": math.radians(25),
            "thrust_lower_n": .581, "thrust_upper_n": 1.682},
        "reference": {"model_sha256": "b" * 64}}
    return {"schema_version": exporter.TRAINING_SCHEMA, "stage": "closed_loop",
        "checkpoint_role": "best", "selected_weights": True, "epoch": 3, "best_score": .375,
        "signature": "c" * 64, "source_signature": exporter.runtime_source_sha256(),
        "model": model.state_dict(), "barrier": barrier.state_dict(),
        "model_config": asdict(model_config), "barrier_config": asdict(barrier_config),
        "normalizer": normalizer.to_dict(), "config": PipelineConfig(model=model_config).to_dict(),
        "dynamics_calibration": calibration,
        "uncertainty": {"schema_version": "ares-empirical-reconstruction-bound-v1", "certified": False,
            "bounds_m": [.05] * 7, "model_state_sha256": exporter.state_sha256(model.state_dict()),
            "quantile": 1., "engineering_margin_m": .01, "sample_counts": [100] * 7},
        "optimizer": {"never_deploy_optimizer": torch.ones(1)}, "rng": {"never_deploy_rng": 123},
        "history": [{"epoch": 3, "selection_score": .375}]}


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "PyTorch and SciPy are required for deployment export.")
class TestAugmentedSaitsBarrierExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.payload = checkpoint_fixture()
        # Adversarial extra runtime/experiment fields must never enter any
        # deployable file, even when nested in otherwise allowed dictionaries.
        forbidden = {"simulator_scene": {"box_min_ned_m": [[123456.789, 98765., -15.]]},
                     "obstacle_centers": "MAP_SECRET_DO_NOT_DEPLOY"}
        for key in ("config", "normalizer", "uncertainty", "dynamics_calibration"):
            cls.payload[key].update(copy.deepcopy(forbidden))
        cls.payload["experiment"] = copy.deepcopy(forbidden)
        cls.source = cls.root / "stage2-best.pt"
        torch.save(cls.payload, cls.source)
        cls.before = cls.source.read_bytes()
        cls.destination = exporter.export_checkpoint(cls.source, cls.root / "deployment")
        cls.deployed = torch.load(cls.destination / "deployment.pt", map_location="cpu", weights_only=True)
        cls.manifest = json.loads((cls.destination / "deployment.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def assert_rejected(self, update, text):
        payload = copy.deepcopy(self.payload)
        update(payload)
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            source = Path(folder) / "candidate.pt"
            output = Path(folder) / "deployment"
            torch.save(payload, source)
            with self.assertRaisesRegex(ValueError, text):
                exporter.export_checkpoint(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(folder).glob(".*.export-*")), [])

    def test_rejects_pretrain_last_or_unselected_checkpoints(self):
        self.assert_rejected(lambda p: p.update(stage="reconstruction"), "Only Stage 2")
        self.assert_rejected(lambda p: p.update(checkpoint_role="last", selected_weights=False), "selected Stage 2 best")
        self.assert_rejected(lambda p: p.pop("selected_weights"), "selected Stage 2 best")
        self.assert_rejected(lambda p: p.update(best_score=.4), "checkpoint history")

    def test_bounds_are_bound_to_selected_weights_and_barrier(self):
        self.assert_rejected(lambda p: p["uncertainty"].update(model_state_sha256="0" * 64), "different model weights")
        self.assert_rejected(lambda p: p["uncertainty"].update(bounds_m=[.1] * 7), "bounds differ")
        self.assert_rejected(lambda p: p["barrier"]["calibration_bounds"].fill_(.1), "calibration buffer differs")

    def test_rejects_runtime_source_drift_and_modified_structural_buffers(self):
        self.assert_rejected(lambda p: p.pop("source_signature"), "runtime source_signature")
        self.assert_rejected(lambda p: p["source_signature"].update({"models.py": "0" * 64}), "Runtime source differs")
        self.assert_rejected(lambda p: p["model"]["physical_scale"].fill_(1), "structural/calibration buffer differs")

    def test_requires_consistent_calibration_and_preprocessing(self):
        self.assert_rejected(lambda p: p["dynamics_calibration"].update(calibrationComplete=False), "completed authoritative")
        self.assert_rejected(lambda p: p["dynamics_calibration"]["tracking_error"].update(xy_bound_mps2=.3), "disturbance bound differs")
        self.assert_rejected(lambda p: p["normalizer"]["history_std"].__setitem__(0, 1), "history_std is inconsistent")
        self.assert_rejected(lambda p: p["config"].pop("nominal_position_gain"), "online-controller defaults")

    def test_rejects_corrupt_parameter_shapes_dtypes_and_nonfinite_weights(self):
        key = "decoder.3.bias"
        self.assert_rejected(lambda p: p["model"].update({key: torch.zeros(15)}), "shape/dtype differs")
        self.assert_rejected(lambda p: p["model"].update({key: p["model"][key].half()}), "shape/dtype differs")
        self.assert_rejected(lambda p: p["model"][key].fill_(float("nan")), "Nonfinite model")

    def test_preserves_selected_weights_and_source_bytes(self):
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(self.deployed["provenance"]["source_checkpoint_sha256"], hashlib.sha256(self.before).hexdigest())
        for name in ("model", "barrier"):
            self.assertEqual(exporter.state_sha256(self.deployed[name]), exporter.state_sha256(self.payload[name]))
        self.assertFalse(any(key in self.deployed for key in ("optimizer", "rng", "scheduler", "scaler", "history", "config")))
        self.assertEqual(self.deployed["training_selection"]["epoch"], 3)

    def test_no_overwrite_and_optional_onnx_failure_leave_outputs_untouched(self):
        before = {path.name: path.read_bytes() for path in self.destination.iterdir()}
        with self.assertRaises(FileExistsError):
            exporter.export_checkpoint(self.source, self.destination)
        self.assertEqual({path.name: path.read_bytes() for path in self.destination.iterdir()}, before)
        with mock.patch.object(exporter.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "optional onnx"):
                exporter.export_checkpoint(self.source, self.root / "missing-onnx", export_onnx=True)
        self.assertFalse((self.root / "missing-onnx").exists())

    def test_partial_export_failure_removes_only_its_staging_directory(self):
        output = self.root / "interrupted-export"
        with mock.patch.object(exporter, "savemat", side_effect=OSError("fixture write failure")):
            with self.assertRaisesRegex(OSError, "fixture write failure"):
                exporter.export_checkpoint(self.source, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".interrupted-export.export-*")), [])
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_no_map_fields_in_any_deployable_container(self):
        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(key, ("simulator_scene", "obstacle_centers", "box_min_ned_m", "box_max_ned_m", "experiment"))
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, str):
                self.assertNotIn("MAP_SECRET_DO_NOT_DEPLOY", value)
        visit(self.deployed)
        visit(self.manifest)
        for path in self.destination.glob("*.json"):
            self.assertNotIn("MAP_SECRET_DO_NOT_DEPLOY", path.read_text())
        mat = loadmat(self.destination / "model.mat", simplify_cells=True)["deployment"]
        for key, value in mat.items():
            if key.endswith("Json"):
                visit(json.loads(value))
        self.assertTrue(self.manifest["integration"]["requires_online_qp"])
        self.assertFalse(self.manifest["integration"]["previous_saits_mex_compatible"])
        self.assertFalse(self.manifest["provenance"]["benchmark_or_matlab_validation_performed"])

    def test_cpu_reference_predictions_and_masks_round_trip(self):
        model = AugmentedSAITS(self.deployed["model_config"]).eval()
        model.load_state_dict(self.deployed["model"], strict=True)
        with np.load(self.destination / "reference_inputs_outputs.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["history_normalized"].shape, (7, 50, 31))
            self.assertEqual(arrays["current_state_normalized"].shape, (7, 34))
            self.assertEqual(arrays["history_mask"].dtype, bool)
            self.assertEqual(arrays["current_state_mask"].dtype, bool)
            self.assertTrue(np.isfinite(arrays["history_normalized"]).all())
            self.assertTrue(np.all(arrays["history_normalized"][~arrays["history_mask"]] == 0))
            self.assertTrue(arrays["history_mask"][0].all())  # No-return != sensor failure.
            self.assertFalse(arrays["history_mask"][2, :, :14].any())
            self.assertTrue(arrays["history_mask"][2, :, 14:].all())  # No artificial motion loss.
            self.assertFalse(arrays["history_mask"][4, :-1].any())  # Cold leading padding.
            with torch.inference_mode():
                output = model(*(torch.from_numpy(arrays[key]) for key in (
                    "history_normalized", "history_mask", "current_state_normalized", "current_state_mask")))
            np.testing.assert_allclose(output["physical"].numpy(), arrays["prediction_physical"], rtol=0, atol=0)
            np.testing.assert_allclose(arrays["prediction_physical"][:, :7],
                                       arrays["prediction_normalized"][:, :7] * 7, rtol=0, atol=0)
            np.testing.assert_allclose(arrays["fused_physical"][0, :7], 7)
            np.testing.assert_allclose(arrays["fused_physical"][0, 7:], 0)
            np.testing.assert_allclose(arrays["fused_physical"][2], arrays["prediction_physical"][2], rtol=0, atol=0)

    def test_online_qp_fixtures_recompute_and_expose_fallback(self):
        barrier = BarrierSafetyFilter(self.deployed["barrier_config"]).eval()
        barrier.load_state_dict(self.deployed["barrier"], strict=True)
        with np.load(self.destination / "reference_inputs_outputs.npz", allow_pickle=False) as arrays:
            state = torch.from_numpy(arrays["qp_physical_state"])
            reference = {name: torch.from_numpy(arrays["qp_reference_" + name])
                         for name in ("p", "v", "a", "rho", "rhodot", "rhoddot")}
            with torch.inference_mode():
                result = barrier(state[:, :3], state[:, 3:6], state[:, 6:9],
                    torch.from_numpy(arrays["fused_physical"]), reference,
                    torch.from_numpy(arrays["qp_previous_acceleration"]),
                    uncertainty_bounds=torch.from_numpy(arrays["qp_uncertainty_bounds_m"]),
                    nominal_acceleration=torch.from_numpy(arrays["qp_nominal_acceleration"][:, :2]))
            self.assertEqual(arrays["qp_constraint_matrix"].dtype, np.float64)
            self.assertEqual(arrays["qp_constraint_matrix"].shape, (7, 30, 3))
            for key in ("acceleration", "k2", "feasible", "emergency_used", "constraint_matrix", "constraint_rhs"):
                np.testing.assert_allclose(result[key].numpy(), arrays["qp_" + key], atol=0, rtol=0)
            residual = np.einsum("bij,bj->bi", arrays["qp_constraint_matrix"], arrays["qp_selected_solution"]) - arrays["qp_constraint_rhs"]
            np.testing.assert_allclose(residual, arrays["qp_inequality_residual"], atol=1e-14, rtol=0)
            self.assertFalse(bool(arrays["qp_input_valid"][-1]))
            self.assertTrue(bool(arrays["qp_emergency_used"][-1]))
            self.assertFalse(bool(arrays["qp_feasible"][-1]))
            self.assertTrue(np.any(arrays["qp_feasible"][:-1]))
            np.testing.assert_allclose(arrays["qp_uncertainty_bounds_m"][0], 0)
            np.testing.assert_allclose(arrays["qp_uncertainty_bounds_m"][2], .05)
            np.testing.assert_allclose(arrays["qp_full_acceleration"][:, 2], -.1)
            commands = acceleration_to_commands(torch.from_numpy(arrays["qp_full_acceleration"]),
                state[:, 8], torch.from_numpy(arrays["qp_yaw_rate_command"]),
                parameters=self.deployed["physical_calibration"]["parameters"])
            np.testing.assert_allclose(commands.numpy(), arrays["allocated_commands"], atol=0, rtol=0)

    def test_matlab_portable_weight_records_and_artifact_hashes(self):
        expected_files = {"deployment.pt", "deployment.json", "model.mat", "preprocessing.json",
                          "reference_inputs_outputs.npz", "reference_inputs_outputs.mat"}
        self.assertEqual({path.name for path in self.destination.iterdir()}, expected_files)
        mat = loadmat(self.destination / "model.mat", simplify_cells=True)["deployment"]
        self.assertEqual(mat["schemaVersion"], exporter.SCHEMA)
        self.assertEqual(len(mat["modelWeights"]), len(self.deployed["model"]))
        records = {record["name"]: record for record in mat["modelWeights"]}
        for key in ("decoder.0.weight", "state.value_weight", "physical_scale"):
            np.testing.assert_allclose(records[key]["value"], self.deployed["model"][key].numpy(), atol=0, rtol=0)
        reference = loadmat(self.destination / "reference_inputs_outputs.mat", simplify_cells=True)["reference"]
        self.assertEqual(reference["prediction_physical"].shape, (7, 14))
        self.assertEqual(reference["qp_constraint_matrix"].shape, (7, 30, 3))
        for name, metadata in self.manifest["artifacts"].items():
            self.assertEqual(exporter.file_sha256(self.destination / name), metadata["sha256"])
            self.assertEqual((self.destination / name).stat().st_size, metadata["bytes"])


if __name__ == "__main__":
    unittest.main()
