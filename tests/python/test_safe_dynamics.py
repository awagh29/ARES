"""Physical invariants, differentiability, stability, and optional MATLAB parity."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from ares_safe_training.dynamics import (
        CALIBRATION_SCHEMA, GuidanceDynamics, acceleration_to_commands, initial_state, load_calibration,
    )

# Explicit test constants copied from the inspected plant, not a production
# default selected by the module. Production reads the measured calibration.
PARAMETERS = {
    "mass_kg": 0.1, "gravity_mps2": 9.81,
    "roll_kp": 3402.97, "roll_kd": 116.67,
    "pitch_kp": 3402.97, "pitch_kd": 116.67,
    "yaw_rate_gain": 1950.0, "thrust_gain": 3900.0,
    "roll_limit_rad": math.radians(25), "pitch_limit_rad": math.radians(25),
    "yaw_rate_limit_radps": 0.5,
}


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for differentiable guidance dynamics.")
class TestGuidanceDynamics(unittest.TestCase):
    def setUp(self):
        self.dynamics = GuidanceDynamics(PARAMETERS)
        self.state = initial_state(torch.tensor([[0.0, 0.0, -7.0]], dtype=torch.float64))
        self.commands = torch.tensor([[0.0, 0.0, 0.0, 0.981]], dtype=torch.float64)

    def test_requires_explicit_calibration_and_stable_substep(self):
        with self.assertRaises(ValueError):
            GuidanceDynamics()
        with self.assertRaises(ValueError):
            GuidanceDynamics(PARAMETERS, integration_dt=0.01)
        incomplete = dict(PARAMETERS)
        del incomplete["thrust_gain"]
        with self.assertRaises(ValueError):
            GuidanceDynamics(incomplete)

    def test_actual_thrust_is_a_separate_state_not_the_command(self):
        rate = self.dynamics.derivative(self.state, self.commands)
        self.assertEqual(float(rate[0, 5]), 9.81)
        self.assertAlmostEqual(float(rate[0, 12]), 3900 * 0.981)
        hover = self.state.clone()
        hover[:, 12] = 0.981
        torch.testing.assert_close(self.dynamics.derivative(hover, self.commands),
                                   torch.zeros_like(hover), rtol=0, atol=1e-12)

    def test_ned_tilt_acceleration_has_the_physical_signs(self):
        roll = self.state.clone()
        roll[:, 6] = 0.2
        roll[:, 12] = 0.981
        rate = self.dynamics.derivative(roll, self.commands)
        self.assertGreater(float(rate[0, 4]), 0.0)  # positive roll accelerates East
        pitch = self.state.clone()
        pitch[:, 7] = 0.2
        pitch[:, 12] = 0.981
        rate = self.dynamics.derivative(pitch, self.commands)
        self.assertLess(float(rate[0, 3]), 0.0)  # positive pitch accelerates South

    def test_yaw_command_is_euler_yaw_rate_not_body_r(self):
        state = self.state.clone()
        state[:, 6:9] = torch.tensor([0.3, -0.2, 0.4])
        state[:, 9:12] = torch.tensor([0.1, 0.2, 0.3])
        expected = (math.sin(0.3) * 0.2 + math.cos(0.3) * 0.3) / math.cos(-0.2)
        rate = self.dynamics.derivative(state, self.commands)
        self.assertAlmostEqual(float(rate[0, 8]), expected, places=7)
        self.assertGreater(abs(expected - 0.3), 0.01)

    def test_control_jacobian_matches_the_actual_control_affine_structure(self):
        state = self.state.clone()
        state[:, 6:9] = torch.tensor([0.2, -0.15, 0.3], dtype=torch.float64)
        command = self.commands[0].clone().requires_grad_()
        jacobian = torch.autograd.functional.jacobian(
            lambda value: self.dynamics.derivative(state, value.unsqueeze(0))[0], command)
        expected = torch.zeros(13, 4, dtype=torch.float64)
        phi, theta = 0.2, -0.15
        inverse = torch.tensor([[1, 0, -math.sin(theta)],
            [0, math.cos(phi), math.sin(phi) * math.cos(theta)],
            [0, -math.sin(phi), math.cos(phi) * math.cos(theta)]], dtype=torch.float64)
        expected[9:12, :3] = inverse @ torch.diag(torch.tensor([3402.97, 3402.97, 1950.0], dtype=torch.float64))
        expected[12, 3] = 3900.0
        torch.testing.assert_close(jacobian, expected, rtol=1e-12, atol=1e-10)

    def test_one_ms_integration_is_stable_with_analytic_fast_thrust(self):
        result = self.dynamics(self.state, self.commands, 0.1)
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertAlmostEqual(float(result[0, 12]), 0.981, places=12)
        exact_velocity = 9.81 * (1 - math.exp(-3900 * 0.1)) / 3900
        exact_position = -7 + (9.81 / 3900) * (0.1 - (1 - math.exp(-3900 * 0.1)) / 3900)
        self.assertLess(abs(float(result[0, 5]) - exact_velocity), 2e-4)
        self.assertLess(abs(float(result[0, 2]) - exact_position), 2e-5)
        torch.testing.assert_close(self.state[:, 12], torch.zeros(1, dtype=torch.float64))

    def test_trace_retains_interior_positions_for_tunneling_checks(self):
        state = self.state.clone()
        state[:, 0] = -0.1
        state[:, 3] = 2.0
        state[:, 12] = 0.981
        result, trace = self.dynamics(state, self.commands, 0.1, return_trace=True)
        self.assertEqual(tuple(trace.shape), (1, 101, 3))
        torch.testing.assert_close(trace[:, 0], state[:, :3])
        torch.testing.assert_close(trace[:, -1], result[:, :3])
        self.assertTrue((trace[0, :, 0].abs() < 0.005).any().item())
        self.assertGreater(abs(float(trace[0, 0, 0])), 0.05)
        self.assertGreater(abs(float(trace[0, -1, 0])), 0.05)

    def test_allocation_has_no_silent_post_qp_clipping(self):
        acceleration = torch.tensor([[1.0, -1.0, 4.0]], dtype=torch.float64, requires_grad=True)
        action = self.dynamics.acceleration_to_commands(acceleration,
            torch.tensor([0.3], dtype=torch.float64), torch.tensor([0.25], dtype=torch.float64))
        self.assertLess(float(action[0, 0].detach().abs()), PARAMETERS["roll_limit_rad"])
        self.assertLess(float(action[0, 1].detach().abs()), PARAMETERS["pitch_limit_rad"])
        self.assertEqual(float(action[0, 2].detach()), 0.25)
        action.sum().backward()
        self.assertTrue(torch.isfinite(acceleration.grad).all().item())
        outside = acceleration_to_commands(torch.tensor([[20.0, 0.0, 0.0]], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), parameters=PARAMETERS)
        self.assertGreater(float(outside[0, 1].abs()), PARAMETERS["pitch_limit_rad"])

    def test_allocation_rejects_implicit_integer_or_mixed_precision(self):
        # MATLAB-v7 can store an integer-valued double fixture as int16 on disk.
        # 9.81 - an integer Torch tensor otherwise silently uses default FP32,
        # even when later trigonometric terms use FP64.
        acceleration = torch.tensor([[0, 0, 0]], dtype=torch.int16)
        yaw = torch.zeros(1, dtype=torch.float64)
        with self.assertRaises(ValueError):
            self.dynamics.acceleration_to_commands(acceleration, yaw, yaw)
        with self.assertRaises(ValueError):
            self.dynamics.acceleration_to_commands(acceleration.double(), yaw.float(), yaw)
        command = self.dynamics.acceleration_to_commands(acceleration.double(), yaw, yaw)
        self.assertAlmostEqual(float(command[0, 3]), 0.981, places=15)

    def test_step_preserves_state_and_control_gradients(self):
        state = self.state.clone().requires_grad_()
        command = self.commands.clone().requires_grad_()
        result, trace = self.dynamics(state, command, 0.002, return_trace=True)
        (result.square().sum() + trace.square().sum()).backward()
        self.assertTrue(torch.isfinite(state.grad).all().item())
        self.assertTrue(torch.isfinite(command.grad).all().item())
        self.assertGreater(float(command.grad.abs().sum()), 0.0)

    def test_load_calibration_rejects_unfinished_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            value = {"schemaVersion": CALIBRATION_SCHEMA, "parameters": PARAMETERS,
                     "calibrationComplete": False}
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration(path)
            value["calibrationComplete"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            loaded = load_calibration(path)
            self.assertEqual(loaded["parameters"], PARAMETERS)
            self.assertEqual(len(loaded["calibration_sha256"]), 64)

    def test_matlab_derivative_and_trajectory_oracles_when_available(self):
        selected = os.environ.get("ARES_DYNAMICS_CALIBRATION")
        if selected is None:
            candidates = list((ROOT / "data/calibration/saits_barrier_dynamics").glob("*/calibration.json"))
            if not candidates:
                self.skipTest("Run the MATLAB calibration to create authoritative numerical fixtures.")
            selected = max(candidates, key=lambda item: item.stat().st_mtime)
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is required to read MATLAB-v7 numerical fixtures.")
        from scipy.io import loadmat
        calibration = load_calibration(selected)
        fixture = loadmat(Path(selected).parent / calibration["fixtures"]["file"], squeeze_me=True)
        dynamics = GuidanceDynamics(calibration["parameters"])
        states = torch.tensor(fixture["states"], dtype=torch.float64)
        commands = torch.tensor(fixture["commands"], dtype=torch.float64)
        np.testing.assert_allclose(dynamics.derivative(states, commands).numpy(),
                                   fixture["derivatives"], rtol=2e-12, atol=3e-9)
        states32 = torch.tensor(fixture["states_single"], dtype=torch.float32)
        commands32 = torch.tensor(fixture["commands_single"], dtype=torch.float32)
        np.testing.assert_allclose(dynamics.derivative(states32, commands32).numpy(),
                                   fixture["derivatives_single"], rtol=3e-6, atol=2e-3)
        rollout = dynamics(torch.tensor(fixture["rollout_initial_states"], dtype=torch.float64),
            torch.tensor(fixture["rollout_commands"], dtype=torch.float64),
            float(fixture["rollout_duration_s"])).numpy()
        np.testing.assert_allclose(rollout, fixture["rollout_next_states"], rtol=1e-5, atol=2e-3)
        # Explicitly restore MATLAB's intended double arithmetic, independently
        # of the MAT-v7 storage dtype selected for integer-valued fixture data.
        allocated = dynamics.acceleration_to_commands(
            torch.tensor(fixture["acceleration_inputs"], dtype=torch.float64),
            torch.tensor(fixture["allocation_yaw"], dtype=torch.float64),
            torch.tensor(fixture["allocation_yaw_rate"], dtype=torch.float64)).numpy()
        np.testing.assert_allclose(allocated, fixture["allocation_commands"], rtol=1e-12, atol=1e-12)

    def test_validation_report_rejects_failed_or_stale_provenance(self):
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is required for the calibration validator.")
        import hashlib
        import validate_dynamics_calibration as validator
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            fixture = directory / "fixtures.mat"
            fixture.write_bytes(b"fixture integrity is checked before reading any arrays")
            calibration_path = directory / "calibration.json"
            calibration_path.write_text(json.dumps({"schemaVersion": CALIBRATION_SCHEMA,
                "calibrationComplete": True, "parameters": PARAMETERS,
                "fixtures": {"file": fixture.name,
                             "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}}), encoding="utf-8")
            report_path = directory / "validation.json"
            with self.assertRaisesRegex(ValueError, "missing or invalid"):
                validator.verify_validation_report(calibration_path)
            report = {"schemaVersion": validator.VALIDATION_SCHEMA, "passed": True,
                "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                "fixtures_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "dynamics_source_sha256": validator._source_sha256(validator.SOURCE_PATH),
                "validator_source_sha256": validator._source_sha256(Path(validator.__file__))}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(validator.verify_validation_report(calibration_path)["passed"])
            report["passed"] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                validator.verify_validation_report(calibration_path)
            report["passed"] = True
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(validator, "_source_sha256", return_value="changed-source"):
                with self.assertRaisesRegex(ValueError, "stale"):
                    validator.verify_validation_report(calibration_path)
            fixture.write_bytes(b"tampered fixture")
            with self.assertRaisesRegex(ValueError, "stale"):
                validator.verify_validation_report(calibration_path)

    def test_validation_source_hash_is_windows_colab_portable(self):
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is required for the calibration validator.")
        import validate_dynamics_calibration as validator
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dynamics.py"
            path.write_bytes(b"x = 1\r\ny = 2\r\n")
            windows_hash = validator._source_sha256(path)
            path.write_bytes(b"x = 1\ny = 2\n")
            self.assertEqual(windows_hash, validator._source_sha256(path))


if __name__ == "__main__":
    unittest.main()
