"""Real-QP optimality/gradients and map-free conditional barrier checks."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "scipy"))
if AVAILABLE:
    import torch
    from scipy.optimize import minimize
    from ares_safe_training.qp import DifferentiableQP
    from ares_safe_training.barrier import BarrierConfig, BarrierSafetyFilter


@unittest.skipUnless(AVAILABLE, "PyTorch/SciPy required")
class TestDifferentiableQP(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)

    def test_batched_optimum_matches_independent_scipy_slsqp(self):
        rng = np.random.default_rng(42)
        batch, n, m = 7, 3, 11
        factor = rng.normal(size=(batch, n, n))
        Q = np.transpose(factor, (0, 2, 1)) @ factor + 0.7 * np.eye(n)
        p = 3 * rng.normal(size=(batch, n))
        G = rng.normal(size=(batch, m, n))
        h = rng.uniform(0.2, 1.5, size=(batch, m))  # x=0 is strictly feasible
        answer = DifferentiableQP(tolerance=1e-10, max_iter=90)(
            *[torch.tensor(value, dtype=torch.float64) for value in (Q, p, G, h)])
        self.assertTrue(answer["feasible"].all().item())
        self.assertTrue(answer["converged"].all().item())
        for index in range(batch):
            solved = minimize(lambda x: 0.5 * x @ Q[index] @ x + p[index] @ x,
                np.zeros(n), jac=lambda x: Q[index] @ x + p[index],
                constraints={"type": "ineq", "fun": lambda x: h[index] - G[index] @ x,
                             "jac": lambda x: -G[index]},
                method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
            self.assertTrue(solved.success, solved.message)
            np.testing.assert_allclose(answer["solution"][index].numpy(), solved.x, atol=1e-5, rtol=1e-5)
        self.assertLess(answer["max_violation"].max().item(), 1e-9)

    def test_implicit_kkt_gradients_match_finite_differences(self):
        solver = DifferentiableQP(tolerance=1e-11, max_iter=90)
        Q = torch.tensor([[2., 0.2], [0.2, 1.4]], dtype=torch.float64)
        p = torch.tensor([-2., -0.1], dtype=torch.float64, requires_grad=True)
        G = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]],
                         dtype=torch.float64, requires_grad=True)
        h = torch.tensor([0.3, 1.5, 1., 1.], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda pp, gg, hh: solver(Q, pp, gg, hh)["solution"], (p, G, h),
            eps=1e-5, atol=2e-4, rtol=2e-3))
        diagonal = torch.tensor([2., 1.4], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda dd: solver(torch.diag(dd), p.detach(), G.detach(), h.detach())["solution"],
            (diagonal,), eps=1e-5, atol=2e-4, rtol=2e-3))

    def test_infeasible_constraints_are_flagged_not_clipped_to_fake_solution(self):
        result = DifferentiableQP()(torch.eye(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), torch.tensor([[1.], [-1.]], dtype=torch.float64),
            torch.tensor([0., -1.], dtype=torch.float64))  # x<=0 and x>=1
        self.assertFalse(result["feasible"].item())
        self.assertFalse(result["converged"].item())
        self.assertTrue(torch.isfinite(result["solution"]).all().item())
        self.assertGreaterEqual(result["max_violation"].item(), 0.49)

    def test_invalid_hessian_and_nonfinite_coefficients_fail_explicitly(self):
        solver = DifferentiableQP()
        with self.assertRaisesRegex(ValueError, "positive definite"):
            solver(-torch.eye(1), torch.zeros(1), torch.ones(1, 1), torch.ones(1))
        with self.assertRaisesRegex(ValueError, "finite"):
            solver(torch.eye(1), torch.tensor([float("nan")]), torch.ones(1, 1), torch.ones(1))


@unittest.skipUnless(AVAILABLE, "PyTorch/SciPy required")
class TestLidarBarrier(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        torch.manual_seed(42)

    @staticmethod
    def inputs(batch=1, distance=7., speed=0.2, device="cpu"):
        p = torch.tensor([[0., 0., -7.]], device=device).repeat(batch, 1)
        v = torch.tensor([[speed, 0., 0.]], device=device).repeat(batch, 1)
        rpy = torch.zeros(batch, 3, device=device)
        lidar = torch.tensor([[distance] * 7 + [0.5] * 7], device=device).repeat(batch, 1)
        reference = {"p": p.clone(), "v": v.clone(), "a": torch.zeros_like(p),
                     "rho": 2., "rhodot": 0., "rhoddot": 0.}
        previous = torch.zeros(batch, 2, device=device)
        nominal = torch.tensor([[1., 0.]], device=device).repeat(batch, 1)
        return p, v, rpy, lidar, reference, previous, nominal

    @staticmethod
    def configuration(**changes):
        values = dict(disturbance_bound_mps2=0.1, range_error_bound_m=(0.1,) * 7,
                      qp_max_iter=90, qp_tolerance=1e-9, beam_sampling_margin_m=0.)
        values.update(changes)
        return BarrierConfig(**values)

    def test_hard_acceleration_jerk_visibility_and_calibration_are_enforced(self):
        safety = BarrierSafetyFilter(self.configuration(learned_k2=False))
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(batch=3)
        nominal[:] = torch.tensor([8., 8.])
        out = safety(p, v, rpy, lidar, reference, previous, 0.1, nominal_acceleration=nominal)
        self.assertTrue(out["feasible"].all().item())
        self.assertLessEqual(out["max_violation"].max().item(), 1e-8)
        self.assertTrue((out["acceleration"].abs() <= 0.100001).all().item())
        next_velocity = v[:, :2].double() + 0.1 * out["acceleration"]
        center = out["visibility_center_rad"]
        n = torch.stack((center.cos(), center.sin()), dim=1)
        t = torch.stack((-center.sin(), center.cos()), dim=1)
        longitudinal, transverse = (n * next_velocity).sum(1), (t * next_velocity).sum(1)
        self.assertTrue((longitudinal >= -1e-8).all().item())
        self.assertTrue((transverse.abs() <= out["visibility_half_angle_rad"].tan() * longitudinal + 1e-8).all().item())
        expected_lower = (7 - 0.1) * np.cos(np.deg2rad(15)) - 0.4 - (
            0.2 * 0.1 + 0.5 * (np.sqrt(2) + 0.1) * 0.1 ** 2)
        np.testing.assert_allclose(out["distance_lower_m"].detach().numpy(), expected_lower, atol=1e-8)

    def test_sector_outer_triangle_constraints_cover_unit_arc_and_all_farther_ranges(self):
        safety = BarrierSafetyFilter(self.configuration(learned_k2=False))
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(distance=1.945)
        out = safety(p, v, rpy, lidar, reference, previous, 0.1, nominal_accel=nominal)
        self.assertTrue(out["feasible"].item())
        acceleration = out["acceleration"][0].detach().numpy()
        radius, k1, k2 = 1.1, 2., 2.
        velocity = v[0, :2].numpy()
        for sector in range(7):
            vertices = out["sector_vertices"][0, sector].detach().numpy()
            angles = np.linspace(np.arctan2(*vertices[0, ::-1]), np.arctan2(*vertices[1, ::-1]), 61)
            direction = np.column_stack((np.cos(angles), np.sin(angles)))
            lower = out["distance_lower_m"][0, sector].item()
            stationary = np.sqrt(max(0., 2 * (velocity @ velocity) / (k1 * k2) - radius ** 2))
            for distance in (lower, max(lower, stationary), lower + 0.2, lower + 2., lower + 20.):
                zeta_over_2s = (-direction @ (acceleration + (k1 + k2) * velocity)
                    + 0.5 * k1 * k2 * distance
                    + ((velocity @ velocity) - 0.5 * k1 * k2 * radius ** 2) / distance - 0.1)
                self.assertGreaterEqual(zeta_over_2s.min(), -1e-7)

    def test_production_beam_margin_and_per_sample_measured_uncertainty(self):
        production = BarrierConfig(0.1, (0.8,) * 7, learned_k2=False)
        self.assertAlmostEqual(production.beam_sampling_margin_m, 0.27)
        pixel_diagonal_rad = np.hypot(np.deg2rad(30/14), np.deg2rad(358/715))
        self.assertLess(abs(2 * 7 * np.sin(pixel_diagonal_rad/2) - 0.27), 0.003)
        safety = BarrierSafetyFilter(production)
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(batch=2)
        bounds = torch.tensor([[0.] * 7, [0.8] * 7], dtype=torch.float64)
        out = safety(p, v, rpy, lidar, reference, previous, 0.1,
                     uncertainty_bounds=bounds, nominal_acceleration=nominal)
        self.assertTrue(out["feasible"].all().item())
        expected_difference = 0.8 * np.cos(np.deg2rad(15))
        np.testing.assert_allclose((out["distance_lower_m"][0] - out["distance_lower_m"][1]).numpy(),
                                   expected_difference, atol=1e-10)
        np.testing.assert_array_equal(out["beam_sampling_margin_m"].numpy(), [0.27, 0.27])
        with self.assertRaisesRegex(ValueError, "beam_sampling_margin_m"):
            BarrierSafetyFilter(BarrierConfig(0.1, (0.1,) * 7, beam_sampling_margin_m=-0.1))

    def test_hocbf_psi1_invalidity_brakes_and_returns_differentiable_violation(self):
        safety = BarrierSafetyFilter(self.configuration(learned_k2=False))
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(distance=1.98, speed=1.)
        lidar.requires_grad_(True)
        out = safety(p, v, rpy, lidar, reference, previous, 0.1, nominal_acceleration=nominal)
        self.assertTrue((out["barrier_values"] > 0).all().item())
        self.assertTrue((out["psi1"] < 0).any().item())
        self.assertFalse(out["cbf_valid"].item())
        self.assertFalse(out["feasible"].item())
        self.assertTrue(out["emergency_used"].item())
        self.assertAlmostEqual(out["acceleration"][0, 0].item(), -0.1, places=7)
        self.assertGreater(out["true_inequality_violation"].sum().item(), 0)
        out["true_inequality_violation"].square().sum().backward()
        self.assertTrue(torch.isfinite(lidar.grad).all().item())
        self.assertGreater(lidar.grad.abs().sum().item(), 0)

    def test_active_qp_backpropagates_into_lidar_and_positive_gain_network(self):
        safety = BarrierSafetyFilter(self.configuration())
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(distance=1.945)
        lidar.requires_grad_(True)
        out = safety(p, v, rpy, lidar, reference, previous, 0.1, nominal_acceleration=nominal)
        self.assertTrue(out["feasible"].item())
        self.assertTrue(((out["k2"] > 0.5) & (out["k2"] < 4)).all().item())
        out["acceleration"][:, 0].sum().backward()
        self.assertTrue(torch.isfinite(lidar.grad).all().item())
        self.assertGreater(lidar.grad.abs().sum().item(), 1e-6)
        gradients = [parameter.grad for parameter in safety.gain_head.parameters()]
        self.assertTrue(all(value is not None and torch.isfinite(value).all().item() for value in gradients))
        self.assertGreater(sum(value.abs().sum().item() for value in gradients), 1e-6)

    def test_nonfinite_inputs_and_unusable_range_bounds_do_not_claim_feasibility(self):
        safety = BarrierSafetyFilter(self.configuration())
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(batch=2)
        lidar[0, 0] = float("nan")
        lidar[1, :7] = 0.5
        out = safety(p, v, rpy, lidar, reference, previous, 0.1, nominal_acceleration=nominal)
        self.assertFalse(out["feasible"].any().item())
        self.assertTrue(torch.isfinite(out["acceleration"]).all().item())
        self.assertTrue(torch.isfinite(out["true_inequality_violation"]).all().item())
        self.assertTrue((out["acceleration"].abs() <= 1).all().item())
        self.assertFalse(out["input_valid"][0].item())
        self.assertTrue((out["distance_lower_m"][1] < 1.1).all().item())

    def test_tracking_tube_is_soft_without_relaxing_actuator_limits(self):
        safety = BarrierSafetyFilter(self.configuration(learned_k2=False))
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(speed=0.)
        reference["p"] = p + torch.tensor([[5., 0., 0.]])
        reference["rho"] = 0.5
        out = safety(p, v, rpy, lidar, reference, previous, 0.1, nominal_acceleration=nominal)
        self.assertTrue(out["feasible"].item())
        self.assertGreater(out["tracking_slack"].item(), 1)
        self.assertTrue((out["acceleration"].abs() <= 0.100001).all().item())
        self.assertLessEqual(out["max_violation"].item(), 1e-8)

    @unittest.skipUnless(AVAILABLE and torch.cuda.is_available(), "CUDA not available; CPU cannot establish GPU performance")
    def test_cuda_batch_forward_backward(self):
        safety = BarrierSafetyFilter(self.configuration()).cuda()
        p, v, rpy, lidar, reference, previous, nominal = self.inputs(batch=16, distance=1.945, device="cuda")
        lidar.requires_grad_(True)
        out = safety(p, v, rpy, lidar, reference, previous, 0.1, nominal_acceleration=nominal)
        self.assertEqual(out["acceleration"].device.type, "cuda")
        self.assertTrue(out["feasible"].all().item())
        out["acceleration"].sum().backward()
        self.assertTrue(torch.isfinite(lidar.grad).all().item())


if __name__ == "__main__":
    unittest.main()
