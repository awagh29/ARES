"""Persistent CPU inference/control for the selected augmented deployment.

No scene, map, future state, or unavailable cloud enters this runtime. MATLAB
owns the plant; this module returns its four executed guidance commands.
"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ares_safe_training.models import AugmentedSAITS
from ares_safe_training.barrier import BarrierSafetyFilter
from ares_safe_training.config import PipelineConfig
from ares_safe_training.control import smooth_reference, nominal_acceleration, desired_yaw_rate
from ares_safe_training.data import FeatureNormalizer
from ares_safe_training.dynamics import acceleration_to_commands
from ares_safe_training.training import TensorNormalizer, _current_features

RUNTIME_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Controller:
    def __init__(self, folder, goal, threads=1, continue_after_failure=False):
        self.folder = Path(str(folder)).resolve(strict=True)
        self.manifest = json.loads((self.folder / 'deployment.json').read_text(encoding='utf-8'))
        if self.manifest['schema_version'] != 'ares-augmented-saits-barrier-deployment-v1':
            raise ValueError('Expected selected augmented SAITS + BarrierNet deployment.')
        for name, info in self.manifest['artifacts'].items():
            if Path(name).name != name or _hash(self.folder / name) != info['sha256']:
                raise ValueError(f'Deployment integrity failure: {name}')
        # Enforce the source functions actually used by this runtime. Added
        # bridge code is separately hashed in the MATLAB preparation report.
        root = Path(__file__).parent
        used = {'models.py', 'barrier.py', 'qp.py', 'control.py', 'dynamics.py', 'data.py', 'config.py', 'saits_endpoint_model.py'}
        for name, expected in self.manifest['provenance']['export_source_sha256'].items():
            if Path(name).name in used:
                actual = hashlib.sha256((root / name).read_text(encoding='utf-8').encode()).hexdigest()
                if actual != expected:
                    raise ValueError(f'Training/export runtime source changed: {name}')
        torch.set_num_threads(int(threads))
        payload = torch.load(self.folder / 'deployment.pt', map_location='cpu', weights_only=True)
        self.model = AugmentedSAITS(payload['model_config']).eval()
        self.model.load_state_dict(payload['model'], strict=True)
        self.barrier = BarrierSafetyFilter(payload['barrier_config']).eval()
        self.barrier.load_state_dict(payload['barrier'], strict=True)
        self.normalizer = FeatureNormalizer.from_dict(payload['normalizer'])
        self.transform = TensorNormalizer(self.normalizer, torch.device('cpu'))
        allowed = {field.name for field in fields(PipelineConfig)}
        self.config = PipelineConfig(**{k: v for k, v in payload['controller_defaults'].items() if k in allowed})
        self.parameters = payload['physical_calibration']['parameters']
        self.goal = torch.as_tensor(np.asarray(goal, dtype=np.float32).reshape(1, 3).copy())
        self.continue_after_failure = bool(continue_after_failure)
        if not torch.isfinite(self.goal).all():
            raise ValueError('Mission goal must be finite.')
        self.reset()

    def reset(self):
        self.history = torch.zeros(1, self.model.config.window_size, 31)
        self.mask = torch.zeros_like(self.history, dtype=torch.bool)
        self.initial = None
        self.last_position = None
        self.last_time = torch.zeros(1)
        self.previous = torch.zeros(1, 2)
        self.previous_vertical = torch.zeros(1)
        self.last_tick = None
        self.cached = None
        self.failed = False

    @torch.inference_mode()
    def step(self, position, velocity, rpy, omega, measured_sectors, available, time_s):
        tick = float(time_s)
        if not np.isfinite(tick) or tick < 0:
            raise ValueError('Time must be finite and nonnegative.')
        if self.last_tick == tick:
            return self.cached.copy()
        if self.last_tick is not None and abs(tick - self.last_tick - .1) > 1e-5:
            raise ValueError('Controller expects exactly one chronological 10-Hz scan per step.')
        started = time.perf_counter()
        parts = [np.asarray(v, dtype=np.float32).reshape(3) for v in (position, velocity, rpy, omega)]
        state = torch.from_numpy(np.concatenate(parts + [np.zeros(1, np.float32)])[None].copy())
        if not torch.isfinite(state).all():
            raise ValueError('Motion state is nonfinite.')
        if self.initial is None:
            self.initial = state[:, :3].clone()
            self.last_position = self.initial.clone()
        enabled = bool(available)
        measured = torch.full((1, 14), enabled, dtype=torch.bool)
        # Crucially do not read even one value of unavailable sector inputs.
        clean = torch.zeros(1, 14)
        if enabled:
            clean = torch.as_tensor(np.asarray(measured_sectors, dtype=np.float32).reshape(1, 14).copy())
            if not torch.isfinite(clean).all():
                raise ValueError('Available sectors must be finite.')
        raw, mask, current, current_mask, self.last_position, self.last_time = _current_features(
            state, clean, measured, self.last_position, self.last_time, tick)
        self.history = torch.cat((self.history[:, 1:], raw[:, None]), 1)
        self.mask = torch.cat((self.mask[:, 1:], mask[:, None]), 1)
        prediction = self.model(self.transform.history(self.history, self.mask), self.mask,
                                self.transform.state(current, current_mask), current_mask)['physical'].float()
        perceived = torch.where(measured, clean, prediction)
        reference = smooth_reference(self.initial, self.goal, tick, self.config.reference_speed_mps,
                                     self.config.tracking_tube_radius_m)
        nominal = nominal_acceleration(state, perceived, reference, self.config)
        bounds = torch.zeros(1, 7, dtype=torch.float64) if enabled else self.barrier.calibration_bounds[None]
        filtered = self.barrier(state[:, :3], state[:, 3:6], state[:, 6:9], perceived,
            reference, self.previous, .1, bounds, nominal_acceleration=nominal[:, :2])
        horizontal = filtered['acceleration'].float()
        tracking_slack = filtered['tracking_slack']
        maximum_violation = filtered['max_violation']
        if self.continue_after_failure and not bool(filtered['feasible'][0]):
            # Explicit diagnostic policy: keep trying to reach the goal using
            # perceived LiDAR. This is NOT a feasible/certified QP command.
            limit = self.barrier.config.acceleration_limit_mps2
            change = self.barrier.config.jerk_limit_mps3 * .1
            lower = torch.maximum(torch.full_like(self.previous, -limit), self.previous - change)
            upper = torch.minimum(torch.full_like(self.previous, limit), self.previous + change)
            horizontal = torch.maximum(lower, torch.minimum(upper, nominal[:, :2]))
            matrix, rhs = filtered['constraint_matrix'], filtered['constraint_rhs']
            # Tracking has slack; obstacles never acquire slack. Recompute
            # residuals for the ACTUALLY EXECUTED fallback, not the QP brake.
            tracking_slack = ((matrix[:, 28, :2] * horizontal.double()).sum(1) - rhs[:, 28]).clamp_min(0)
            selected = torch.cat((horizontal.double(), tracking_slack[:, None]), 1)
            residual = torch.matmul(matrix, selected.unsqueeze(-1)).squeeze(-1) - rhs
            maximum_violation = residual.clamp_min(0).amax(1)
        vertical = torch.maximum(self.previous_vertical - .1,
                                 torch.minimum(self.previous_vertical + .1, nominal[:, 2]))
        acceleration = torch.cat((horizontal, vertical[:, None]), 1)
        yaw = desired_yaw_rate(state, self.goal, self.config.maximum_yaw_rate_radps)
        command = acceleration_to_commands(acceleration, state[:, 8], yaw, parameters=self.parameters)
        if not torch.isfinite(command).all():
            raise ValueError('Allocated guidance command is nonfinite.')
        if abs(float(command[0, 0])) > self.parameters['roll_limit_rad'] + 1e-6 or abs(float(command[0, 1])) > self.parameters['pitch_limit_rad'] + 1e-6:
            raise ValueError('Allocated attitude is outside the calibrated envelope.')
        self.failed |= not bool(filtered['feasible'][0])
        self.previous, self.previous_vertical = horizontal.clone(), vertical.clone()
        self.last_tick = tick
        # Fixed layout documented by Bridge output_layout(); all numeric for
        # MATLAB's ndarray conversion in a single interprocess round trip.
        result = np.concatenate([
            command[0].numpy(), prediction[0].numpy(), perceived[0].numpy(),
            raw[0].numpy(), mask[0].numpy(), reference['p'][0].numpy(),
            acceleration[0].numpy(), nominal[0].numpy(),
            np.array([enabled, not enabled, bool(filtered['feasible'][0]),
                      float(maximum_violation[0]), float(tracking_slack[0]),
                      float(filtered['k2'][0]), self.failed, time.perf_counter()-started,
                      float(filtered['barrier_values'].min()), float(filtered['psi1'].min())])
        ]).astype(np.float64)
        self.cached = result.copy()
        return result

    @torch.inference_mode()
    def reference_check(self):
        """Compare actual installed inference/QP against exported CPU fixtures."""
        from export_augmented_saits_barrier import make_reference_cases
        payload = torch.load(self.folder / 'deployment.pt', map_location='cpu', weights_only=True)
        arrays, _ = make_reference_cases(self.model, self.barrier, self.normalizer, payload)
        checks = {}
        with np.load(self.folder / 'reference_inputs_outputs.npz', allow_pickle=False) as expected:
            for key, tolerance in [('prediction_physical', 2e-5), ('qp_constraint_matrix', 2e-6),
                                   ('qp_constraint_rhs', 2e-5), ('qp_selected_solution', 2e-5),
                                   ('allocated_commands', 2e-5)]:
                error = float(np.max(np.abs(arrays[key] - expected[key])))
                # In invalid-start fixtures s is clamped at 1e-4 m, so a
                # one-ULP FP32 gain/output difference is amplified in large
                # FP64 rhs coefficients. Check their scale as well as the
                # absolute budget; command/feasibility parity stays strict.
                relative = 2e-7 if key == 'qp_constraint_rhs' else 0.0
                passed = bool(np.allclose(arrays[key], expected[key], atol=tolerance, rtol=relative))
                checks[key] = {'maximum_absolute_error': error, 'absolute_tolerance': tolerance,
                               'relative_tolerance': relative, 'passed': passed}
            checks['qp_feasible'] = {'passed': bool(np.array_equal(arrays['qp_feasible'], expected['qp_feasible']))}
        return json.dumps({'passed': all(v['passed'] for v in checks.values()), 'checks': checks,
                           'device': 'cpu', 'neural_dtype': 'float32', 'qp_dtype': 'float64'})


def output_layout():
    return {'command': (0, 4), 'prediction': (4, 18), 'perceived': (18, 32),
            'features': (32, 63), 'mask': (63, 94), 'reference': (94, 97),
            'acceleration': (97, 100), 'nominal': (100, 103), 'diagnostics': (103, 113)}
