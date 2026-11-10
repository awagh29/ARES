"""Conditional, map-free forward-sector BarrierNet control filter.

This uses an actual differentiable hard-inequality QP. Its geometry is
conditional on empirical sector lower bounds and static surfaces; discrete
LiDAR rays and finite calibration do not constitute a global safety certificate.
The explicit beam-gap margin is an engineering coverage assumption: subtracting
a pixel-sized range allowance cannot prove that a thin surface between rays was
observed, nor bound arbitrary occlusion or an unknown rear obstacle.
Side/rear surfaces are not invented. A hard next-velocity visibility cone keeps
the requested direction within the common projected forward field of view.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .qp import DifferentiableQP


@dataclass
class BarrierConfig:
    # REQUIRED calibrated constants, recorded/exported with the trained model.
    disturbance_bound_mps2: float
    range_error_bound_m: tuple
    vehicle_radius_m: float = 0.1
    clearance_m: float = 1.0
    acceleration_limit_mps2: float = 1.0
    jerk_limit_mps3: float = 1.0
    dt: float = 0.1
    k1: float = 2.0
    k2_min: float = 0.5
    k2_max: float = 4.0
    k2_fixed: float = 2.0
    learned_k2: bool = True
    tracking_slack_penalty: float = 100.0
    mount_lever_m: float = 0.4
    # Full diagonal angular pixel chord at 7 m is approximately 0.269 m for
    # the 15x716 grid (30/14 degree elevation, 358/715 degree azimuth steps).
    # This is recorded engineering conservatism, not a sampling certificate.
    beam_sampling_margin_m: float = 0.27
    position_error_bound_m: float = 0.0
    velocity_error_bound_mps: float = 0.0
    exact_simulator_state: bool = True
    orientation_margin_deg: float = 0.0
    sensor_elevation_half_deg: float = 15.0
    forward_half_angle_deg: float = 60.0
    qp_tolerance: float = 1e-8
    qp_max_iter: int = 70

    def validate(self):
        for name in ("vehicle_radius_m", "clearance_m", "acceleration_limit_mps2",
                     "jerk_limit_mps3", "dt", "k1", "k2_min", "k2_max",
                     "tracking_slack_penalty", "qp_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("disturbance_bound_mps2", "mount_lever_m", "beam_sampling_margin_m", "orientation_margin_deg"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.k2_min < self.k2_max or not self.k2_min <= self.k2_fixed <= self.k2_max:
            raise ValueError("Invalid bounded positive k2 configuration")
        if (not self.exact_simulator_state or self.position_error_bound_m != 0
                or self.velocity_error_bound_mps != 0):
            raise ValueError("This implementation explicitly assumes exact simulator position/velocity")
        bounds = tuple(float(value) for value in self.range_error_bound_m)
        if len(bounds) != 7 or any(not math.isfinite(value) or value < 0 for value in bounds):
            raise ValueError("range_error_bound_m must contain seven nonnegative calibrated bounds")
        self.range_error_bound_m = bounds
        if not 0 < self.forward_half_angle_deg < 89 or not 0 <= self.sensor_elevation_half_deg < 89:
            raise ValueError("Invalid sensor angular support")
        if self.qp_max_iter < 1:
            raise ValueError("qp_max_iter must be positive")


def _mv(matrix, vector):
    return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)


def _rotation(rpy):
    roll, pitch, yaw = rpy.unbind(dim=1)
    sr, cr = torch.sin(roll), torch.cos(roll)
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    return torch.stack((
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ), dim=1).reshape(-1, 3, 3)


def _rays(bearing, elevation_half):
    elevations = torch.tensor([-elevation_half, elevation_half], dtype=torch.float64)
    az, el = torch.meshgrid(bearing, elevations, indexing="ij")
    return torch.stack((torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az),
                        torch.sin(el)), dim=-1).reshape(-1, 3)


class BarrierSafetyFilter(nn.Module):
    """Horizontal z=[a_N,a_E,tracking_slack] BarrierNet QP.

    Inputs/outputs use physical NED units. ``nominal_acceleration`` is the
    frozen controller's requested XY acceleration. ``uncertainty_bounds`` can
    override fixed calibration bounds per sample, but cannot be learned. The
    caller passes zero for a currently measured sector and the calibrated
    reconstruction bound only for its unobserved distance/fraction pair. A
    normal no-return measurement is still measured, not an outage. The
    gain head learns only bounded positive k2; k1 and all safety margins remain
    fixed. Failed validity or QP checks select a bounded emergency brake and
    return feasible=False, never relabeling that brake as a feasible CBF control.
    """

    def __init__(self, config=None, **config_kwargs):
        super().__init__()
        if config is None:
            config = BarrierConfig(**config_kwargs)
        elif isinstance(config, dict):
            config = BarrierConfig(**config)
        elif config_kwargs:
            raise ValueError("Pass either config or configuration keywords")
        config.validate()
        self.config = config
        self.qp = DifferentiableQP(max_iter=config.qp_max_iter, tolerance=config.qp_tolerance)
        self.register_buffer("calibration_bounds", torch.tensor(config.range_error_bound_m, dtype=torch.float64))
        edges = torch.linspace(-math.radians(config.forward_half_angle_deg),
                               math.radians(config.forward_half_angle_deg), 8, dtype=torch.float64)
        body_azimuth = -torch.linspace(-179, 179, 716, dtype=torch.float64) * math.pi / 180
        elevation = math.radians(config.sensor_elevation_half_deg)
        for sector in range(7):
            columns = body_azimuth[(body_azimuth >= edges[sector]) & (body_azimuth <= edges[sector + 1])]
            # Include exact sector edges as well as actual organized ray columns.
            angles = torch.cat((edges[sector:sector + 1], columns, edges[sector + 1:sector + 2]))
            self.register_buffer(f"sector_rays_{sector}", _rays(angles, elevation))
        self.register_buffer("fov_edge_rays", _rays(edges[[0, -1]], elevation))
        self.gain_head = nn.Sequential(nn.Linear(22, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.normal_(self.gain_head[-1].weight, std=0.01)
        ratio = (config.k2_fixed - config.k2_min) / (config.k2_max - config.k2_min)
        ratio = min(max(ratio, 1e-6), 1 - 1e-6)
        nn.init.constant_(self.gain_head[-1].bias, math.log(ratio / (1 - ratio)))
        if not config.learned_k2:
            for parameter in self.gain_head.parameters():
                parameter.requires_grad_(False)

    def _angles(self, rpy):
        rotation = _rotation(rpy)
        yaw = rpy[:, 2]
        margin = math.radians(self.config.orientation_margin_deg)
        centers, halves = [], []
        for sector in range(7):
            rays = getattr(self, f"sector_rays_{sector}")
            world = torch.einsum("bij,rj->bri", rotation, rays)
            relative = torch.atan2(world[..., 1], world[..., 0]) - yaw[:, None]
            relative = torch.atan2(torch.sin(relative), torch.cos(relative))
            low, high = relative.amin(1), relative.amax(1)
            centers.append(yaw + 0.5 * (low + high))
            halves.append(0.5 * (high - low) + margin)
        center, half = torch.stack(centers, 1), torch.stack(halves, 1)
        world = torch.einsum("bij,rj->bri", rotation, self.fov_edge_rays)
        relative = torch.atan2(world[..., 1], world[..., 0]) - yaw[:, None]
        relative = torch.atan2(torch.sin(relative), torch.cos(relative))
        left = relative[:, :2].amax(1) + margin
        right = relative[:, 2:].amin(1) - margin
        cone_center = yaw + 0.5 * (left + right)
        cone_half = 0.5 * (right - left)
        # atan2/norm has a finite zero-tilt derivative; acos(R33) does not.
        tilt = torch.atan2(torch.linalg.vector_norm(rotation[:, :2, 2], dim=1), rotation[:, 2, 2])
        total_tilt = tilt + math.radians(self.config.sensor_elevation_half_deg) + margin
        angular_valid = (total_tilt < math.radians(89)) & (cone_half > 0)
        angular_valid &= (cone_half < math.pi / 2) & (half < math.pi / 2).all(1)
        horizontal_cos = torch.cos(total_tilt).clamp_min(1e-4)
        half_safe = half.clamp(0, math.radians(89))
        left_n = torch.stack((torch.cos(center - half_safe), torch.sin(center - half_safe)), -1)
        right_n = torch.stack((torch.cos(center + half_safe), torch.sin(center + half_safe)), -1)
        middle = torch.stack((torch.cos(center), torch.sin(center)), -1) / torch.cos(half_safe)[..., None]
        vertices = torch.stack((left_n, right_n, middle), dim=2)
        return vertices, horizontal_cos, cone_center, cone_half, angular_valid

    @staticmethod
    def _batch_scalar(value, batch, device, name):
        result = torch.as_tensor(value, dtype=torch.float64, device=device)
        if result.numel() == 1:
            return result.reshape(1).expand(batch)
        if result.numel() != batch:
            raise ValueError(f"{name} must be scalar or have one value per batch example")
        return result.reshape(batch)

    def forward(self, position, velocity, rpy, predicted_lidar, reference,
                previous_acceleration, dt=None, uncertainty_bounds=None, *,
                nominal_acceleration=None, nominal_accel=None):
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position must have shape [B,3]")
        batch, device = position.shape[0], position.device
        for name, value, width in (("velocity", velocity, 3), ("rpy", rpy, 3),
                                   ("predicted_lidar", predicted_lidar, 14),
                                   ("previous_acceleration", previous_acceleration, 2)):
            if value.shape != (batch, width) or value.device != device:
                raise ValueError(f"{name} has an invalid batch shape/device")
        if nominal_acceleration is not None and nominal_accel is not None:
            raise ValueError("Pass one nominal acceleration keyword")
        nominal = nominal_acceleration if nominal_acceleration is not None else nominal_accel
        if nominal is None or nominal.shape != (batch, 2) or nominal.device != device:
            raise ValueError("Provide nominal_acceleration [B,2] from the frozen controller")
        with torch.autocast(device_type=device.type, enabled=False):
            original = [position, velocity, rpy, predicted_lidar, previous_acceleration, nominal]
            input_valid = torch.ones(batch, dtype=torch.bool, device=device)
            for value in original:
                input_valid &= torch.isfinite(value).all(1)
            p, v, attitude, lidar, previous, nominal = [torch.nan_to_num(value.double(), nan=0.0,
                posinf=0.0, neginf=0.0) for value in original]
            cfg = self.config
            step = self._batch_scalar(cfg.dt if dt is None else dt, batch, device, "dt")
            input_valid &= torch.isfinite(step) & (step > 0)
            step = torch.nan_to_num(step, nan=cfg.dt, posinf=cfg.dt, neginf=cfg.dt).clamp_min(1e-4)
            bounds = (self.calibration_bounds.expand(batch, 7) if uncertainty_bounds is None
                      else torch.as_tensor(uncertainty_bounds, dtype=torch.float64, device=device))
            if bounds.shape != (batch, 7) or bounds.requires_grad:
                raise ValueError("uncertainty_bounds must be fixed calibrated [B,7] values, not learned")
            input_valid &= torch.isfinite(bounds).all(1) & (bounds >= 0).all(1)
            bounds = torch.nan_to_num(bounds, nan=7.0, posinf=7.0, neginf=7.0).clamp_min(0)
            input_valid &= (lidar[:, :7] >= 0).all(1) & (lidar[:, :7] <= 7).all(1)
            input_valid &= (lidar[:, 7:] >= 0).all(1) & (lidar[:, 7:] <= 1).all(1)
            input_valid &= (previous.abs() <= cfg.acceleration_limit_mps2 + 1e-8).all(1)
            previous = previous.clamp(-cfg.acceleration_limit_mps2, cfg.acceleration_limit_mps2)
            reference_values = []
            for name in ("p", "v", "a"):
                value = torch.as_tensor(reference[name], dtype=torch.float64, device=device)
                if value.ndim != 2 or value.shape[0] != batch or value.shape[1] not in (2, 3):
                    raise ValueError(f"reference[{name}] must have shape [B,2] or [B,3]")
                input_valid &= torch.isfinite(value).all(1)
                reference_values.append(torch.nan_to_num(value[:, :2], nan=0., posinf=0., neginf=0.))
            refp, refv, refa = reference_values
            rho, rhodot, rhoddot = [self._batch_scalar(reference[name], batch, device, name)
                                    for name in ("rho", "rhodot", "rhoddot")]
            input_valid &= torch.isfinite(rho) & torch.isfinite(rhodot) & torch.isfinite(rhoddot) & (rho > 0)
            rho = torch.nan_to_num(rho, nan=1., posinf=1., neginf=1.).clamp_min(1e-4)
            rhodot = torch.nan_to_num(rhodot, nan=0., posinf=0., neginf=0.)
            rhoddot = torch.nan_to_num(rhoddot, nan=0., posinf=0., neginf=0.)

            orientation = torch.stack((torch.sin(attitude), torch.cos(attitude)), dim=2).reshape(batch, 6)
            head_input = torch.cat((v[:, :2] / 5, orientation, lidar[:, :7] / 7, lidar[:, 7:]), dim=1)
            if cfg.learned_k2:
                head_dtype = self.gain_head[0].weight.dtype
                logits = self.gain_head(head_input.to(head_dtype)).double().squeeze(1)
                k2 = cfg.k2_min + (cfg.k2_max - cfg.k2_min) * torch.sigmoid(logits)
            else:
                k2 = p.new_full((batch,), cfg.k2_fixed)
            product, total_gain = cfg.k1 * k2, cfg.k1 + k2
            vertices, horizontal_cos, cone_center, cone_half, angular_valid = self._angles(attitude)
            speed2 = v[:, :2].square().sum(1)
            sampling_margin = torch.linalg.vector_norm(v[:, :2], dim=1) * step + 0.5 * (
                math.sqrt(2) * cfg.acceleration_limit_mps2 + cfg.disturbance_bound_mps2) * step.square()
            lower = ((lidar[:, :7] - bounds) * horizontal_cos[:, None] - cfg.mount_lever_m
                     - cfg.beam_sampling_margin_m - sampling_margin[:, None])
            radius = cfg.vehicle_radius_m + cfg.clearance_m
            safe_lower = lower.clamp_min(1e-4)
            stationary_squared = 2 * speed2 / product - radius ** 2
            stationary = torch.where(stationary_squared > 0,
                torch.sqrt(stationary_squared.clamp_min(1e-12)), torch.zeros_like(stationary_squared))
            distance = torch.maximum(safe_lower, stationary[:, None])
            beta = 0.5 * product[:, None] * distance + (
                speed2[:, None] - 0.5 * product[:, None] * radius ** 2) / distance
            beta = beta - cfg.disturbance_bound_mps2
            projected_velocity = (vertices * v[:, None, None, :2]).sum(-1)
            obstacle_rhs = beta[:, :, None] - total_gain[:, None, None] * projected_velocity
            obstacle_rows = torch.cat((vertices, vertices.new_zeros(batch, 7, 3, 1)), dim=-1).reshape(batch, 21, 3)
            obstacle_rhs = obstacle_rhs.reshape(batch, 21)
            h_obstacle = lower.clamp_min(0).square() - radius ** 2
            worst_approach = projected_velocity.amax(2)
            psi_distance = torch.maximum(safe_lower, worst_approach / cfg.k1)
            psi1 = cfg.k1 * (psi_distance.square() - radius ** 2) - 2 * psi_distance * worst_approach
            cbf_valid = input_valid & angular_valid & (lower >= radius).all(1) & (psi1 >= 0).all(1)

            center_n = torch.stack((torch.cos(cone_center), torch.sin(cone_center)), dim=1)
            tangent = torch.stack((-center_n[:, 1], center_n[:, 0]), dim=1)
            tangent_half = torch.tan(cone_half.clamp(1e-4, math.radians(89)))
            cone_rows_xy = torch.stack((tangent - tangent_half[:, None] * center_n,
                -tangent - tangent_half[:, None] * center_n, -center_n), dim=1)
            cone_rows = torch.cat((step[:, None, None] * cone_rows_xy, p.new_zeros(batch, 3, 1)), dim=2)
            cone_rhs = -(cone_rows_xy * v[:, None, :2]).sum(2)
            error, error_rate = p[:, :2] - refp, v[:, :2] - refv
            tracking_h = rho.square() - error.square().sum(1)
            tracking_hdot = 2 * rho * rhodot - 2 * (error * error_rate).sum(1)
            tracking_rhs = (2 * (rhodot.square() + rho * rhoddot) - 2 * error_rate.square().sum(1)
                + 2 * (error * refa).sum(1) + total_gain * tracking_hdot + product * tracking_h
                - 2 * torch.linalg.vector_norm(error, dim=1) * cfg.disturbance_bound_mps2)
            tracking_row = torch.cat((2 * error, -p.new_ones(batch, 1)), dim=1)[:, None, :]
            lo = torch.maximum(p.new_full((batch, 2), -cfg.acceleration_limit_mps2),
                               previous - cfg.jerk_limit_mps3 * step[:, None])
            hi = torch.minimum(p.new_full((batch, 2), cfg.acceleration_limit_mps2),
                               previous + cfg.jerk_limit_mps3 * step[:, None])
            box_xy = p.new_tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
            box = box_xy.unsqueeze(0).expand(batch, 4, 3)
            box_rhs = torch.cat((hi, -lo), dim=1)
            nonnegative_slack = p.new_tensor([[[0., 0., -1.]]]).expand(batch, 1, 3)
            G = torch.cat((obstacle_rows, cone_rows, box, tracking_row, nonnegative_slack), dim=1)
            rhs = torch.cat((obstacle_rhs, cone_rhs, box_rhs, tracking_rhs[:, None], p.new_zeros(batch, 1)), dim=1)
            Q = torch.diag(p.new_tensor([1., 1., 2 * cfg.tracking_slack_penalty])).unsqueeze(0).expand(batch, 3, 3)
            linear = torch.cat((-nominal, p.new_zeros(batch, 1)), dim=1)
            result = self.qp(Q, linear, G, rhs)
            feasible = cbf_valid & result["feasible"] & result["converged"]
            candidate = result["solution"]
            # This is explicitly a separate emergency policy, not post-clipping
            # of a feasible QP solution and not a claimed feasible safety action.
            emergency = torch.maximum(lo, torch.minimum(hi, -v[:, :2] / step[:, None]))
            emergency_tracking_slack = (2 * (error * emergency).sum(1) - tracking_rhs).clamp_min(0)
            fallback = torch.cat((emergency, emergency_tracking_slack[:, None]), dim=1)
            selected = torch.where(feasible[:, None], candidate, fallback)
            true_residual = _mv(G, selected) - rhs
            true_violation = true_residual.clamp_min(0)
            return {
                "acceleration": selected[:, :2], "tracking_slack": selected[:, 2],
                "feasible": feasible, "max_violation": true_violation.amax(1),
                "trueinequality_violation": true_violation,
                "true_inequality_violation": true_violation,
                "inequality_residual": true_residual,
                "cbf_valid": cbf_valid, "input_valid": input_valid,
                "angular_valid": angular_valid, "barrier_values": h_obstacle,
                "psi1": psi1, "distance_lower_m": lower,
                "motion_sampling_margin_m": sampling_margin,
                "beam_sampling_margin_m": p.new_full((batch,), cfg.beam_sampling_margin_m),
                "tracking_barrier": tracking_h, "tracking_barrier_dot": tracking_hdot,
                "k2": k2, "sector_vertices": vertices, "constraint_matrix": G,
                "constraint_rhs": rhs, "acceleration_lower": lo, "acceleration_upper": hi,
                "qp_solution": candidate, "qp_feasible": result["feasible"],
                "qp_converged": result["converged"], "qp_max_violation": result["max_violation"],
                "qp_dual_residual": result["dual_residual"], "emergency_used": ~feasible,
                "visibility_center_rad": cone_center, "visibility_half_angle_rad": cone_half,
            }
