"""Frozen differentiable controller and smooth reference used in both stages.

This is a new acceleration-interface controller for co-design. It is explicitly
not a differentiable imitation of the existing discrete VFH selector or PID
filter states. Deployment must retain this controller, allocation and safety QP.
"""
from __future__ import annotations

import torch


def smooth_reference(initial_position, goal, time, speed=2.5, tube_radius=2.5):
    """C2 minimum-jerk reference with analytic first and second derivatives."""
    displacement = goal - initial_position
    duration = (1.875 * torch.linalg.vector_norm(displacement, dim=-1) / speed).clamp_min(10.)
    t = torch.as_tensor(time, dtype=initial_position.dtype, device=initial_position.device)
    tau = (t / duration).clamp(0, 1)
    blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    first = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    second = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    return {"p": initial_position + displacement * blend[:, None],
            "v": displacement * first[:, None], "a": displacement * second[:, None],
            "rho": torch.full_like(duration, tube_radius),
            "rhodot": torch.zeros_like(duration), "rhoddot": torch.zeros_like(duration)}


def nominal_acceleration(state, lidar, reference, config):
    """Fixed, differentiable tracking plus observed-sector repulsion."""
    position, velocity = state[:, :3], state[:, 3:6]
    tracking = (reference["a"] + config.nominal_position_gain * (reference["p"] - position)
                + config.nominal_velocity_gain * (reference["v"] - velocity))
    bearings = torch.linspace(-60 + 60 / 7, 60 - 60 / 7, 7, device=state.device, dtype=state.dtype)
    bearings = bearings * (torch.pi / 180) + state[:, 8, None]
    directions = torch.stack((bearings.cos(), bearings.sin()), dim=-1)
    distance = lidar[:, :7].clamp_min(.2)
    # Return fractions are occupancy estimates, not calibrated confidences.
    weight = lidar[:, 7:].clamp(0, 1) * torch.relu(1 / distance - 1 / 7) / distance**2
    repulsion = -(directions * weight[..., None]).sum(dim=1) * config.nominal_repulsion_gain
    horizontal = torch.tanh(tracking[:, :2] + repulsion)
    vertical = tracking[:, 2].clamp(-4, 4)
    return torch.cat((horizontal, vertical[:, None]), dim=-1)


def desired_yaw_rate(state, goal, maximum=.5):
    delta = goal[:, :2] - state[:, :2]
    # atan2(0,0) has undefined derivatives: retain heading at the goal.
    near = delta.square().sum(dim=-1) < 1e-8
    safe_delta = torch.where(near[:, None], torch.tensor([1., 0.], device=state.device), delta)
    desired = torch.atan2(safe_delta[:, 1], safe_delta[:, 0])
    error = torch.atan2(torch.sin(desired - state[:, 8]), torch.cos(desired - state[:, 8]))
    return torch.where(near, torch.zeros_like(error), maximum * torch.tanh(2 * error / maximum))
