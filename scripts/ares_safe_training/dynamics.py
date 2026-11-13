"""Differentiable version of the ARES Simulink multirotor guidance plant.

State order is [position_NED(3), velocity_NED(3), roll/pitch/yaw(3),
body_angular_rate_RPY(3), actual_thrust_N(1)]. Controls are commanded
[roll, pitch, Euler_yaw_rate, thrust]. This is the existing closed-loop
guidance/autopilot model, not a rotor/inertia or motor-hardware model.

Parameters must be supplied explicitly, normally from the JSON written by
calibrateSaitsBarrierDynamics.m. There are no implicit production constants.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn


CALIBRATION_SCHEMA = "ares-guidance-dynamics-calibration-v1"
STATE_NAMES = (
    "position_north_m", "position_east_m", "position_down_m",
    "velocity_north_mps", "velocity_east_mps", "velocity_down_mps",
    "roll_rad", "pitch_rad", "yaw_rad",
    "body_roll_rate_radps", "body_pitch_rate_radps", "body_yaw_rate_radps",
    "actual_thrust_n",
)
COMMAND_NAMES = ("roll_rad", "pitch_rad", "euler_yaw_rate_radps", "thrust_n")
PARAMETER_NAMES = (
    "mass_kg", "gravity_mps2", "roll_kp", "roll_kd", "pitch_kp", "pitch_kd",
    "yaw_rate_gain", "thrust_gain", "roll_limit_rad", "pitch_limit_rad",
    "yaw_rate_limit_radps",
)


def _parameters(value: Mapping | None) -> dict[str, float]:
    if value is None:
        raise ValueError("Supply explicit parameters from the MATLAB dynamics calibration JSON.")
    source = value["parameters"] if "parameters" in value else value
    missing = set(PARAMETER_NAMES) - set(source)
    if missing:
        raise ValueError(f"Dynamics calibration lacks parameters: {sorted(missing)}")
    result = {name: float(source[name]) for name in PARAMETER_NAMES}
    if any(not math.isfinite(number) or number <= 0 for number in result.values()):
        raise ValueError("Guidance dynamics parameters must be finite and positive.")
    if max(result["roll_limit_rad"], result["pitch_limit_rad"]) >= math.pi / 2:
        raise ValueError("Guidance attitude limits must remain below the Euler singularity.")
    return result


def load_calibration(path: str | Path) -> dict:
    """Read a measured calibration; retain its bounds/coverage/provenance."""
    target = Path(path).expanduser().resolve(strict=True)
    value = json.loads(target.read_text(encoding="utf-8-sig"))
    if value.get("schemaVersion") != CALIBRATION_SCHEMA:
        raise ValueError("Unsupported guidance dynamics calibration schema.")
    _parameters(value)
    if value.get("calibrationComplete") is not True:
        raise ValueError("The MATLAB dynamics calibration has not completed.")
    value["loaded_from"] = str(target)
    value["calibration_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return value


def initial_state(positions: Tensor) -> Tensor:
    """Match the plant's initial position and zero velocity/attitude/rates/thrust."""
    if positions.ndim < 2 or positions.shape[-1] != 3 or not positions.is_floating_point():
        raise ValueError("positions must be a floating-point [...,3] tensor.")
    return torch.cat((positions, positions.new_zeros((*positions.shape[:-1], 10))), dim=-1)


def acceleration_to_commands(acceleration: Tensor, yaw: Tensor, yaw_rate: Tensor,
                             *, parameters: Mapping) -> Tensor:
    """Allocate world-NED acceleration with the existing chart's exact formula.

    No clipping occurs after the barrier/QP. On the calibrated software box
    a_N/E in [-1,1] and a_D in [-4,4], the original +/-25-degree chart clamps
    are inactive, so this allocation is identical to that chart. The caller
    must enforce its declared acceleration/slew/attitude/yaw bounds upstream.
    """
    values = _parameters(parameters)
    if acceleration.ndim < 2 or acceleration.shape[-1] != 3:
        raise ValueError("acceleration must have shape [...,3].")
    if not all(value.is_floating_point() for value in (acceleration, yaw, yaw_rate)):
        raise ValueError("Acceleration, yaw and yaw_rate must be floating-point tensors.")
    if any(value.dtype != acceleration.dtype or value.device != acceleration.device
           for value in (yaw, yaw_rate)):
        raise ValueError("Acceleration, yaw and yaw_rate must share a dtype and device.")
    if yaw.shape == acceleration.shape[:-1] + (1,):
        yaw = yaw.squeeze(-1)
    if yaw_rate.shape == acceleration.shape[:-1] + (1,):
        yaw_rate = yaw_rate.squeeze(-1)
    if yaw.shape != acceleration.shape[:-1] or yaw_rate.shape != yaw.shape:
        raise ValueError("yaw and yaw_rate must match the acceleration batch dimensions.")
    a_north, a_east, a_down = acceleration.unbind(-1)
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    vertical = values["gravity_mps2"] - a_down
    pitch = torch.atan2(-a_north * cosine - a_east * sine, vertical)
    roll = torch.atan2((-a_north * sine + a_east * cosine) * torch.cos(pitch), vertical)
    thrust = values["mass_kg"] * vertical / (torch.cos(pitch) * torch.cos(roll))
    return torch.stack((roll, pitch, yaw_rate, thrust), dim=-1)


class GuidanceDynamics(nn.Module):
    """The exact 13-state ODE plus stable batched zero-order-hold integration.

    RK4 integrates the first twelve states with steps <=1 ms. Actual thrust
    uses its analytic first-order solution at all RK4 stages and endpoints;
    applying Euler or RK4 directly to the 3900/s thrust lag at 1 ms is unstable.
    The 1950/s yaw dynamics remain inside RK4's stability interval at 1 ms.
    No tensor is moved to the CPU and no .item() synchronization occurs inside
    integration. This implementation supports ordinary autograd and optional
    torch.compile around ``forward`` or the RK4 step at the caller's discretion.
    """

    def __init__(self, parametersdict: Mapping | None = None, integration_dt: float = 0.001):
        super().__init__()
        self.parameters_dict = _parameters(parametersdict)
        self.integration_dt = float(integration_dt)
        if not math.isfinite(self.integration_dt) or not 0 < self.integration_dt <= 0.001:
            raise ValueError("integration_dt must be finite, positive and no larger than 0.001 s.")
        if self.parameters_dict["yaw_rate_gain"] * self.integration_dt > 2.5:
            raise ValueError("integration_dt is too large for the calibrated yaw-rate lag.")

    @classmethod
    def from_calibration(cls, path: str | Path, integration_dt: float = 0.001):
        calibration = load_calibration(path)
        result = cls(calibration["parameters"], integration_dt)
        result.calibration = calibration
        return result

    @staticmethod
    def initial_state(positions: Tensor) -> Tensor:
        return initial_state(positions)

    def acceleration_to_commands(self, acceleration: Tensor, yaw: Tensor, yaw_rate: Tensor) -> Tensor:
        return acceleration_to_commands(acceleration, yaw, yaw_rate, parameters=self.parameters_dict)

    @staticmethod
    def _check_shapes(state: Tensor, commands: Tensor) -> None:
        if state.ndim < 2 or state.shape[-1] != 13 or commands.shape != state.shape[:-1] + (4,):
            raise ValueError("State/commands must have matching [...,13] and [...,4] shapes.")
        if not state.is_floating_point() or not commands.is_floating_point():
            raise ValueError("State and commands must be floating point.")
        if state.dtype != commands.dtype or state.device != commands.device:
            raise ValueError("State and commands must use the same dtype and device.")

    def _derivative12(self, state12: Tensor, thrust: Tensor, commands: Tensor) -> Tensor:
        params = self.parameters_dict
        velocity = state12[..., 3:6]
        roll, pitch, _ = state12[..., 6:9].unbind(-1)
        sin_roll, sin_pitch, sin_yaw = torch.sin(state12[..., 6:9]).unbind(-1)
        cos_roll, cos_pitch, cos_yaw = torch.cos(state12[..., 6:9]).unbind(-1)
        p, q, r = state12[..., 9:12].unbind(-1)
        thrust_mass = thrust / params["mass_kg"]
        acceleration = torch.stack((
            -thrust_mass * (cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll),
            -thrust_mass * (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll),
            params["gravity_mps2"] - thrust_mass * cos_pitch * cos_roll,
        ), dim=-1)
        # Exactly the installed guidance model's Euler/body-rate maps. Do not
        # clamp cos(pitch): that would silently replace its actual dynamics.
        common = sin_roll * q + cos_roll * r
        roll_rate = p + (sin_pitch / cos_pitch) * common
        pitch_rate = cos_roll * q - sin_roll * r
        yaw_rate = common / cos_pitch
        euler_rate = torch.stack((roll_rate, pitch_rate, yaw_rate), dim=-1)
        roll_acceleration = params["roll_kp"] * (commands[..., 0] - roll) - params["roll_kd"] * roll_rate
        pitch_acceleration = params["pitch_kp"] * (commands[..., 1] - pitch) - params["pitch_kd"] * pitch_rate
        yaw_acceleration = params["yaw_rate_gain"] * (commands[..., 2] - yaw_rate)
        body_rate_derivative = torch.stack((
            roll_acceleration - sin_pitch * yaw_acceleration,
            cos_roll * pitch_acceleration + sin_roll * cos_pitch * yaw_acceleration,
            -sin_roll * pitch_acceleration + cos_roll * cos_pitch * yaw_acceleration,
        ), dim=-1)
        return torch.cat((velocity, acceleration, euler_rate, body_rate_derivative), dim=-1)

    def derivative(self, state: Tensor, commands: Tensor) -> Tensor:
        """Continuous 13-state derivative in the canonical RPY state ordering."""
        self._check_shapes(state, commands)
        thrust_rate = self.parameters_dict["thrust_gain"] * (commands[..., 3] - state[..., 12])
        return torch.cat((self._derivative12(state[..., :12], state[..., 12], commands),
                          thrust_rate.unsqueeze(-1)), dim=-1)

    def _rk4_step(self, state: Tensor, commands: Tensor, step: float) -> Tensor:
        """One <=1 ms step; a suitable small torch.compile compilation boundary."""
        half_decay = math.exp(-self.parameters_dict["thrust_gain"] * step * 0.5)
        full_decay = half_decay * half_decay
        state12, thrust = state[..., :12], state[..., 12]
        commanded_thrust = commands[..., 3]
        delta = thrust - commanded_thrust
        half_thrust = commanded_thrust + delta * half_decay
        next_thrust = commanded_thrust + delta * full_decay
        first = self._derivative12(state12, thrust, commands)
        second = self._derivative12(state12 + 0.5 * step * first, half_thrust, commands)
        third = self._derivative12(state12 + 0.5 * step * second, half_thrust, commands)
        fourth = self._derivative12(state12 + step * third, next_thrust, commands)
        state12 = state12 + (step / 6.0) * (first + 2.0 * second + 2.0 * third + fourth)
        return torch.cat((state12, next_thrust.unsqueeze(-1)), dim=-1)

    def step(self, state: Tensor, commands: Tensor, sub_dt: float | None = None) -> Tensor:
        """Public single-substep API; commands are held over sub_dt <=1 ms."""
        self._check_shapes(state, commands)
        duration = self.integration_dt if sub_dt is None else float(sub_dt)
        if not math.isfinite(duration) or not 0 < duration <= self.integration_dt:
            raise ValueError("sub_dt must be positive and no greater than integration_dt.")
        function = getattr(self, "_compiled_rk4_step", self._rk4_step)
        return function(state, commands, duration)

    def prepare_compile(self, device=None, *, backend="inductor", fullgraph=True):
        """Lazily compile ONE RK4 substep, avoiding a 100-step unrolled graph.

        The first tensor call selects the target device. The optional device
        argument documents that intent; it does not move or cast any data.
        Compilation/throughput must be measured on the actual Colab GPU.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not support torch.compile.")
        self.compile_target_device = None if device is None else str(torch.device(device))
        self._compiled_rk4_step = torch.compile(self._rk4_step, backend=backend,
                                               fullgraph=fullgraph, dynamic=False)
        return self

    def forward(self, state: Tensor, commands: Tensor, dt: float, *, return_trace=False):
        """Hold commands over dt, optionally returning every substep's position.

        With return_trace=True return (next_state, positions[...,steps+1,3]),
        including the initial position. Collision scoring can therefore inspect
        the same <=1 ms integration grid instead of only sensor-rate endpoints.
        """
        self._check_shapes(state, commands)
        duration = float(dt)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("The zero-order-hold duration must be finite and positive.")
        substeps = max(1, int(math.ceil(duration / self.integration_dt - 1e-12)))
        step = duration / substeps
        function = getattr(self, "_compiled_rk4_step", self._rk4_step)
        trace = [state[..., :3]] if return_trace else None
        for _ in range(substeps):
            state = function(state, commands, step)
            if return_trace:
                trace.append(state[..., :3])
        if return_trace:
            return state, torch.stack(trace, dim=-2)
        return state
