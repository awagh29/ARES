"""The baseline model's ``Controller`` subsystem (position -> velocity -> acceleration).

Simulink discrete PID blocks, parallel form, Forward Euler integrator and
derivative filter, output saturation (``models/ares_single_uav_top.slx``):

    u[k] = P e[k] + I x[k] + D N (e[k] - s[k]),   s[k+1] = s[k] + Ts N (e[k] - s[k])

Position loop: P=6/6/6.5, D=1.5/1.5/2.5, N=10/10/14.4947, I=0, |v_cmd| <= 3 m/s.
Velocity loop: same gains, |a_cmd| <= 1/1/4 m/s^2 (N/E/D).
Yaw: yaw_rate = clamp(2 * (yaw_ref - yaw), +-0.5 rad/s), yaw_ref NaN -> 0.
The MATLAB model runs these at 1 kHz; the Gazebo runner uses 100 Hz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# initializeSingleUav.m
GAINS_P = (6.0, 6.0, 6.5)
GAINS_I = (0.0, 0.0, 0.0)
GAINS_D = (1.5, 1.5, 2.5)
GAINS_N = (10.0, 10.0, 14.4947065605712)
VELOCITY_LIMITS = (3.0, 3.0, 3.0)  # position-loop output saturation
ACCELERATION_LIMITS = (1.0, 1.0, 4.0)  # velocity-loop output saturation
YAW_GAIN = 2.0
YAW_RATE_LIMIT = 0.5


@dataclass
class DiscretePID:
    p: float
    i: float
    d: float
    n: float
    limit: float
    ts: float
    integrator: float = 0.0
    filter_state: float = 0.0

    def step(self, error: float) -> float:
        derivative = self.d * self.n * (error - self.filter_state)
        output = self.p * error + self.i * self.integrator + derivative
        saturated = min(max(output, -self.limit), self.limit)
        # Clamping anti-windup: freeze the integrator while saturated in the same direction.
        if not (saturated != output and error * output > 0):
            self.integrator += self.ts * error
        self.filter_state += self.ts * self.n * (error - self.filter_state)
        return saturated


class PidCascade:
    def __init__(self, ts: float):
        self.ts = ts
        self.position = [DiscretePID(GAINS_P[k], GAINS_I[k], GAINS_D[k], GAINS_N[k], VELOCITY_LIMITS[k], ts)
                         for k in range(3)]
        self.velocity = [DiscretePID(GAINS_P[k], GAINS_I[k], GAINS_D[k], GAINS_N[k], ACCELERATION_LIMITS[k], ts)
                         for k in range(3)]

    def step(self, reference_ned, position_ned, velocity_ned) -> tuple[np.ndarray, np.ndarray]:
        """Return (acceleration command NED, velocity command NED)."""
        error = np.asarray(reference_ned, float) - np.asarray(position_ned, float)
        velocity_command = np.array([self.position[k].step(float(error[k])) for k in range(3)])
        velocity_error = velocity_command - np.asarray(velocity_ned, float)
        acceleration = np.array([self.velocity[k].step(float(velocity_error[k])) for k in range(3)])
        return acceleration, velocity_command


def yaw_rate_command(yaw_reference: float, yaw: float) -> float:
    """Attitude Control subsystem: Sum(-yaw, +ref) -> gain 2 -> saturation (no angle wrapping)."""
    if yaw_reference is None or not math.isfinite(yaw_reference):
        yaw_reference = 0.0
    return min(max(YAW_GAIN * (yaw_reference - yaw), -YAW_RATE_LIMIT), YAW_RATE_LIMIT)
