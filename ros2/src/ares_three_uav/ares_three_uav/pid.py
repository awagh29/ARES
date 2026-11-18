"""Discrete parallel PID used by the Simulink position and velocity loops.

Forward Euler integrator and forward-Euler derivative filter, output
saturation, and clamping anti-windup. Integral gains in this model are zero.
The derivative filter is ``N(z-1)/(z-1+N*Ts)`` applied to the error.
"""

from __future__ import annotations


class DiscretePID:
    def __init__(self, kp: float, ki: float, kd: float, n: float, lower: float, upper: float, dt: float):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.n = float(n)
        self.lower = float(lower)
        self.upper = float(upper)
        self.dt = float(dt)
        self.integral = 0.0
        self.previous_error = 0.0
        self.filter = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.filter = 0.0

    def step(self, error: float) -> float:
        self.integral += self.ki * self.dt * self.previous_error
        self.filter = (1.0 - self.n * self.dt) * self.filter + self.n * (error - self.previous_error)
        raw = self.kp * error + self.integral + self.kd * self.filter
        limited = min(self.upper, max(self.lower, raw))
        if limited != raw and ((raw > self.upper and error > 0.0) or (raw < self.lower and error < 0.0)):
            self.integral -= self.ki * self.dt * self.previous_error
        self.previous_error = error
        return limited
