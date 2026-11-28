"""Common controller interface used by the runner."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..navigation.options import NavigationOptions


@dataclass(frozen=True)
class VehicleState:
    time_s: float
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    euler_zyx: np.ndarray  # [yaw pitch roll]
    body_rates_frd: np.ndarray


@dataclass
class Command:
    acceleration_ned: np.ndarray
    yaw_rate_ned: float  # positive clockwise seen from above (NED z)
    stop: bool = False  # controller asks to end the run (goal at rest, or its own failure)
    goal_reached: bool = False
    healthy: bool = True  # False = the controller itself failed (infeasible filter, invalid prediction)
    record: dict = field(default_factory=dict)  # extra columns for control.csv


class Controller:
    """Lifecycle: ``start`` once per run, ``step`` every ``tick_s`` seconds, ``finish`` at the end."""

    name = "controller"
    tick_s = 0.05
    guidance_period_s = 0.1  # how often the controller consumes a lidar scan

    def start(self, options: NavigationOptions, route_hint=None) -> dict:
        raise NotImplementedError

    def step(self, state: VehicleState, lidar_available: bool, points_matlab_order) -> Command:
        raise NotImplementedError

    def finish(self) -> None:
        pass

    def summary_extras(self) -> dict:
        """Controller-specific numbers added to the run's summary.json."""
        return {}

    def close(self) -> None:
        pass
