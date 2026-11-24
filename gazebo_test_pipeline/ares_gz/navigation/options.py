"""Port of ``defaultOptions.m`` and the checks in ``validateOptions.m``.

Field names are the MATLAB names, so a MATLAB ``MapCorrectionOptions`` struct
exported to JSON can be passed straight to :func:`validate_options`.
``tests/test_parity.py`` checks these defaults against MATLAB.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Mapping

import numpy as np

LIDAR_MODES = ("healthy", "intermittent", "blackout", "custom")


@dataclass(frozen=True)
class NavigationOptions:
    StartState: tuple = (0.0, 0.0, -7.0, 0.0, 0.0, 0.0)  # NED [N E D vN vE vD]
    GoalState: tuple = (30.0, 35.0, -7.0, 0.0, 0.0, 0.0)
    StartEulerZYXRad: tuple = (0.0, 0.0, 0.0)  # [yaw pitch roll]
    StartBodyRatesRadps: tuple = (0.0, 0.0, 0.0)
    StopTimeS: float = 60.0
    ShowSimulation: bool = False
    LidarMode: str = "intermittent"
    NumberOfOutages: int = 5
    OutageDurationS: tuple = (1.0, 2.0)
    InitialHealthyTimeS: float = 2.0
    FinalHealthyTimeS: float = 1.0
    MinimumRecoveryTimeS: float = 1.0
    OutageWindowEndS: float = 20.0
    OutageIntervalsS: tuple = ()  # rows (start, end), "custom" mode only
    Seed: int = 513
    ControlSampleTimeS: float = 0.05
    GridResolutionM: float = 1.0
    WorldBoundsNedM: tuple = ((-10.0, 45.0), (-10.0, 50.0), (-20.0, -2.0))  # rows N, E, D
    VehicleRadiusM: float = 0.1
    MinimumSurfaceClearanceM: float = 1.0
    GeometryUncertaintyM: float = 0.15
    TrackingMarginM: float = 0.25
    PlanningExtraMarginM: float = 0.6
    ClearanceWeight: float = 1.5
    CruiseSpeedMps: float = 2.5
    BrakingDecelerationMps2: float = 1.0
    AccelerationLimitMps2: float = 1.5
    JerkLimitMps3: float = 3.0
    VelocityGain: float = 2.0
    PositionGain: float = 1.2
    BarrierGain: float = 1.2
    AccelerationUncertaintyMps2: float = 0.2
    GoalPositionToleranceM: float = 0.35
    GoalVelocityToleranceMps: float = 0.25
    WaypointToleranceM: float = 0.45

    # Convenience numpy views -------------------------------------------------
    @property
    def start_state(self) -> np.ndarray:
        return np.asarray(self.StartState, dtype=float)

    @property
    def goal_state(self) -> np.ndarray:
        return np.asarray(self.GoalState, dtype=float)

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.WorldBoundsNedM, dtype=float).reshape(3, 2)

    @property
    def outage_intervals(self) -> np.ndarray:
        return np.asarray(self.OutageIntervalsS, dtype=float).reshape(-1, 2)

    @property
    def safety_radius_m(self) -> float:
        """body radius + clearance + geometry uncertainty + tracking margin (1.5 m)."""
        return (self.VehicleRadiusM + self.MinimumSurfaceClearanceM
                + self.GeometryUncertaintyM + self.TrackingMarginM)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_options() -> NavigationOptions:
    return NavigationOptions()


def _as_tuple(value: Any) -> Any:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return tuple(float(item) for item in array)
    return tuple(tuple(float(item) for item in row) for row in array)


def validate_options(overrides: Mapping[str, Any] | NavigationOptions | None = None) -> NavigationOptions:
    """Merge partial inputs into the defaults and apply validateOptions.m checks."""
    if isinstance(overrides, NavigationOptions):
        options = overrides
    else:
        options = default_options()
        known = {item.name for item in fields(NavigationOptions)}
        changes: dict[str, Any] = {}
        for name, value in dict(overrides or {}).items():
            if name not in known:
                raise ValueError(f"Unknown map navigation option: {name}")
            changes[name] = value
        for name in ("StartState", "GoalState", "StartEulerZYXRad", "StartBodyRatesRadps",
                     "OutageDurationS", "WorldBoundsNedM", "OutageIntervalsS"):
            if name in changes:
                changes[name] = _as_tuple(changes[name])
        if "LidarMode" in changes:
            changes["LidarMode"] = str(changes["LidarMode"])
        if "NumberOfOutages" in changes:
            changes["NumberOfOutages"] = int(changes["NumberOfOutages"])
        if "Seed" in changes:
            changes["Seed"] = int(changes["Seed"])
        if "ShowSimulation" in changes:
            changes["ShowSimulation"] = bool(changes["ShowSimulation"])
        options = replace(options, **changes)

    o = options
    for name, size in (("StartState", 6), ("GoalState", 6), ("StartEulerZYXRad", 3), ("StartBodyRatesRadps", 3)):
        value = np.asarray(getattr(o, name), dtype=float)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be {size} finite values")
    positive = ["StopTimeS", "ControlSampleTimeS", "GridResolutionM", "VehicleRadiusM",
                "MinimumSurfaceClearanceM", "PlanningExtraMarginM", "CruiseSpeedMps",
                "BrakingDecelerationMps2", "AccelerationLimitMps2", "JerkLimitMps3",
                "VelocityGain", "PositionGain", "BarrierGain", "GoalPositionToleranceM",
                "GoalVelocityToleranceMps", "WaypointToleranceM", "OutageWindowEndS"]
    for name in positive:
        value = float(getattr(o, name))
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive")
    nonnegative = ["GeometryUncertaintyM", "TrackingMarginM", "ClearanceWeight",
                   "AccelerationUncertaintyMps2", "InitialHealthyTimeS", "FinalHealthyTimeS",
                   "MinimumRecoveryTimeS"]
    for name in nonnegative:
        value = float(getattr(o, name))
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be nonnegative")
    if o.NumberOfOutages < 0:
        raise ValueError("NumberOfOutages must be a nonnegative integer")
    if not (0 <= o.Seed <= 2**32 - 1):
        raise ValueError("Seed must be in [0, 2^32-1]")
    bounds = o.bounds
    if not np.all(np.diff(bounds, axis=1) > 0):
        raise ValueError("World bounds must increase.")
    if not bounds[2, 1] < 0:
        raise ValueError("Use negative Down coordinates above ground.")
    ratio = 0.1 / o.ControlSampleTimeS
    if not (o.ControlSampleTimeS <= 0.1 and abs(ratio - round(ratio)) < 1e-8):
        raise ValueError("ControlSampleTimeS must divide the 0.1 s LiDAR period.")
    if abs(o.StopTimeS / 0.1 - round(o.StopTimeS / 0.1)) >= 1e-8:
        raise ValueError("StopTimeS must be a multiple of 0.1 s.")
    if np.linalg.norm(o.start_state[3:6]) > o.CruiseSpeedMps:
        raise ValueError("Start speed exceeds CruiseSpeedMps.")
    if np.linalg.norm(o.goal_state[3:6]) >= 1e-12:
        raise ValueError("This mission ends in hover: use zero goal velocity.")
    if not np.all(np.abs(np.asarray(o.StartEulerZYXRad)[1:3]) <= 25 * math.pi / 180):
        raise ValueError("Initial roll/pitch must be within 25 degrees.")
    if not (o.AccelerationLimitMps2 <= 1.5 and o.JerkLimitMps3 <= 3):
        raise ValueError("Use acceleration <=1.5 m/s^2 and jerk <=3 m/s^3 for this allocator.")
    if o.LidarMode not in LIDAR_MODES:
        raise ValueError("Choose healthy, intermittent, blackout or custom.")
    duration = np.asarray(o.OutageDurationS, dtype=float)
    if duration.shape != (2,) or np.any(duration <= 0) or duration[1] < duration[0]:
        raise ValueError("OutageDurationS must be two increasing positive values.")
    return o
