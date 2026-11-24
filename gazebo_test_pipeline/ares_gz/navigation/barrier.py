"""Port of ``correctMotion.m``: hard map/measurement CBF-QP with acceleration and slew bounds.

For a convex box, h = distance_to_box - radius and n is its outward normal.
The filter enforces  n'u + 2 k n'v + k^2 h >= allowance  for every box,
the six workspace planes and each extra measured point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import EPS, map_clearance
from .options import NavigationOptions
from .qp import project_control


@dataclass(frozen=True)
class BarrierInfo:
    Feasible: bool
    MaximumViolation: float
    MinimumH: float
    MinimumPsi1: float


def correct_motion(position, velocity, nominal, previous, box_min: np.ndarray, box_max: np.ndarray,
                   options: NavigationOptions, measured_world_points=None) -> tuple[np.ndarray, BarrierInfo]:
    o = options
    p = np.asarray(position, dtype=float).reshape(3)
    v = np.asarray(velocity, dtype=float).reshape(3)
    radius = o.VehicleRadiusM + o.MinimumSurfaceClearanceM + o.GeometryUncertaintyM + o.TrackingMarginM
    distance, normals = map_clearance(p, box_min, box_max)
    h = distance - radius
    bounds = o.bounds
    normals = np.vstack((normals, np.eye(3), -np.eye(3)))
    h = np.concatenate((h, p - bounds[:, 0], bounds[:, 1] - p))
    if measured_world_points is not None and np.size(measured_world_points) > 0:
        points = np.asarray(measured_world_points, dtype=float).reshape(-1, 3)
        delta = p - points
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        # A return at the vehicle centre keeps negative h and a zero normal so the QP rejects it.
        normals = np.vstack((normals, delta / np.maximum(distances, EPS)[:, None]))
        h = np.concatenate((h, distances - radius))
    hdot = normals @ v
    gain = o.BarrierGain
    allowance = o.AccelerationUncertaintyMps2 + gain * o.AccelerationLimitMps2 * o.ControlSampleTimeS
    A = -normals
    b = 2 * gain * hdot + gain ** 2 * h - allowance
    change = o.JerkLimitMps3 * o.ControlSampleTimeS
    previous = np.asarray(previous, dtype=float).reshape(3)
    lower = np.maximum(-o.AccelerationLimitMps2, previous - change)
    upper = np.minimum(o.AccelerationLimitMps2, previous + change)
    candidate, solved = project_control(nominal, A, b, lower, upper)
    minimum_h = float(np.min(h))
    minimum_psi1 = float(np.min(hdot + gain * h))
    feasible = bool(solved and candidate.size == 3 and np.all(np.isfinite(candidate)))
    violation = 1e6
    if feasible:
        violation = float(np.max(np.concatenate(([0.0], A @ candidate - b, lower - candidate, candidate - upper))))
        feasible = violation <= 1e-6 and minimum_h >= -1e-5 and minimum_psi1 >= -1e-4
    if not feasible:
        # Bounded brake used only so the caller can stop; never labelled feasible or safe.
        candidate = np.minimum(np.maximum(-o.VelocityGain * v, lower), upper)
        violation = float(np.max(np.concatenate(([0.0], A @ candidate - b, lower - candidate, candidate - upper))))
    return candidate, BarrierInfo(feasible, violation, minimum_h, minimum_psi1)
