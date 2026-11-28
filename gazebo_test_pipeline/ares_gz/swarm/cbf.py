"""Pairwise separation invariant: control barrier function (CBF) filter.

Safe set for every drone pair (i, j):  h_ij = ||p_i - p_j||^2 - D_ij^2 >= 0.
With single-integrator velocity commands (p_dot = u) the set is forward
invariant if  dh/dt >= -gamma * h, i.e.

    2 (p_i - p_j)^T (u_i - u_j) >= -gamma * h_ij        (Wang, Ames & Egerstedt 2017)

The hub splits each pair constraint between the two drones (decentralised
safety barrier certificates): drone i enforces its share alpha_ij,

    2 (p_i - p_j)^T u_i >= -alpha_ij * gamma * h_ij,     alpha_ij + alpha_ji = 1,

so the pair constraint holds whenever both drones honour their own share,
and a drone needs only the other's *position* (not its command). That makes
the invariant robust to message delay; stale positions are covered by
inflating D_ij by v_max * age. A drone that stops (u = 0) always satisfies
its share while h >= 0.

Each drone projects its fresh nominal velocity onto its constraints and its
speed box with the small QP below (Hildreth's dual coordinate ascent; exact
for the few constraints involved).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeparationContract:
    """What the hub tells drone i about one neighbour j."""

    neighbour: str
    position_ned: np.ndarray  # neighbour position as known to the hub
    radius_m: float  # required centre distance D_ij for this pair (incl. buffer and staleness)
    share: float  # alpha_ij: fraction of the pair constraint drone i must enforce


def project_box_halfspaces(nominal: np.ndarray, rows: np.ndarray, bounds: np.ndarray, cap: np.ndarray,
                           iterations: int = 400, tolerance: float = 1e-9) -> np.ndarray:
    """min ||u - nominal||^2  s.t.  rows @ u >= bounds,  |u_k| <= cap_k   (Hildreth)."""
    nominal = np.asarray(nominal, float)
    n = nominal.size
    eye = np.eye(n)
    a = np.vstack([np.asarray(rows, float).reshape(-1, n), eye, -eye])
    b = np.concatenate([np.asarray(bounds, float).reshape(-1), -np.asarray(cap, float), -np.asarray(cap, float)])
    norms = np.einsum("ij,ij->i", a, a)
    lam = np.zeros(a.shape[0])
    u = nominal.copy()
    for _ in range(iterations):
        change = 0.0
        for k in range(a.shape[0]):
            # u = nominal + a^T lam ; maximise dual coordinate-wise with lam >= 0
            step = (b[k] - a[k] @ u) / norms[k]
            new = max(0.0, lam[k] + step)
            delta = new - lam[k]
            if delta != 0.0:
                u = u + delta * a[k]
                lam[k] = new
                change = max(change, abs(delta))
        if change < tolerance:
            break
    return u


def pair_rows(position: np.ndarray, contracts: list[SeparationContract], gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constraint rows/bounds for drone i plus each pair's current barrier value h_ij."""
    rows, bounds, barrier = [], [], []
    for contract in contracts:
        delta = np.asarray(position, float) - np.asarray(contract.position_ned, float)
        h = float(delta @ delta - contract.radius_m ** 2)
        rows.append(2.0 * delta)
        bounds.append(-contract.share * gamma * h)
        barrier.append(h)
    return np.asarray(rows).reshape(-1, 3), np.asarray(bounds), np.asarray(barrier)


def filter_velocity(nominal: np.ndarray, position: np.ndarray, contracts: list[SeparationContract],
                    gamma: float, speed_cap: float, vertical_cap: float | None = None) -> tuple[np.ndarray, dict]:
    """Drone-side CBF filter. Returns (safe velocity, diagnostics)."""
    nominal = np.asarray(nominal, float).reshape(3)
    cap = np.array([speed_cap, speed_cap, speed_cap if vertical_cap is None else vertical_cap])
    rows, bounds, barrier = pair_rows(position, contracts, gamma)
    if rows.shape[0] == 0:
        safe = np.clip(nominal, -cap, cap)
    else:
        safe = project_box_halfspaces(nominal, rows, bounds, cap)
    active = int(np.sum(rows @ safe <= bounds + 1e-6)) if rows.shape[0] else 0
    return safe, {"cbf_active": active, "cbf_min_h": float(barrier.min()) if barrier.size else np.inf,
                  "cbf_correction": float(np.linalg.norm(safe - nominal))}


def joint_constraint_holds(p_i, u_i, p_j, u_j, radius: float, gamma: float, tolerance: float = 1e-6) -> bool:
    """The un-split pair condition 2(p_i-p_j)^T(u_i-u_j) >= -gamma*h (used by tests and monitoring)."""
    delta = np.asarray(p_i, float) - np.asarray(p_j, float)
    h = delta @ delta - radius ** 2
    return bool(2.0 * delta @ (np.asarray(u_i, float) - np.asarray(u_j, float)) >= -gamma * h - tolerance)
