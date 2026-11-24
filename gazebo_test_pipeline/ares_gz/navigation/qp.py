"""Port of ``projectControl.m``: exact 3-variable Euclidean QP by active sets.

    minimise ||u - request||^2 / 2   subject to   A u <= b,  lower <= u <= upper
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

TOLERANCE = 1e-7


def _rcond(matrix: np.ndarray) -> float:
    """Reciprocal 1-norm condition number, the quantity MATLAB ``rcond`` estimates."""
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return 0.0
    norm = np.linalg.norm(matrix, 1) * np.linalg.norm(inverse, 1)
    if not np.isfinite(norm) or norm == 0.0:
        return 0.0
    return 1.0 / norm


def project_control(request, A, b, lower, upper) -> tuple[np.ndarray, bool]:
    request = np.asarray(request, dtype=float).reshape(3)
    A = np.asarray(A, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(3)
    upper = np.asarray(upper, dtype=float).reshape(3)

    solution = np.minimum(np.maximum(request, lower), upper)
    if np.all(A @ solution <= b + TOLERANCE) and np.all(lower <= upper):
        return solution, True  # projection onto the actuator box is already safe
    if np.any(lower > upper):
        return solution, False
    A = np.vstack((A, np.eye(3), -np.eye(3)))
    b = np.concatenate((b, upper, -lower))
    norms = np.sqrt(np.sum(A * A, axis=1))
    if np.any((norms < 1e-12) & (b < -TOLERANCE)):
        return solution, False
    keep = norms >= 1e-12
    A = A[keep] / norms[keep, None]
    b = b[keep] / norms[keep]
    # MATLAB sort(...,'descend') is stable for ties.
    order = np.argsort(-(A @ request - b), kind="stable")
    A = A[order]
    b = b[order]
    n = A.shape[0]

    def candidate(indices: tuple[int, ...]):
        active = A[list(indices)]
        gram = active @ active.T
        if _rcond(gram) < 1e-11:
            return None
        multiplier = np.linalg.solve(gram, active @ request - b[list(indices)])
        if np.any(multiplier < -TOLERANCE):
            return None
        value = request - active.T @ multiplier
        if np.any(A @ value > b + TOLERANCE):
            return None
        return value

    # Same enumeration order as the MATLAB nested loops: singles, pairs, triples.
    for size in (1, 2, 3):
        for indices in combinations(range(n), size):
            value = candidate(indices)
            if value is not None:
                return value, True
    return solution, False
