"""Ports of ``mapClearance.m``, ``segmentFree.m`` and MATLAB ``eul2rotm(...,'ZYX')``."""

from __future__ import annotations

import math

import numpy as np

EPS = np.finfo(float).eps  # MATLAB eps


def eul2rotm_zyx(euler_zyx) -> np.ndarray:
    """MATLAB ``eul2rotm([yaw pitch roll],'ZYX')`` = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    yaw, pitch, roll = (float(value) for value in np.asarray(euler_zyx, dtype=float).reshape(3))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _row_norm(matrix: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(matrix * matrix, axis=1))


def map_clearance(position, box_min: np.ndarray, box_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``mapClearance``: signed center-to-box distances and outward unit normals."""
    p = np.asarray(position, dtype=float).reshape(1, 3)
    center = (box_min + box_max) / 2.0
    half = (box_max - box_min) / 2.0
    q = np.abs(p - center) - half
    distance = _row_norm(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=1), 0.0)
    closest = np.minimum(np.maximum(p, box_min), box_max)
    delta = p - closest
    normal = delta / np.maximum(_row_norm(delta), EPS)[:, None]
    for k in np.flatnonzero(distance <= 0):
        axis = int(np.argmax(q[k]))  # first maximum, like MATLAB max
        normal[k, :] = 0.0
        signum = float(np.sign(p[0, axis] - center[k, axis]))
        if signum == 0.0:
            signum = 1.0
        normal[k, axis] = signum
    return distance, normal


def segment_free(first, last, box_min: np.ndarray, box_max: np.ndarray, padding: float,
                 bounds: np.ndarray) -> bool:
    """``segmentFree``: exact segment/slab test against expanded boxes."""
    first = np.asarray(first, dtype=float).reshape(3)
    last = np.asarray(last, dtype=float).reshape(3)
    lower_bound = bounds[:, 0]
    upper_bound = bounds[:, 1]
    if np.any((first < lower_bound) | (first > upper_bound) | (last < lower_bound) | (last > upper_bound)):
        return False
    lo = box_min - padding
    hi = box_max + padding
    direction = last - first
    count = lo.shape[0]
    tmin = np.zeros(count)
    tmax = np.ones(count)
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            outside = (first[axis] < lo[:, axis]) | (first[axis] > hi[:, axis])
            tmin[outside] = np.inf
        else:
            a = (lo[:, axis] - first[axis]) / direction[axis]
            b = (hi[:, axis] - first[axis]) / direction[axis]
            tmin = np.maximum(tmin, np.minimum(a, b))
            tmax = np.minimum(tmax, np.maximum(a, b))
    return not bool(np.any(tmin <= tmax))
