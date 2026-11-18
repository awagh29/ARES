"""NED guidance attitude to an ENU, x-forward body pose for Gazebo."""

from __future__ import annotations

import math

import numpy as np


def ned_to_enu(north: float, east: float, down: float) -> tuple[float, float, float]:
    return float(east), float(north), float(-down)


def enu_orientation(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Return ENU roll, pitch, yaw and a wxyz quaternion for the FLU body."""
    rotation = _enu_flu(roll, pitch, yaw)
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-8:
        enu_roll = math.atan2(rotation[2, 1], rotation[2, 2])
        enu_pitch = math.atan2(-rotation[2, 0], sy)
        enu_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        enu_roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        enu_pitch = math.atan2(-rotation[2, 0], sy)
        enu_yaw = 0.0
    return enu_roll, enu_pitch, enu_yaw, _quaternion(rotation)


def _enu_flu(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    body = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    # NED body to FLU sensor/body is roll 180, and NED world to ENU swaps axes.
    mounted = body @ np.diag([1.0, -1.0, -1.0])
    enu_from_ned = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    return enu_from_ned @ mounted


def _quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    return w, x, y, z
