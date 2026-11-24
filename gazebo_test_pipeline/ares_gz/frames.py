"""NED/FRD (MATLAB solution) <-> ENU/FLU (Gazebo) conversions.

Gazebo world: x = East, y = North, z = Up. Gazebo body: x forward, y left, z up.
MATLAB world: North, East, Down. MATLAB body: x forward, y right, z down.
"""

from __future__ import annotations

import math

import numpy as np

from .navigation.geometry import eul2rotm_zyx

# Swaps ENU <-> NED (its own inverse).
ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# Swaps FLU <-> FRD body axes (its own inverse).
FLU_FRD = np.diag([1.0, -1.0, -1.0])


def ned_to_enu(vector_ned) -> np.ndarray:
    return ENU_NED @ np.asarray(vector_ned, dtype=float).reshape(3)


def enu_to_ned(vector_enu) -> np.ndarray:
    return ENU_NED @ np.asarray(vector_enu, dtype=float).reshape(3)


def quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotm_to_eul_zyx(rotation: np.ndarray) -> np.ndarray:
    """Inverse of ``eul2rotm(...,'ZYX')``: returns [yaw, pitch, roll]."""
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    return np.array([yaw, pitch, roll])


def gazebo_orientation_to_ned_euler(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Gazebo ENU/FLU body quaternion -> MATLAB NED/FRD [yaw pitch roll]."""
    rotation_enu_flu = quaternion_to_matrix(w, x, y, z)
    rotation_ned_frd = ENU_NED @ rotation_enu_flu @ FLU_FRD
    return rotm_to_eul_zyx(rotation_ned_frd)


def ned_euler_to_gazebo_rpy(euler_zyx) -> tuple[float, float, float]:
    """MATLAB NED/FRD [yaw pitch roll] -> Gazebo ENU/FLU (roll, pitch, yaw) for an SDF pose."""
    rotation = ENU_NED @ eul2rotm_zyx(euler_zyx) @ FLU_FRD
    yaw, pitch, roll = rotm_to_eul_zyx(rotation)
    return float(roll), float(pitch), float(yaw)


def matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale)
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return ((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale)
    if rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return ((rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale)
    scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return ((rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale)
