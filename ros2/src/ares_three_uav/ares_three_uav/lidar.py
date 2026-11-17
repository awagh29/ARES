"""Organized 15-by-716 lidar and ray casts against the building boxes.

Azimuth and elevation use the same ``linspace`` endpoints as
``sectorFeatures``: 716 azimuth samples from -179 to 179 degrees and 15
elevation samples from -15 to 15 degrees. The sensor frame is x-forward,
y-left, z-up. That is the NED body frame after the roll-180 mount, so a
body bearing is the negative of the sensor azimuth.
"""

from __future__ import annotations

import numpy as np

from .mission import (
    AZIMUTH_DEG,
    ELEVATION_DEG,
    SENSOR_MAX_RANGE_M,
    SENSOR_MIN_RANGE_M,
    SENSOR_OFFSET_BODY_NED,
)


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def sensor_rays() -> np.ndarray:
    azimuth = np.deg2rad(AZIMUTH_DEG)
    elevation = np.deg2rad(ELEVATION_DEG)
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation, indexing="ij")
    rays = np.stack((
        np.cos(elevation_grid) * np.cos(azimuth_grid),
        np.cos(elevation_grid) * np.sin(azimuth_grid),
        np.sin(elevation_grid),
    ), axis=-1)
    return rays.reshape(-1, 3)


SENSOR_RAYS = sensor_rays()
# Roll 180 takes NED-body vectors into the FLU sensor frame, and the reverse.
_MOUNT = np.diag([1.0, -1.0, -1.0])


def cast_obstacles(position_ned, roll: float, pitch: float, yaw: float, box_min, box_max) -> np.ndarray:
    """Return world-NED hit points inside the sensor's accepted range."""
    rotation = _rotation(roll, pitch, yaw)
    origin = np.asarray(position_ned, dtype=float) + rotation @ SENSOR_OFFSET_BODY_NED
    directions = SENSOR_RAYS @ _MOUNT.T @ rotation.T
    ranges = _ray_boxes(origin, directions, box_min, box_max)
    valid = (ranges >= SENSOR_MIN_RANGE_M) & (ranges <= SENSOR_MAX_RANGE_M)
    if not np.any(valid):
        return np.zeros((0, 3))
    return origin + directions[valid] * ranges[valid, None]


def _ray_boxes(origin: np.ndarray, directions: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions
    t0 = (box_min[:, None, :] - origin) * inverse
    t1 = (box_max[:, None, :] - origin) * inverse
    near = np.minimum(t0, t1)
    far = np.maximum(t0, t1)
    parallel = np.abs(directions) < 1e-12
    outside = ((origin < box_min[:, None, :]) | (origin > box_max[:, None, :])) & parallel
    near = np.where(parallel, np.where(outside, np.inf, -np.inf), near)
    far = np.where(parallel, np.where(outside, -np.inf, np.inf), far)
    enter = np.max(near, axis=-1)
    leave = np.min(far, axis=-1)
    hit = (leave >= np.maximum(enter, 0.0)) & (leave > 0.0) & (enter <= SENSOR_MAX_RANGE_M)
    distance = np.where(enter > 0.0, enter, np.where(leave > 0.0, 0.0, np.inf))
    distance = np.where(hit, distance, np.inf)
    return np.min(distance, axis=0)
