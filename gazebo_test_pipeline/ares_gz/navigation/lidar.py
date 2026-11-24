"""Ports of ``virtualLidar.m`` and ``MapCorrectionController.measuredSurfaces``.

Point clouds are (10740, 3) sensor-frame arrays in MATLAB order: 15 elevation
rows x 716 azimuth columns, elevation varying fastest; misses are NaN. The
sensor frame is x-forward, y-left, z-up (the NED body after the roll-180 mount).
"""

from __future__ import annotations

import numpy as np

from ..scene import SceneMap
from .geometry import eul2rotm_zyx

ROWS = 15
COLUMNS = 716

# MATLAB: [el,az]=ndgrid(1:15,1:716); bins=ceil(el(:)/5)+3*floor((az(:)-1)*12/716)
_EL, _AZ = np.meshgrid(np.arange(1, ROWS + 1), np.arange(1, COLUMNS + 1), indexing="ij")
_EL_FLAT = _EL.flatten(order="F")
_AZ_FLAT = _AZ.flatten(order="F")
SECTOR_BINS = np.ceil(_EL_FLAT / 5).astype(int) + 3 * np.floor((_AZ_FLAT - 1) * 12 / 716).astype(int)


def virtual_lidar(scene: SceneMap, position, euler_zyx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 15x716 ray cast of the map boxes; returns (cloud, ranges, hits)."""
    rotation = eul2rotm_zyx(euler_zyx)
    origin = np.asarray(position, dtype=float).reshape(3) + rotation @ scene.sensor_mount_body_m
    direction = scene.sensor_directions @ scene.sensor_to_body_rotation.T @ rotation.T
    nearest = np.full(direction.shape[0], np.inf)
    for k in range(scene.box_min.shape[0]):
        near = np.full(nearest.shape, -np.inf)
        far = np.full(nearest.shape, np.inf)
        for axis in range(3):
            parallel = np.abs(direction[:, axis]) < 1e-12
            divisor = direction[:, axis].copy()
            divisor[parallel] = 1.0
            a = (scene.box_min[k, axis] - origin[axis]) / divisor
            b = (scene.box_max[k, axis] - origin[axis]) / divisor
            lower = np.minimum(a, b)
            upper = np.maximum(a, b)
            lower[parallel] = -np.inf
            upper[parallel] = np.inf
            if origin[axis] < scene.box_min[k, axis] or origin[axis] > scene.box_max[k, axis]:
                lower[parallel] = np.inf
            near = np.maximum(near, lower)
            far = np.minimum(far, upper)
        valid = far >= np.maximum(near, 0.0)
        candidate = near.copy()
        candidate[near < 0] = far[near < 0]
        candidate[~valid | (candidate <= 0)] = np.inf
        nearest = np.minimum(nearest, candidate)
    hits = np.isfinite(nearest) & (nearest <= scene.max_range_m)
    ranges = scene.max_range_m * np.ones(nearest.shape)
    ranges[hits] = nearest[hits]
    cloud = np.full((nearest.size, 3), np.nan, dtype=np.float32)
    cloud[hits, :] = (scene.sensor_directions[hits, :] * ranges[hits, None]).astype(np.float32)
    return cloud, ranges, hits


def measured_surfaces(scene: SceneMap, points, position, euler_zyx, geometry_uncertainty_m: float) -> np.ndarray:
    """World-NED points from a real scan that lie outside every mapped box's uncertainty band.

    One nearest return per angular bin (36 bins) is kept. Mapped faces are
    already protected by the box constraints, so points on them are dropped.
    """
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    with np.errstate(invalid="ignore"):
        ranges = np.sqrt(np.sum(xyz * xyz, axis=1))
        valid = np.all(np.isfinite(xyz), axis=1) & (ranges > 0) & (ranges <= scene.max_range_m + 1e-5)
    rotation = eul2rotm_zyx(euler_zyx)
    body = xyz @ scene.sensor_to_body_rotation.T + scene.sensor_mount_body_m
    world = body @ rotation.T + np.asarray(position, dtype=float).reshape(1, 3)
    for k in range(scene.box_min.shape[0]):
        center = (scene.box_min[k] + scene.box_max[k]) / 2.0
        half = (scene.box_max[k] - scene.box_min[k]) / 2.0
        q = np.abs(world - center) - half
        with np.errstate(invalid="ignore"):
            distance = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=1)) + np.minimum(np.max(q, axis=1), 0.0)
            valid = valid & (distance > geometry_uncertainty_m)
    selected = []
    for k in range(1, 37):
        ids = np.flatnonzero(valid & (SECTOR_BINS == k))
        if ids.size == 0:
            continue
        selected.append(ids[int(np.argmin(ranges[ids]))])
    return world[selected, :] if selected else np.zeros((0, 3))


def gazebo_cloud_to_matlab_order(xyz_row_major: np.ndarray, flip_vertical: bool = False,
                                 flip_horizontal: bool = False, max_range_m: float = 7.0) -> np.ndarray:
    """Reorder a Gazebo 15x716 organized cloud (azimuth fastest) into MATLAB order (elevation fastest).

    Gazebo's gpu_lidar packs one row per vertical angle, each row sweeping the
    horizontal angles. ``scripts/diagnose.py`` measures whether rows/columns run
    in the opposite direction to MATLAB's and stores the flips in
    ``config/vehicle_calibration.json``. Missing or out-of-range returns become NaN.
    """
    grid = np.asarray(xyz_row_major, dtype=np.float32).reshape(ROWS, COLUMNS, 3)
    if flip_vertical:
        grid = grid[::-1]
    if flip_horizontal:
        grid = grid[:, ::-1]
    cloud = grid.transpose(1, 0, 2).reshape(-1, 3).copy()  # column (azimuth) outer, elevation inner
    with np.errstate(invalid="ignore"):
        ranges = np.sqrt(np.sum(cloud.astype(float) ** 2, axis=1))
        invalid = ~np.all(np.isfinite(cloud), axis=1) | (ranges > max_range_m + 1e-3) | (ranges <= 0)
    cloud[invalid] = np.nan
    return cloud
