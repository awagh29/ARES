"""3DVFH+ direction used by the UAV Toolbox Obstacle Avoidance block.

MATLAB documents this block as Baumann's 3DVFH+ with the histogram parameters
saved in ``ares_single_uav_top.slx``. Points from the most recent ``MaxAge``
scans are kept in the world frame. A cell is occupied when it holds at least
``HistogramThreshold`` points, then expanded by the angle subtended by the
vehicle radius plus the safety distance. The commanded direction is the free
cell inside the current field of view with the lowest weighted cost. With no
occupied cell, the direction is the target direction itself.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .mission import (
    CURRENT_DIRECTION_WEIGHT,
    HISTOGRAM_RESOLUTION_DEG,
    HISTOGRAM_THRESHOLD,
    HORIZONTAL_FOV_DEG,
    MAX_AGE,
    PREVIOUS_DIRECTION_WEIGHT,
    SAFETY_DISTANCE_M,
    TARGET_DIRECTION_WEIGHT,
    VEHICLE_RADIUS_M,
    VERTICAL_FOV_DEG,
    WINDOW_SIZE,
)


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


class VFH3D:
    def __init__(self):
        self.resolution = HISTOGRAM_RESOLUTION_DEG
        self.azimuth_count = int(round(360.0 / self.resolution))
        self.elevation_count = int(round(180.0 / self.resolution))
        azimuth = np.deg2rad(-180.0 + self.resolution / 2.0 + self.resolution * np.arange(self.azimuth_count))
        elevation = np.deg2rad(-90.0 + self.resolution / 2.0 + self.resolution * np.arange(self.elevation_count))
        elevation_grid, azimuth_grid = np.meshgrid(elevation, azimuth, indexing="ij")
        self.cell_direction = np.stack((
            np.cos(elevation_grid) * np.cos(azimuth_grid),
            np.cos(elevation_grid) * np.sin(azimuth_grid),
            -np.sin(elevation_grid),
        ), axis=-1)
        self.memory: deque[np.ndarray] = deque(maxlen=MAX_AGE + 1)
        self.previous = np.array([1.0, 0.0, 0.0])

    def step(self, position_ned, roll: float, pitch: float, yaw: float, hits_ned, target_ned):
        position = np.asarray(position_ned, dtype=float)
        target = np.asarray(target_ned, dtype=float)
        hits = np.asarray(hits_ned, dtype=float).reshape(-1, 3)
        self.memory.append(hits)
        remembered = np.concatenate(list(self.memory)) if self.memory else np.zeros((0, 3))
        target_direction = self._unit(target - position, fallback=self.previous)
        current = _rotation(roll, pitch, yaw) @ np.array([1.0, 0.0, 0.0])
        if remembered.size == 0:
            direction = self._clamp_elevation(target_direction)
            self.previous = direction
            return direction, math.atan2(direction[1], direction[0]), 0

        relative = remembered - position
        distance = np.linalg.norm(relative, axis=1)
        keep = distance > 1e-6
        relative, distance = relative[keep], distance[keep]
        close = bool(np.any(distance <= VEHICLE_RADIUS_M + SAFETY_DISTANCE_M))
        azimuth = np.arctan2(relative[:, 1], relative[:, 0])
        horizontal = np.hypot(relative[:, 0], relative[:, 1])
        elevation = np.arctan2(-relative[:, 2], horizontal)
        azimuth_index = np.floor((np.rad2deg(azimuth) + 180.0) / self.resolution).astype(int) % self.azimuth_count
        elevation_index = np.clip(
            np.floor((np.rad2deg(elevation) + 90.0) / self.resolution).astype(int),
            0, self.elevation_count - 1,
        )
        counts = np.zeros((self.elevation_count, self.azimuth_count), dtype=int)
        nearest = np.full(counts.shape, np.inf)
        np.add.at(counts, (elevation_index, azimuth_index), 1)
        np.minimum.at(nearest, (elevation_index, azimuth_index), distance)
        occupied = counts >= HISTOGRAM_THRESHOLD
        blocked = self._inflate(occupied, nearest)
        if not np.any(occupied):
            direction = self._clamp_elevation(target_direction)
            self.previous = direction
            return direction, math.atan2(direction[1], direction[0]), 2 if close else 0

        free = ~blocked
        half_window = WINDOW_SIZE // 2
        if half_window:
            free = self._opening(free, half_window)
        candidates = free & self._in_fov(roll, pitch, yaw)
        if not np.any(candidates):
            status = 3 if close else 1
            return self.previous.copy(), math.atan2(self.previous[1], self.previous[0]), status
        cells = self.cell_direction[candidates]
        cost = (
            TARGET_DIRECTION_WEIGHT * self._angle(cells, target_direction)
            + CURRENT_DIRECTION_WEIGHT * self._angle(cells, current)
            + PREVIOUS_DIRECTION_WEIGHT * self._angle(cells, self.previous)
        )
        direction = self._clamp_elevation(cells[int(np.argmin(cost))])
        self.previous = direction
        return direction, math.atan2(direction[1], direction[0]), 2 if close else 0

    def _inflate(self, occupied: np.ndarray, nearest: np.ndarray) -> np.ndarray:
        blocked = occupied.copy()
        margin = VEHICLE_RADIUS_M + SAFETY_DISTANCE_M
        angle = np.degrees(np.arcsin(np.clip(margin / np.maximum(nearest, 1e-3), 0.0, 1.0)))
        radius = np.ceil(np.where(occupied, angle / self.resolution, 0.0)).astype(int)
        rows, columns = np.nonzero(occupied)
        for row, column, cells in zip(rows, columns, radius[rows, columns]):
            elevation = slice(max(0, int(row) - int(cells)), min(self.elevation_count, int(row) + int(cells) + 1))
            left = int(column) - int(cells)
            right = int(column) + int(cells)
            if left >= 0 and right < self.azimuth_count:
                blocked[elevation, left:right + 1] = True
            else:
                columns = np.arange(left, right + 1) % self.azimuth_count
                blocked[np.ix_(np.arange(elevation.start, elevation.stop), columns)] = True
        return blocked

    def _opening(self, free: np.ndarray, half_window: int) -> np.ndarray:
        opened = np.zeros_like(free)
        rows, columns = np.nonzero(free)
        for row, column in zip(rows, columns):
            window = free[
                max(0, row - half_window):row + half_window + 1,
                :,
            ]
            azimuth = np.arange(column - half_window, column + half_window + 1) % self.azimuth_count
            if window[:, azimuth].all() if window.ndim == 2 else False:
                opened[row, column] = True
        return opened

    def _in_fov(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        rotation = _rotation(roll, pitch, yaw)
        body = self.cell_direction @ rotation
        sensor = body * np.array([1.0, -1.0, -1.0])
        azimuth = np.rad2deg(np.arctan2(sensor[..., 1], sensor[..., 0]))
        elevation = np.rad2deg(np.arctan2(sensor[..., 2], np.hypot(sensor[..., 0], sensor[..., 1])))
        return (
            (azimuth >= HORIZONTAL_FOV_DEG[0]) & (azimuth <= HORIZONTAL_FOV_DEG[1])
            & (elevation >= VERTICAL_FOV_DEG[0]) & (elevation <= VERTICAL_FOV_DEG[1])
        )

    @staticmethod
    def _angle(directions: np.ndarray, reference: np.ndarray) -> np.ndarray:
        cosine = np.clip(directions @ reference, -1.0, 1.0)
        return np.arccos(cosine)

    @staticmethod
    def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            return fallback.copy()
        return vector / length

    @staticmethod
    def _clamp_elevation(direction: np.ndarray) -> np.ndarray:
        horizontal = math.hypot(float(direction[0]), float(direction[1]))
        elevation = math.atan2(-float(direction[2]), horizontal)
        limit = math.radians(VERTICAL_FOV_DEG[1])
        elevation = min(limit, max(-limit, elevation))
        heading = math.atan2(float(direction[1]), float(direction[0]))
        return np.array((
            math.cos(elevation) * math.cos(heading),
            math.cos(elevation) * math.sin(heading),
            -math.sin(elevation),
        ))
