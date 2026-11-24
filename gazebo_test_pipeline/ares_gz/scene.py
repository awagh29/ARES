"""The building scene, read from the same file the MATLAB simulation uses.

``scripts/initializeSingleUav.m`` stores ``ObstaclePositions`` as addMesh
polygon coordinates, which are ENU: column 1 is East, column 2 is North (the
loop variables there are misleadingly named north/east). This module mirrors
``ares.navigation.sceneMap``: duplicate rows are removed, the columns are
swapped to NED, and each building becomes an axis-aligned box from the ground
(Down 0) up to Down -height.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import INITIALIZER


@dataclass(frozen=True)
class SceneMap:
    """Same fields as the MATLAB ``map`` struct from ``sceneMap.m``."""

    box_min: np.ndarray  # (K,3) NED
    box_max: np.ndarray  # (K,3) NED
    centers_enu: np.ndarray  # (K,2) unique [East North]
    width_m: float
    height_m: float
    sensor_mount_body_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.4]))
    sensor_to_body_rotation: np.ndarray = field(default_factory=lambda: np.diag([1.0, -1.0, -1.0]))
    max_range_m: float = 7.0
    sensor_directions: np.ndarray = field(default_factory=lambda: sensor_directions())

    @property
    def centers_ned(self) -> np.ndarray:
        return (self.box_min[:, :2] + self.box_max[:, :2]) / 2.0


def sensor_directions() -> np.ndarray:
    """15x716 unit rays in MATLAB order (elevation varies fastest).

    MATLAB: ``[az,el]=meshgrid(linspace(-179,179,716)*pi/180, linspace(-15,15,15)*pi/180)``
    followed by ``(:)``, which stacks the 15-row columns one after another.
    """
    az = np.linspace(-179.0, 179.0, 716) * np.pi / 180.0
    el = np.linspace(-15.0, 15.0, 15) * np.pi / 180.0
    az_grid, el_grid = np.meshgrid(az, el)  # shape (15, 716), like MATLAB meshgrid
    az_flat = az_grid.flatten(order="F")
    el_flat = el_grid.flatten(order="F")
    return np.column_stack((
        np.cos(el_flat) * np.cos(az_flat),
        np.cos(el_flat) * np.sin(az_flat),
        np.sin(el_flat),
    ))


def _matlab_scalar(text: str, name: str) -> float:
    match = re.search(rf"^\s*{name}\s*=\s*([-+0-9.eE]+)\s*;", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot find scalar {name} in initializeSingleUav.m")
    return float(match.group(1))


def read_initializer(path: Path = INITIALIZER) -> tuple[np.ndarray, float, float]:
    """Return (ObstaclePositions as ENU rows, ObstacleWidth, ObstacleHeight)."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"^\s*ObstaclePositions\s*=\s*\[(.*?)\]\s*;", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError("Cannot find ObstaclePositions in initializeSingleUav.m")
    rows = []
    for line in match.group(1).splitlines():
        line = line.split("%", 1)[0].replace(",", " ").replace(";", " ").strip()
        if line:
            rows.append([float(value) for value in line.split()])
    positions = np.array(rows, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"ObstaclePositions must have two columns, got {positions.shape}")
    return positions, _matlab_scalar(text, "ObstacleWidth"), _matlab_scalar(text, "ObstacleHeight")


def unique_rows_stable(rows: np.ndarray) -> np.ndarray:
    """MATLAB ``unique(rows,'rows','stable')``."""
    seen = set()
    keep = []
    for row in rows:
        key = tuple(float(value) for value in row)
        if key not in seen:
            seen.add(key)
            keep.append(row)
    return np.array(keep, dtype=float)


def scene_map(positions_enu: np.ndarray | None = None, width: float | None = None,
              height: float | None = None) -> SceneMap:
    """Port of ``ares.navigation.sceneMap``."""
    if positions_enu is None or width is None or height is None:
        read_positions, read_width, read_height = read_initializer()
        positions_enu = read_positions if positions_enu is None else positions_enu
        width = read_width if width is None else width
        height = read_height if height is None else height
    centers_enu = unique_rows_stable(np.asarray(positions_enu, dtype=float))
    centers_ned = centers_enu[:, [1, 0]]
    count = centers_ned.shape[0]
    box_min = np.column_stack((centers_ned - width / 2.0, -height * np.ones(count)))
    box_max = np.column_stack((centers_ned + width / 2.0, np.zeros(count)))
    return SceneMap(box_min=box_min, box_max=box_max, centers_enu=centers_enu,
                    width_m=float(width), height_m=float(height))
