"""Hub-side shared occupancy map at the flight altitude.

Drones send the horizontal ring of each *measured* scan (716 rays at 0 deg
elevation, world frame). The hub fuses rays into a 2-D log-odds grid
(0.5 m cells over the trained domain N/E -5..40 m). Each drone's contribution
is kept separately so the trust model can compare one drone's scan with what
the *other* drones mapped (peer evidence). Only trusted scans are fused;
reconstructed scans never are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene import SceneMap

RING_ELEVATION_INDEX = 7  # linspace(-15, 15, 15)[7] == 0 deg
HIT_LOGODDS = 0.9
MISS_LOGODDS = -0.7
CLAMP = 4.0
FREE_BELOW = -0.5
OCCUPIED_ABOVE = 0.5


@dataclass(frozen=True)
class Ring:
    """One horizontal scan ring in the world frame."""

    origin_ned: np.ndarray  # (3,) sensor origin
    endpoints_ned: np.ndarray  # (R, 3) hit point, or max-range point for a miss
    hits: np.ndarray  # (R,) bool


def ring_from_scan(points_matlab_order: np.ndarray, position_ned, euler_zyx, scene: SceneMap) -> Ring:
    """Measured sensor-frame cloud (MATLAB order, NaN = miss) -> world-frame horizontal ring."""
    from ..navigation.geometry import eul2rotm_zyx

    rotation = eul2rotm_zyx(euler_zyx)
    index = np.arange(716) * 15 + RING_ELEVATION_INDEX
    cloud = np.asarray(points_matlab_order, float)[index]
    hits = np.all(np.isfinite(cloud), axis=1)
    sensor = scene.sensor_directions[index] * scene.max_range_m
    sensor[hits] = cloud[hits]
    body = sensor @ scene.sensor_to_body_rotation.T + scene.sensor_mount_body_m
    world = body @ rotation.T + np.asarray(position_ned, float)
    origin = np.asarray(position_ned, float) + rotation @ scene.sensor_mount_body_m
    return Ring(origin, world, hits)


class Grid:
    def __init__(self, bounds=((-5.0, 40.0), (-5.0, 40.0)), resolution: float = 0.5):
        self.resolution = resolution
        self.north = np.arange(bounds[0][0], bounds[0][1] + 1e-9, resolution)
        self.east = np.arange(bounds[1][0], bounds[1][1] + 1e-9, resolution)
        self.shape = (self.north.size, self.east.size)

    def index(self, points_ne: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points_ne, float).reshape(-1, 2)
        n = np.rint((points[:, 0] - self.north[0]) / self.resolution).astype(int)
        e = np.rint((points[:, 1] - self.east[0]) / self.resolution).astype(int)
        valid = (n >= 0) & (n < self.shape[0]) & (e >= 0) & (e < self.shape[1])
        return n, e, valid

    def center(self, n: int, e: int) -> np.ndarray:
        return np.array([self.north[n], self.east[e]])


class SharedMap:
    def __init__(self, drones: list[str], altitude_down: float, grid: Grid | None = None):
        self.grid = grid or Grid()
        self.altitude_down = altitude_down
        self.layers = {drone: np.zeros(self.grid.shape) for drone in drones}
        self.fused_scans = {drone: 0 for drone in drones}
        self.searched = np.zeros(self.grid.shape, bool)  # seen by a victim camera
        n, e = np.meshgrid(self.grid.north, self.grid.east, indexing="ij")
        self._cells_ne = np.stack((n, e), axis=-1)

    def mark_searched(self, position_ned, range_m: float = 5.0, samples: int = 12) -> None:
        """Cells inside the camera footprint, with no mapped obstacle between drone and cell."""
        p = np.asarray(position_ned, float)[:2]
        n, e, valid = self.grid.index(p[None, :])
        reach = int(np.ceil(range_m / self.grid.resolution))
        if not valid[0]:
            return
        n0, e0 = int(n[0]), int(e[0])
        rows = slice(max(0, n0 - reach), min(self.grid.shape[0], n0 + reach + 1))
        cols = slice(max(0, e0 - reach), min(self.grid.shape[1], e0 + reach + 1))
        cells = self._cells_ne[rows, cols].reshape(-1, 2)
        near = np.linalg.norm(cells - p, axis=1) <= range_m
        _, occupied, _ = self.states()
        visible = near.copy()
        for fraction in np.linspace(0.15, 0.95, samples):
            probe = p + fraction * (cells - p)
            pn, pe, ok = self.grid.index(probe)
            blocked = np.zeros(len(cells), bool)
            blocked[ok] = occupied[pn[ok], pe[ok]]
            visible &= ~blocked
        block = self.searched[rows, cols].reshape(-1)
        block |= visible
        self.searched[rows, cols] = block.reshape(self.searched[rows, cols].shape)

    # ---- fusion ------------------------------------------------------------
    def ray_cells(self, ring: Ring, step_m: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
        """(free cell flat indices, occupied cell flat indices) observed by one ring."""
        origin = ring.origin_ned[:2]
        delta = ring.endpoints_ned[:, :2] - origin
        length = np.linalg.norm(delta, axis=1)
        reach = np.where(ring.hits, np.maximum(length - 0.3, 0.0), np.minimum(length, 6.5))
        samples = np.arange(0.0, 7.0, step_m)
        along = samples[None, :] <= reach[:, None]
        unit = delta / np.maximum(length, 1e-9)[:, None]
        points = origin + unit[:, None, :] * samples[None, :, None]
        n, e, valid = self.grid.index(points[along])
        free = np.unique(np.ravel_multi_index((n[valid], e[valid]), self.grid.shape))
        in_band = np.abs(ring.endpoints_ned[:, 2] - self.altitude_down) < 0.65
        hit_points = ring.endpoints_ned[ring.hits & in_band, :2]
        n, e, valid = self.grid.index(hit_points)
        occupied = np.unique(np.ravel_multi_index((n[valid], e[valid]), self.grid.shape))
        free = np.setdiff1d(free, occupied)
        return free, occupied

    def fuse(self, drone: str, ring: Ring) -> None:
        free, occupied = self.ray_cells(ring)
        layer = self.layers[drone].reshape(-1)
        layer[free] = np.clip(layer[free] + MISS_LOGODDS, -CLAMP, CLAMP)
        layer[occupied] = np.clip(layer[occupied] + HIT_LOGODDS, -CLAMP, CLAMP)
        self.fused_scans[drone] += 1

    # ---- queries -----------------------------------------------------------
    def logodds(self, exclude: str | None = None) -> np.ndarray:
        total = np.zeros(self.grid.shape)
        for drone, layer in self.layers.items():
            if drone != exclude:
                total += layer
        return np.clip(total, -CLAMP, CLAMP)

    def states(self, exclude: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(free, occupied, unknown) boolean grids."""
        value = self.logodds(exclude)
        free = value < FREE_BELOW
        occupied = value > OCCUPIED_ABOVE
        return free, occupied, ~(free | occupied)

    def peer_contradiction(self, drone: str, ring: Ring) -> tuple[float, int]:
        """Share of this ring's observations that contradict the other drones' map, and how many were checked."""
        free, occupied = self.ray_cells(ring)
        others_free, others_occupied, _ = self.states(exclude=drone)
        others_free = others_free.reshape(-1)
        others_occupied = others_occupied.reshape(-1)
        checked = int(np.count_nonzero(others_free[free] | others_occupied[free])
                      + np.count_nonzero(others_free[occupied] | others_occupied[occupied]))
        if checked == 0:
            return 0.0, 0
        contradictions = int(np.count_nonzero(others_occupied[free]) + np.count_nonzero(others_free[occupied]))
        return contradictions / checked, checked

    def search_coverage(self, scene: SceneMap) -> float:
        """Share of the truly free cells a victim camera has looked at."""
        truth = true_free_cells(self.grid, scene)
        return float(np.count_nonzero(self.searched & truth) / max(np.count_nonzero(truth), 1))

    def coverage(self, scene: SceneMap) -> float:
        """Share of the truly free cells (outside every building) that the hub has mapped."""
        free, occupied, _ = self.states()
        truth = true_free_cells(self.grid, scene)
        return float(np.count_nonzero((free | occupied) & truth) / max(np.count_nonzero(truth), 1))


def true_free_cells(grid: Grid, scene: SceneMap) -> np.ndarray:
    n, e = np.meshgrid(grid.north, grid.east, indexing="ij")
    inside = np.zeros(grid.shape, bool)
    for k in range(scene.box_min.shape[0]):
        inside |= ((n >= scene.box_min[k, 0]) & (n <= scene.box_max[k, 0])
                   & (e >= scene.box_min[k, 1]) & (e <= scene.box_max[k, 1]))
    return ~inside


def dilate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Binary dilation by a disc (no SciPy needed)."""
    out = mask.copy()
    reach = int(np.ceil(radius_cells))
    rows, cols = mask.shape
    for dn in range(-reach, reach + 1):
        for de in range(-reach, reach + 1):
            if dn * dn + de * de > radius_cells * radius_cells or (dn == 0 and de == 0):
                continue
            shifted = np.zeros_like(mask)
            src_n = slice(max(0, -dn), rows - max(0, dn))
            dst_n = slice(max(0, dn), rows - max(0, -dn))
            src_e = slice(max(0, -de), cols - max(0, de))
            dst_e = slice(max(0, de), cols - max(0, -de))
            shifted[dst_n, dst_e] = mask[src_n, src_e]
            out |= shifted
    return out
