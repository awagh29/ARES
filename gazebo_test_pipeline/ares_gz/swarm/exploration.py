"""Frontier detection and hub task allocation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .mapping import SharedMap, dilate


@dataclass
class Task:
    id: str
    kind: str  # explore | confirm | rendezvous | hold
    target_ned: np.ndarray
    gain: float = 1.0
    assigned_to: str | None = None
    status: str = "open"  # open | assigned | done | failed
    created_s: float = 0.0
    assigned_s: float | None = None
    note: str = ""
    history: list = field(default_factory=list)


def safe_free_mask(shared: SharedMap, clearance_m: float) -> np.ndarray:
    """Free cells at least ``clearance_m`` from every mapped obstacle cell."""
    free, occupied, _ = shared.states()
    return free & ~dilate(occupied, clearance_m / shared.grid.resolution)


def frontier_targets(shared: SharedMap, clearance_m: float = 2.0, min_cluster: int = 4,
                     blocked: list | None = None, blocked_radius_m: float = 3.0, tile_m: float = 5.0) -> list[tuple[np.ndarray, int]]:
    """[(target N/E, frontier cells)]: free cells next to unknown space, grouped in tiles, one safe target each."""
    free, occupied, unknown = shared.states()
    cells_per_m = 1.0 / shared.grid.resolution
    # Targets sit in known-free space: clear of mapped obstacles and 1.5 m back from the unknown,
    # so an unmapped wall cannot be right at the goal (the 7 m LiDAR still sees past the frontier).
    safe = safe_free_mask(shared, clearance_m) & ~dilate(unknown, 1.5 * cells_per_m)
    # Unknown cells hugging a mapped wall are usually the inside of a building: not worth exploring.
    unknown = unknown & ~dilate(occupied, 2)
    near_unknown = np.zeros_like(unknown)
    near_unknown[1:, :] |= unknown[:-1, :]
    near_unknown[:-1, :] |= unknown[1:, :]
    near_unknown[:, 1:] |= unknown[:, :-1]
    near_unknown[:, :-1] |= unknown[:, 1:]
    # Search targets: places no victim camera has looked at yet (mapped or not), plus the
    # classic LiDAR frontier. Cells next to mapped walls are excluded (building interiors).
    unsearched = ~shared.searched & ~dilate(occupied, 2) & (free | unknown)
    frontier = (free & near_unknown) | unsearched
    grid = shared.grid
    # Group frontier cells into square tiles (a ring-shaped frontier becomes several local goals).
    tile = max(1, int(round(tile_m / grid.resolution)))
    groups: dict = {}
    for n, e in zip(*np.nonzero(frontier)):
        groups.setdefault((n // tile, e // tile), []).append((n, e))
    safe_cells = np.argwhere(safe)
    targets = []
    if safe_cells.size == 0:
        return targets
    for cells in groups.values():
        if len(cells) < min_cluster:
            continue
        cells_arr = np.array(cells)
        # The frontier cell nearest the group's mean is a point *on* the frontier.
        anchor = cells_arr[int(np.argmin(np.linalg.norm(cells_arr - cells_arr.mean(axis=0), axis=1)))]
        distance = np.linalg.norm(safe_cells - anchor, axis=1)
        if distance.min() * grid.resolution > 4.0:
            continue
        point = grid.center(*safe_cells[int(np.argmin(distance))])
        if blocked and any(np.linalg.norm(point - np.asarray(b)[:2]) < blocked_radius_m for b in blocked):
            continue
        targets.append((point, len(cells)))
    return targets


def allocate(drone_positions: dict, tasks: list[Task], min_target_separation_m: float, time_s: float) -> dict:
    """Greedy auction: repeatedly give the best (drone, task) pair; returns {drone: task}.

    utility = gain / (1 + distance); confirm tasks get a large gain so they are
    served first (and ignore the spacing rule). Targets of different drones are kept apart so the drones do
    not crowd one spot (the CBF would otherwise make them queue).
    """
    free_drones = dict(drone_positions)
    open_tasks = [task for task in tasks if task.status == "open"]
    taken_targets = [task.target_ned for task in tasks if task.status == "assigned"]
    result = {}
    while free_drones and open_tasks:
        best = None
        for drone, position in free_drones.items():
            for task in open_tasks:
                if task.kind == "explore" and any(
                        np.linalg.norm(task.target_ned[:2] - t[:2]) < min_target_separation_m for t in taken_targets):
                    continue
                distance = float(np.linalg.norm(task.target_ned[:2] - np.asarray(position)[:2]))
                utility = task.gain / (1.0 + distance)
                if best is None or utility > best[0]:
                    best = (utility, drone, task)
        if best is None:
            break
        _, drone, task = best
        task.status, task.assigned_to, task.assigned_s = "assigned", drone, time_s
        task.history.append((round(time_s, 2), f"assigned to {drone}"))
        result[drone] = task
        taken_targets.append(task.target_ned)
        del free_drones[drone]
        open_tasks.remove(task)
    return result


def assign_slots(drone_positions: dict, slots: list) -> dict:
    """Rendezvous: the slot permutation with the least total distance."""
    drones = list(drone_positions)
    best, best_cost = None, np.inf
    for perm in itertools.permutations(range(len(slots)), len(drones)):
        cost = sum(np.linalg.norm(np.asarray(slots[k])[:2] - np.asarray(drone_positions[d])[:2])
                   for d, k in zip(drones, perm))
        if cost < best_cost:
            best, best_cost = perm, cost
    return {drone: np.asarray(slots[k], float) for drone, k in zip(drones, best or [])}
