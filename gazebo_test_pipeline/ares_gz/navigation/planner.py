"""Port of ``planPath.m``: 26-connected 3-D A*, clearance-weighted cost, checked shortcuts.

Node numbering, neighbour order and the binary heap follow the MATLAB code
exactly (column-major ``ndgrid`` order), so ties break the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..scene import SceneMap
from .geometry import map_clearance, segment_free
from .options import NavigationOptions


@dataclass(frozen=True)
class Plan:
    WaypointsNedM: np.ndarray
    RawWaypointsNedM: np.ndarray
    LengthM: float
    ExpandedNodes: int
    SafetyRadiusM: float
    PlanningPaddingM: float

    def to_dict(self) -> dict:
        return {
            "WaypointsNedM": self.WaypointsNedM.tolist(),
            "RawWaypointsNedM": self.RawWaypointsNedM.tolist(),
            "LengthM": self.LengthM,
            "ExpandedNodes": self.ExpandedNodes,
            "SafetyRadiusM": self.SafetyRadiusM,
            "PlanningPaddingM": self.PlanningPaddingM,
        }


def _colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB ``start:step:stop``."""
    count = int(math.floor((stop - start) / step + 1e-10)) + 1
    return start + step * np.arange(count)


def plan_path(scene: SceneMap, options: NavigationOptions) -> Plan:
    o = options
    box_min, box_max = scene.box_min, scene.box_max
    start = o.start_state[0:3]
    goal = o.goal_state[0:3]
    bounds = o.bounds
    radius = o.safety_radius_m
    padding = radius + o.PlanningExtraMarginM

    def free(first, last) -> bool:
        return segment_free(first, last, box_min, box_max, padding, bounds)

    def edge_cost(first: np.ndarray, last: np.ndarray) -> float:
        length = float(np.linalg.norm(last - first))
        samples = max(2, math.ceil(length / (o.GridResolutionM / 2)) + 1)
        total = 0.0
        for index in range(samples):
            p = first + (last - first) * index / (samples - 1)
            distance, _ = map_clearance(p, box_min, box_max)
            clearance = max(0.0, float(np.min(distance)) - radius)
            total += 1 + o.ClearanceWeight / (1 + clearance) ** 2
        return length * (total / samples)

    for point in (start, goal):
        if not free(point, point):
            raise ValueError("Start/goal is outside bounds or inside a planning-padded obstacle.")

    if free(start, goal):
        route = np.vstack((start, goal))
        expanded = 0
    else:
        step = o.GridResolutionM
        axes = [_colon(bounds[k, 0], step, bounds[k, 1]) for k in range(3)]
        shape = tuple(axis.size for axis in axes)
        count = int(np.prod(shape))
        if count > 1_000_000:
            raise ValueError("Grid exceeds one million cells; increase GridResolutionM or tighten bounds.")
        north, east, down = np.meshgrid(*axes, indexing="ij")
        nodes = np.column_stack((north.flatten(order="F"), east.flatten(order="F"), down.flatten(order="F")))
        occupied = np.zeros(count, dtype=bool)
        for k in range(box_min.shape[0]):
            occupied |= np.all((nodes >= box_min[k] - padding) & (nodes <= box_max[k] + padding), axis=1)

        def anchor(point: np.ndarray) -> int:
            squared = np.sum((nodes - point) ** 2, axis=1)
            squared[occupied] = np.inf
            order = np.argsort(squared, kind="stable")
            for candidate in order[: min(200, order.size)]:
                if np.isfinite(squared[candidate]) and free(point, nodes[candidate]):
                    return int(candidate)
            raise ValueError("No collision-free grid connection for start/goal. Use a finer grid.")

        source = anchor(start)
        target = anchor(goal)
        cost = np.full(count, np.inf)
        parent = np.full(count, -1, dtype=np.int64)
        closed = np.zeros(count, dtype=bool)
        cost[source] = 0.0
        expanded = 0

        # 1-based binary heap, a copy of the MATLAB push/pop helpers.
        heap_ids = [0] * (max(1024, count) + 1)
        heap_keys = [math.inf] * (max(1024, count) + 1)
        heap_size = 0

        def push(node: int, key: float) -> None:
            nonlocal heap_size
            heap_size += 1
            if heap_size >= len(heap_ids):
                heap_ids.extend([0] * heap_size)
                heap_keys.extend([math.inf] * heap_size)
            slot = heap_size
            while slot > 1:
                up = slot // 2
                if heap_keys[up] <= key:
                    break
                heap_ids[slot] = heap_ids[up]
                heap_keys[slot] = heap_keys[up]
                slot = up
            heap_ids[slot] = node
            heap_keys[slot] = key

        def pop() -> int:
            nonlocal heap_size
            node = heap_ids[1]
            last_id = heap_ids[heap_size]
            key = heap_keys[heap_size]
            heap_size -= 1
            slot = 1
            while 2 * slot <= heap_size:
                child = 2 * slot
                if child < heap_size and heap_keys[child + 1] < heap_keys[child]:
                    child += 1
                if key <= heap_keys[child]:
                    break
                heap_ids[slot] = heap_ids[child]
                heap_keys[slot] = heap_keys[child]
                slot = child
            heap_ids[slot] = last_id
            heap_keys[slot] = key
            return node

        push(source, float(np.linalg.norm(nodes[source] - nodes[target])))
        # ndgrid(-1:1,-1:1,-1:1) flattened column-major; drop the zero offset.
        a, b, c = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
        offsets = np.column_stack((a.flatten(order="F"), b.flatten(order="F"), c.flatten(order="F")))
        offsets = offsets[~np.all(offsets == 0, axis=1)]
        n1, n2, n3 = shape
        while heap_size > 0:
            current = pop()
            if closed[current]:
                continue
            closed[current] = True
            expanded += 1
            if current == target:
                break
            i = current % n1
            j = (current // n1) % n2
            k = current // (n1 * n2)
            subs = np.array([i, j, k]) + offsets
            valid = np.all((subs >= 0) & (subs < np.array(shape)), axis=1)
            subs = subs[valid]
            neighbors = subs[:, 0] + n1 * subs[:, 1] + n1 * n2 * subs[:, 2]
            neighbors = neighbors[~occupied[neighbors] & ~closed[neighbors]]
            for nxt in neighbors:
                nxt = int(nxt)
                # Checking the full edge prevents diagonal corner cutting.
                if not free(nodes[current], nodes[nxt]):
                    continue
                trial = cost[current] + edge_cost(nodes[current], nodes[nxt])
                if trial < cost[nxt]:
                    cost[nxt] = trial
                    parent[nxt] = current
                    push(nxt, trial + float(np.linalg.norm(nodes[nxt] - nodes[target])))
        if not closed[target]:
            raise ValueError("No route in the configured 3-D grid. Change bounds/resolution or mission.")
        ids = [target]
        while ids[0] != source:
            ids.insert(0, int(parent[ids[0]]))
        route = np.vstack((start, nodes[ids], goal))

    keep = np.concatenate(([True], np.sqrt(np.sum(np.diff(route, axis=0) ** 2, axis=1)) > 1e-10))
    route = route[keep]
    if route.shape[0] == 1:
        route = np.vstack((route, route))
    raw_route = route.copy()
    # Keep a shortcut only when its sampled weighted cost does not increase.
    result = [route[0]]
    cursor = 0
    while cursor < route.shape[0] - 1:
        selected = cursor + 1
        old_cost = 0.0
        for last in range(cursor + 1, route.shape[0]):
            old_cost += edge_cost(route[last - 1], route[last])
            if free(route[cursor], route[last]) and edge_cost(route[cursor], route[last]) <= old_cost + 1e-9:
                selected = last
        result.append(route[selected])
        cursor = selected
    waypoints = np.vstack(result)
    return Plan(
        WaypointsNedM=waypoints,
        RawWaypointsNedM=raw_route,
        LengthM=float(np.sum(np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1)))),
        ExpandedNodes=int(expanded),
        SafetyRadiusM=float(radius),
        PlanningPaddingM=float(padding),
    )
