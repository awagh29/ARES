"""Simulated victims and the drones' victim detector (a downward camera, independent of LiDAR)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene import SceneMap


@dataclass(frozen=True)
class Victim:
    id: str
    north: float
    east: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east])


def line_of_sight(a_ne, b_ne, scene: SceneMap, samples: int = 40) -> bool:
    """True when the horizontal segment a->b crosses no building footprint."""
    a, b = np.asarray(a_ne, float), np.asarray(b_ne, float)
    points = a + np.linspace(0, 1, samples)[:, None] * (b - a)
    for k in range(scene.box_min.shape[0]):
        inside = ((points[:, 0] > scene.box_min[k, 0]) & (points[:, 0] < scene.box_max[k, 0])
                  & (points[:, 1] > scene.box_min[k, 1]) & (points[:, 1] < scene.box_max[k, 1]))
        if inside.any():
            return False
    return True


class VictimDetector:
    """Detects victims within ``range_m`` horizontally with a clear line of sight."""

    def __init__(self, victims: list[Victim], scene: SceneMap, range_m: float = 5.0, noise_m: float = 0.3,
                 seed: int = 0, repeat_s: float = 2.0):
        self.victims = victims
        self.scene = scene
        self.range_m = range_m
        self.noise_m = noise_m
        self.rng = np.random.default_rng(seed)
        self.repeat_s = repeat_s
        self.last_report: dict = {}

    def detect(self, drone: str, position_ned, time_s: float) -> list[dict]:
        found = []
        p = np.asarray(position_ned, float)[:2]
        for victim in self.victims:
            if np.linalg.norm(victim.position - p) > self.range_m:
                continue
            if not line_of_sight(p, victim.position, self.scene):
                continue
            key = (drone, victim.id)
            if time_s - self.last_report.get(key, -np.inf) < self.repeat_s:
                continue
            self.last_report[key] = time_s
            estimate = victim.position + self.rng.normal(0.0, self.noise_m, 2)
            found.append({"truth_id": victim.id, "position_ne": estimate.tolist()})
        return found
