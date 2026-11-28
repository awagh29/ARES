"""Hub <-> drone radio links (star topology: drones never talk to each other).

Each link delays messages by ``latency + U(0, jitter)`` and drops them with
probability ``drop``; everything is seeded, so a run is repeatable.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommsProfile:
    name: str = "ideal"
    latency_s: float = 0.0
    jitter_s: float = 0.0
    drop: float = 0.0


PROFILES = {
    "ideal": CommsProfile("ideal"),
    "nominal": CommsProfile("nominal", 0.05, 0.02, 0.02),
    "lossy": CommsProfile("lossy", 0.10, 0.05, 0.15),
}


class Link:
    def __init__(self, profile: CommsProfile, seed: int):
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.queue: list = []
        self.order = itertools.count()
        self.sent = 0
        self.dropped = 0

    def send(self, time_s: float, message) -> None:
        self.sent += 1
        if self.profile.drop > 0 and self.rng.random() < self.profile.drop:
            self.dropped += 1
            return
        delay = self.profile.latency_s + (self.rng.uniform(0, self.profile.jitter_s) if self.profile.jitter_s else 0.0)
        heapq.heappush(self.queue, (time_s + delay, next(self.order), message))

    def receive(self, time_s: float) -> list:
        delivered = []
        while self.queue and self.queue[0][0] <= time_s + 1e-9:
            delivered.append(heapq.heappop(self.queue)[2])
        return delivered

    def stats(self) -> dict:
        return {"sent": self.sent, "dropped": self.dropped}
