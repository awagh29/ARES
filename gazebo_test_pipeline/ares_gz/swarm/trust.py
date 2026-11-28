"""Interpretable LiDAR trust model.

Trust is 1 minus a weighted sum of four evidence penalties, each in [0, 1]
and each reported with a plain-language reason:

* physical   - is there a usable measured scan right now? (no scan = 1.0;
               returns inside the airframe or an empty scan beside a mapped
               wall = 0.5)
* temporal   - share of the last ``window_s`` without a measured scan, and the
               current continuous outage relative to ``outage_limit_s``
* peer       - share of the drone's scan that contradicts the map built by the
               *other* drones (needs enough overlap to count)
* historical - how many failure episodes so far and the cumulative outage share

    trust = 1 - (0.35 physical + 0.30 temporal + 0.20 peer + 0.15 historical)

States, with hysteresis so a drone does not flicker:
    NOMINAL      trust >= 0.70 (re-entered only after trust >= 0.80 for ``recover_s``)
    DEGRADED     0.40 <= trust < 0.70   ("flag")
    QUARANTINED  trust < 0.40           ("contain")
A blackout longer than ``outage_limit_s`` gives temporal = 1, so trust drops below
0.40 and the drone is quarantined; a 1-2 s intermittent outage only flags it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

WEIGHTS = {"physical": 0.35, "temporal": 0.30, "peer": 0.20, "historical": 0.15}
NOMINAL, DEGRADED, QUARANTINED = "NOMINAL", "DEGRADED", "QUARANTINED"


@dataclass
class TrustAssessment:
    trust: float
    state: str
    penalties: dict
    reasons: list = field(default_factory=list)


class TrustModel:
    def __init__(self, window_s: float = 5.0, outage_limit_s: float = 3.0, recover_s: float = 3.0,
                 peer_min_checked: int = 30, period_s: float = 0.1):
        self.window = deque(maxlen=int(round(window_s / period_s)))
        self.outage_limit_s = outage_limit_s
        self.recover_s = recover_s
        self.peer_min_checked = peer_min_checked
        self.period_s = period_s
        self.state = NOMINAL
        self.outage_s = 0.0
        self.total_s = 0.0
        self.outage_total_s = 0.0
        self.episodes = 0
        self.high_trust_s = 0.0
        self.peer_penalty = 0.0
        self.last: TrustAssessment | None = None

    def update(self, measured: bool, scan_sane: bool = True, peer_contradiction: float | None = None,
               peer_checked: int = 0) -> TrustAssessment:
        dt = self.period_s
        self.total_s += dt
        if not measured:
            if self.outage_s == 0.0:
                self.episodes += 1
            self.outage_s += dt
            self.outage_total_s += dt
        else:
            self.outage_s = 0.0
        self.window.append(0 if measured else 1)
        reasons = []

        physical = 0.0
        if not measured:
            physical = 1.0
            reasons.append("no measured LiDAR scan this tick")
        elif not scan_sane:
            physical = 0.5
            reasons.append("scan fails sanity checks")

        window_share = sum(self.window) / self.window.maxlen
        continuous = min(1.0, self.outage_s / self.outage_limit_s)
        temporal = max(window_share, continuous)
        if temporal > 0:
            reasons.append(f"{window_share:.0%} of last {self.window.maxlen * dt:.0f} s without scans"
                           + (f", outage running {self.outage_s:.1f} s" if self.outage_s else ""))

        if peer_contradiction is not None and peer_checked >= self.peer_min_checked:
            # Smooth: one odd scan should not swing trust.
            self.peer_penalty = 0.7 * self.peer_penalty + 0.3 * min(1.0, 2.0 * peer_contradiction)
        elif not measured:
            self.peer_penalty *= 0.95
        peer = self.peer_penalty
        if peer > 0.1:
            reasons.append(f"scan disagrees with peers' map ({peer:.2f})")

        share = self.outage_total_s / max(self.total_s, 10.0)  # no spike in the first seconds
        historical = 0.5 * min(1.0, self.episodes / 4.0) + 0.5 * min(1.0, share)
        if self.episodes:
            reasons.append(f"{self.episodes} failure episode(s), {share:.0%} of flight without scans")

        penalties = {"physical": physical, "temporal": temporal, "peer": peer, "historical": historical}
        trust = max(0.0, 1.0 - sum(WEIGHTS[name] * value for name, value in penalties.items()))

        if trust < 0.40:
            self.state = QUARANTINED
            self.high_trust_s = 0.0
        elif self.state == NOMINAL:
            if trust < 0.70:
                self.state = DEGRADED
        else:  # DEGRADED or QUARANTINED: recover only after sustained high trust
            self.high_trust_s = self.high_trust_s + dt if trust >= 0.80 else 0.0
            if self.high_trust_s >= self.recover_s:
                self.state = NOMINAL
            elif self.state == QUARANTINED and trust >= 0.40 and measured:
                self.state = DEGRADED
        self.last = TrustAssessment(round(trust, 4), self.state, penalties, reasons)
        return self.last
