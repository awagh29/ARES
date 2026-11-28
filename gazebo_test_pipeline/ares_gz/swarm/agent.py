"""Drone-side swarm logic: obey the hub's downlink and enforce the separation invariant.

The drone's own guidance (Atharva's CNN + planner + braking governor in
MATLAB) produces a nominal velocity. Before it reaches the velocity PID, the
agent:

* holds position (nominal 0) when the hub says ``hold`` or before the first
  downlink, and after ``comms_timeout_s`` without one (comms-loss safe mode);
* clips speed to the hub's cap for the current mode;
* projects the command onto its share of every pairwise CBF constraint,
  inflating each neighbour's radius by ``speed * age`` of the position the
  hub reported (so stale data stays conservative).
"""

from __future__ import annotations

import numpy as np

from .cbf import SeparationContract, filter_velocity
from .hub import Downlink


class DroneAgent:
    def __init__(self, drone: str, comms_timeout_s: float = 0.5, hold_cap_mps: float = 0.5,
                 max_inflation_m: float = 2.0):
        self.drone = drone
        self.comms_timeout_s = comms_timeout_s
        self.hold_cap_mps = hold_cap_mps
        self.max_inflation_m = max_inflation_m
        self.downlink: Downlink | None = None
        self.goal_ned: np.ndarray | None = None  # goal currently loaded in the onboard planner
        self.rejected_goal: np.ndarray | None = None
        self.last_filter: dict = {}
        self.comms_loss_ticks = 0

    def receive(self, messages: list[Downlink]) -> None:
        for message in messages:
            if self.downlink is None or message.seq > self.downlink.seq:
                self.downlink = message

    def mode(self, time_s: float) -> str:
        if self.downlink is None:
            return "WAITING"
        if time_s - self.downlink.time_s > self.comms_timeout_s:
            return "COMMS-LOSS"
        return self.downlink.mode

    def wanted_goal(self, time_s: float) -> np.ndarray | None:
        """Goal the onboard planner should have (None = keep the current one and hold)."""
        if self.downlink is None or self.mode(time_s) == "COMMS-LOSS" or self.downlink.target_ned is None:
            return None
        return np.asarray(self.downlink.target_ned, float)

    def needs_retarget(self, time_s: float) -> np.ndarray | None:
        goal = self.wanted_goal(time_s)
        if goal is None:
            return None
        if self.rejected_goal is not None and np.linalg.norm(goal - self.rejected_goal) <= 0.5:
            return None  # the onboard planner refused it; the hub will reassign
        if self.goal_ned is None or np.linalg.norm(goal - self.goal_ned) > 0.5:
            return goal
        return None

    def holding(self, time_s: float) -> bool:
        mode = self.mode(time_s)
        return (mode in ("WAITING", "COMMS-LOSS") or self.downlink is None or self.downlink.target_ned is None
                or self.goal_ned is None or np.linalg.norm(self.downlink.target_ned - self.goal_ned) > 0.5)

    def filter(self, time_s: float, position_ned, nominal_velocity) -> np.ndarray:
        nominal = np.asarray(nominal_velocity, float)
        downlink = self.downlink
        if self.holding(time_s):
            nominal = np.zeros(3)
            cap = self.hold_cap_mps
        else:
            cap = float(downlink.speed_cap) if downlink.speed_cap > 0 else self.hold_cap_mps
            if downlink.speed_cap <= 0:
                nominal = np.zeros(3)
        if self.mode(time_s) == "COMMS-LOSS":
            self.comms_loss_ticks += 1
        contracts = []
        gamma = 1.0
        if downlink is not None:
            gamma = downlink.gamma
            for contract, stamp, speed in zip(downlink.contracts, downlink.stamps, downlink.neighbour_speed):
                inflation = min(self.max_inflation_m, max(0.0, time_s - stamp) * speed)
                contracts.append(SeparationContract(contract.neighbour, contract.position_ned,
                                                    contract.radius_m + inflation, contract.share))
        speed = np.linalg.norm(nominal)
        if speed > cap > 0:
            nominal = nominal * cap / speed
        safe, info = filter_velocity(nominal, position_ned, contracts, gamma, max(cap, 1e-3))
        self.last_filter = info
        return safe
