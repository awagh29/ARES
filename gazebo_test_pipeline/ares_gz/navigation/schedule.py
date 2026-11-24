"""Port of ``outageSchedule.m``: quantized, reproducible lidar outages.

Returns the same (times, enabled) zero-order-hold schedule MATLAB feeds to the
``LidarEnableSchedule`` From Workspace block, plus the [start, end) intervals.
"""

from __future__ import annotations

import math

import numpy as np

from .matlab_rng import MatlabMT19937
from .options import NavigationOptions

TICK_S = 0.1


def outage_schedule(options: NavigationOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    o = options
    T = o.StopTimeS
    tick = TICK_S
    mode = o.LidarMode
    if mode == "blackout":
        intervals = np.array([[0.0, T]])
    elif mode == "healthy":
        intervals = np.zeros((0, 2))
    elif mode == "custom":
        intervals = o.outage_intervals.copy()
    else:
        n = o.NumberOfOutages
        intervals = np.zeros((n, 2))
        if n > 0:
            first = math.ceil(o.InitialHealthyTimeS / tick)
            tail = math.ceil(o.FinalHealthyTimeS / tick)
            gap = max(1, math.ceil(o.MinimumRecoveryTimeS / tick))
            last = math.floor(min(T, o.OutageWindowEndS) / tick)
            lo = math.ceil(o.OutageDurationS[0] / tick)
            hi = math.floor(o.OutageDurationS[1] / tick)
            if hi < lo:
                raise ValueError("No duration fits the 0.1 s scan grid.")
            budget = last - first - tail - (n - 1) * gap
            if budget < n * lo:
                raise ValueError(f"Not enough time for {n} outages.")
            stream = MatlabMT19937(o.Seed)
            durations = [lo] * n
            for k in range(n):
                extra = min(hi - lo, budget - sum(durations))
                durations[k] = durations[k] + stream.randi(extra + 1) - 1
            spare = budget - sum(durations)
            weights = [stream.rand() for _ in range(n + 1)]
            total = 0.0
            for weight in weights:  # sequential sum, as MATLAB does for short vectors
                total += weight
            gaps = [math.floor(spare * weight / total) for weight in weights]
            gaps[-1] = gaps[-1] + spare - sum(gaps)
            cursor = first + gaps[0]
            for k in range(n):
                intervals[k, :] = (cursor * tick, (cursor + durations[k]) * tick)
                cursor = cursor + durations[k] + gap + gaps[k + 1]
    if intervals.size:
        if not np.all((intervals[:, 0] >= 0) & (intervals[:, 1] > intervals[:, 0]) & (intervals[:, 1] <= T)):
            raise ValueError("Outage intervals must satisfy 0 <= start < end <= StopTimeS.")
        if not np.all(intervals[1:, 0] > intervals[:-1, 1]):
            raise ValueError("Outages must be sorted, disjoint, and separated by a healthy interval.")
        ticks = intervals.reshape(-1) / tick
        if not np.all(np.abs(ticks - np.round(ticks)) < 1e-7):
            raise ValueError("Outage edges must be multiples of 0.1 s.")
    times = np.unique(np.concatenate(([0.0], intervals.flatten(order="F"), [T])))
    enabled = np.ones(times.shape)
    for start, end in intervals:
        enabled[(times >= start) & (times < end)] = 0.0
    if intervals.size and intervals[-1, 1] == T:
        enabled[-1] = 0.0
    return times, enabled, intervals


def availability_at(times: np.ndarray, enabled: np.ndarray, time_s: float) -> bool:
    """Zero-order hold lookup (MATLAB ``interp1(...,'previous')``) with a tick tolerance."""
    index = int(np.searchsorted(times, time_s + 1e-9, side="right") - 1)
    return bool(enabled[max(index, 0)] >= 0.5)
