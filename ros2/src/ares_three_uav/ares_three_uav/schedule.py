"""Seed-513 LiDAR-guidance outages from the single-UAV experiment.

The draws, order and zero-order hold edges match
``createIntermittentSchedule`` and the local generator in
``scripts/captureIntermittentLidarDisabledSingleUav.m``. While the schedule is
false, the Simulink switch bypasses obstacle avoidance and flies the waypoint
lookahead. The lidar model itself keeps running.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matlab_rand import MatlabMT19937


@dataclass(frozen=True)
class FaultConfiguration:
    random_seed: int = 513
    number_of_outages: int = 3
    initial_healthy_time_s: float = 3.0
    final_healthy_time_s: float = 4.0
    minimum_outage_duration_s: float = 1.5
    maximum_outage_duration_s: float = 3.0
    minimum_recovery_duration_s: float = 2.0
    fault_window_end_time_s: float = 21.0


def intermittent_schedule(stop_time_s: float = 60.0, fault: FaultConfiguration | None = None):
    fault = fault or FaultConfiguration()
    window_end = min(stop_time_s, fault.fault_window_end_time_s)
    count = fault.number_of_outages
    required = (
        fault.initial_healthy_time_s
        + count * fault.minimum_outage_duration_s
        + max(count - 1, 0) * fault.minimum_recovery_duration_s
        + fault.final_healthy_time_s
    )
    if window_end < required:
        raise ValueError(f"Fault window {window_end} s is shorter than the required {required} s.")
    stream = MatlabMT19937(fault.random_seed)
    intervals = np.zeros((count, 2))
    earliest = fault.initial_healthy_time_s
    for index in range(count):
        remaining = count - index - 1
        remaining_minimum = (
            fault.final_healthy_time_s
            + remaining * fault.minimum_outage_duration_s
            + remaining * fault.minimum_recovery_duration_s
        )
        latest = window_end - fault.minimum_outage_duration_s - remaining_minimum
        if latest < earliest:
            raise ValueError(f"No start window remains for outage {index + 1}.")
        start = earliest + stream.rand() * (latest - earliest)
        maximum_duration = min(
            fault.maximum_outage_duration_s,
            window_end - start - remaining_minimum,
        )
        if maximum_duration < fault.minimum_outage_duration_s:
            raise ValueError(f"No duration window remains for outage {index + 1}.")
        duration = fault.minimum_outage_duration_s + stream.rand() * (
            maximum_duration - fault.minimum_outage_duration_s
        )
        end = start + duration
        intervals[index] = (start, end)
        earliest = end + fault.minimum_recovery_duration_s
    if intervals[-1, 1] + fault.final_healthy_time_s > window_end + 1e-9:
        raise ValueError("The final healthy interval was not preserved.")
    times = np.concatenate(([0.0], intervals.reshape(-1), [stop_time_s]))
    # Starts enabled, then off/on for every outage, and stays enabled through the stop time.
    enabled = np.concatenate(([1.0], np.tile([0.0, 1.0], count), [1.0]))
    if np.any(np.diff(times) <= 0):
        raise ValueError("LiDAR schedule timestamps must increase strictly.")
    return times, enabled, intervals


def lidar_guidance_enabled(times: np.ndarray, enabled: np.ndarray, time_s: float) -> bool:
    """Zero-order hold, matching ``setinterpmethod(schedule, 'zoh')``."""
    index = int(np.searchsorted(times, time_s, side="right") - 1)
    return bool(enabled[max(index, 0)] >= 0.5)
