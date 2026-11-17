"""Parity checks for the schedule, frames and the three-vehicle closed loop."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ares_three_uav.frames import enu_orientation
from ares_three_uav.matlab_rand import MatlabMT19937
from ares_three_uav.schedule import intermittent_schedule, lidar_guidance_enabled
from ares_three_uav.simulate import fly


RECORDED_OUTAGES = np.array([
    [3.5443486030280162, 5.1058226856548945],
    [10.356031289257292, 12.010370739258892],
    [14.974288494995, 16.549329689818645],
])


class ScheduleAndFlightTest(unittest.TestCase):
    def test_matlab_stream_and_recorded_outages(self):
        self.assertAlmostEqual(MatlabMT19937(5489).rand(), 0.8147236863931789, places=12)
        _times, enabled, intervals = intermittent_schedule()
        np.testing.assert_allclose(intervals, RECORDED_OUTAGES, atol=1e-9)
        self.assertEqual(enabled.tolist(), [1, 0, 1, 0, 1, 0, 1, 1])
        self.assertFalse(lidar_guidance_enabled(_times, enabled, 4.0))
        self.assertTrue(lidar_guidance_enabled(_times, enabled, 2.0))
        self.assertTrue(lidar_guidance_enabled(_times, enabled, 8.0))
        self.assertFalse(lidar_guidance_enabled(_times, enabled, 11.0))
        self.assertFalse(lidar_guidance_enabled(_times, enabled, 15.5))
        self.assertTrue(lidar_guidance_enabled(_times, enabled, 18.0))

    def test_level_nose_points_north(self):
        roll, pitch, yaw, _quat = enu_orientation(0.0, 0.0, 0.0)
        self.assertAlmostEqual(roll, 0.0, places=8)
        self.assertAlmostEqual(pitch, 0.0, places=8)
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=8)

    def test_three_drones_match_and_clear_the_goal(self):
        _pilots, rows, summary, _playback = fly()
        relative = []
        for drone in summary["drones"]:
            self.assertTrue(drone["goal_reached"], drone)
            self.assertTrue(drone["clearance_ok"], drone)
            self.assertLess(drone["finish_time_s"], 60.0)
            north, east, down = drone["final_position_ned_m"]
            relative.append((north, east - drone["east_offset_m"], down))
        np.testing.assert_allclose(relative[0], relative[1], atol=1e-8)
        np.testing.assert_allclose(relative[0], relative[2], atol=1e-8)
        by_drone = {}
        for row in rows:
            by_drone.setdefault(row["drone"], []).append(row)
        for row in by_drone["uav_001"]:
            if 3.6 <= row["time_s"] <= 5.0:
                self.assertEqual(row["avoidance_used"], 0)
            if 1.0 <= row["time_s"] <= 3.0:
                self.assertEqual(row["avoidance_used"], 1)


if __name__ == "__main__":
    unittest.main()
