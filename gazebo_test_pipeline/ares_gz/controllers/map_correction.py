"""Atharva's map-correction controller (Python port of src/+ares/+navigation)."""

from __future__ import annotations

from ..navigation import MapCorrectionController
from ..navigation.options import NavigationOptions
from ..scene import SceneMap
from .base import Command, Controller, VehicleState


class MapCorrectionAdapter(Controller):
    name = "map-correction"
    tick_s = 0.05
    guidance_period_s = 0.05

    def __init__(self, scene: SceneMap):
        self.scene = scene

    def start(self, options: NavigationOptions, route_hint=None) -> dict:
        self.controller = MapCorrectionController(self.scene, options, route_hint)
        return {"route_ned": [list(map(float, row)) for row in route_hint]}

    def step(self, state: VehicleState, lidar_available: bool, points_matlab_order) -> Command:
        out = self.controller.step(points_matlab_order if lidar_available else None, state.position_ned,
                                   state.velocity_ned, state.euler_zyx, lidar_available, state.time_s)
        record = {
            "ref_n": out.reference_ned[0], "ref_e": out.reference_ned[1], "ref_d": out.reference_ned[2],
            "nominal_n": out.nominal_ned[0], "nominal_e": out.nominal_ned[1], "nominal_d": out.nominal_ned[2],
            "filter_feasible": int(out.info.Feasible), "minimum_h": out.info.MinimumH,
            "minimum_psi1": out.info.MinimumPsi1, "waypoint_index": out.waypoint_index,
            "measured_points": out.measured_point_count,
        }
        return Command(out.acceleration_ned, out.yaw_rate, stop=out.stop, goal_reached=out.goal_reached,
                       healthy=out.info.Feasible, record=record)
