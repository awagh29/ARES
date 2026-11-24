"""Python port of the MATLAB map-correction solution (``src/+ares/+navigation``).

Every function keeps the MATLAB name in its docstring and the same operation
order, so the parity tests can compare outputs value for value.
"""

from .options import NavigationOptions, default_options, validate_options
from .geometry import eul2rotm_zyx, map_clearance, segment_free
from .qp import project_control
from .barrier import BarrierInfo, correct_motion
from .planner import Plan, plan_path
from .schedule import availability_at, outage_schedule
from .lidar import measured_surfaces, virtual_lidar
from .controller import ControllerOutput, MapCorrectionController

__all__ = [
    "NavigationOptions", "default_options", "validate_options",
    "eul2rotm_zyx", "map_clearance", "segment_free", "project_control",
    "BarrierInfo", "correct_motion", "Plan", "plan_path",
    "availability_at", "outage_schedule", "measured_surfaces", "virtual_lidar",
    "ControllerOutput", "MapCorrectionController",
]
