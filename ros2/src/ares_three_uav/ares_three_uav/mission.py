"""Scene, sensor and controller constants taken from the running single-UAV model.

The flight altitude is the waypoint and UAV Toolbox example value, NED down = -7 m.
``defaultSingleUav`` currently stores ``[0 0 7]``; the axis-start checks, the
guidance calibration and the MathWorks model this Simulink file was built from
all use ``[0 0 -7]``. The polygons in ``initializeSingleUav`` are ENU mesh
coordinates, so the first column is East and the second is North.
"""

from __future__ import annotations

import math

import numpy as np

INITIAL_NED = np.array([0.0, 0.0, -7.0])
WAYPOINTS_NED = np.array([
    [0.0, 0.0, -7.0],
    [30.0, 35.0, -7.0],
], dtype=float)

# addMesh polygon centers. The duplicate [10, 20] row is the same building.
OBSTACLE_CENTERS_ENU = np.array([
    [10.0, 0.0],
    [15.0, 15.0],
    [10.0, 20.0],
    [5.0, 15.0],
    [22.0, 18.0],
    [15.0, 25.0],
    [10.0, 20.0],
], dtype=float)
OBSTACLE_WIDTH_M = 3.0
OBSTACLE_HEIGHT_M = 15.0

DRONE_IDS = ("uav_001", "uav_002", "uav_003")
# Farther apart than the 7 m lidar, so each vehicle sees only its own copy of the buildings.
DRONE_EAST_OFFSETS_M = (0.0, 60.0, 120.0)

PLANT_DT_S = 0.001
LIDAR_DT_S = 0.1
STOP_TIME_S = 60.0

WAYPOINT_LOOKAHEAD_M = 5.0
WAYPOINT_MIN_LOOKAHEAD_M = 0.2
TRANSITION_RADIUS_M = 0.1
AVOIDANCE_LOOKAHEAD_M = 0.5

VEHICLE_RADIUS_M = 0.1
SAFETY_DISTANCE_M = 1.5
SENSOR_OFFSET_BODY_NED = np.array([0.0, 0.0, -0.4])
SENSOR_MIN_RANGE_M = 0.2
SENSOR_MAX_RANGE_M = 7.0
AZIMUTH_DEG = np.linspace(-179.0, 179.0, 716)
ELEVATION_DEG = np.linspace(-15.0, 15.0, 15)
HORIZONTAL_FOV_DEG = (-179.0, 179.0)
VERTICAL_FOV_DEG = (-15.0, 15.0)

# Histogram parameters on the Obstacle Avoidance block.
HISTOGRAM_RESOLUTION_DEG = 6.0
WINDOW_SIZE = 1
HISTOGRAM_THRESHOLD = 1
MAX_AGE = 3
TARGET_DIRECTION_WEIGHT = 5.0
CURRENT_DIRECTION_WEIGHT = 2.0
PREVIOUS_DIRECTION_WEIGHT = 2.0

# Position loop writes a velocity command. Velocity loop writes acceleration.
# Both read Px/Py/Pz, Dx/Dy/Dz, Nx/Ny/Nz from initializeSingleUav.m.
POSITION_GAINS = ((6.0, 0.0, 1.5, 10.0), (6.0, 0.0, 1.5, 10.0), (6.5, 0.0, 2.5, 14.4947065605712))
POSITION_LIMITS = ((-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0))
VELOCITY_LIMITS = ((-1.0, 1.0), (-1.0, 1.0), (-4.0, 4.0))
YAW_P_GAIN = 2.0
YAW_RATE_LIMIT_RADPS = 0.5
ATTITUDE_LIMIT_RAD = math.radians(25.0)

MASS_KG = 0.1
GRAVITY_MPS2 = 9.81
ROLL_KP = 3402.97
ROLL_KD = 116.67
PITCH_KP = 3402.97
PITCH_KD = 116.67
YAW_RATE_GAIN = 1950.0
THRUST_GAIN = 3900.0


def unique_obstacle_centers_enu() -> np.ndarray:
    rounded = np.round(OBSTACLE_CENTERS_ENU, 6)
    _, index = np.unique(rounded, axis=0, return_index=True)
    return OBSTACLE_CENTERS_ENU[np.sort(index)]


def obstacle_boxes_ned(east_offset_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned building boxes in NED, including a drone's east offset."""
    half = OBSTACLE_WIDTH_M / 2.0
    centers = unique_obstacle_centers_enu()
    lower = np.column_stack((
        centers[:, 1] - half,
        centers[:, 0] + east_offset_m - half,
        np.full(len(centers), -OBSTACLE_HEIGHT_M),
    ))
    upper = np.column_stack((
        centers[:, 1] + half,
        centers[:, 0] + east_offset_m + half,
        np.zeros(len(centers)),
    ))
    return lower, upper
