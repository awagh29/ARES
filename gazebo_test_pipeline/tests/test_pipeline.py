"""Unit tests for the Gazebo-side pieces that do not need Gazebo running."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest

from ares_gz.frames import (gazebo_orientation_to_ned_euler, matrix_to_quaternion, ned_euler_to_gazebo_rpy,
                            ned_to_enu, ENU_NED, FLU_FRD)
from ares_gz.metrics import evaluate_axis_run, route_deviation, signed_box_distance
from ares_gz.navigation import default_options, virtual_lidar
from ares_gz.navigation.geometry import eul2rotm_zyx
from ares_gz.navigation.lidar import COLUMNS, ROWS, gazebo_cloud_to_matlab_order
from ares_gz.run import CommandMapper
from ares_gz.scenarios import load_matrix
from ares_gz.scene import scene_map
from ares_gz.wind import PRESETS, DragModel, Gust, WindField
from ares_gz.world import world_sdf


@pytest.fixture(scope="module")
def scene():
    return scene_map()


def test_ned_enu_round_trip():
    assert np.allclose(ned_to_enu([1.0, 2.0, -3.0]), [2.0, 1.0, 3.0])
    rng = np.random.default_rng(1)
    for _ in range(50):
        euler = rng.uniform([-math.pi, -0.6, -0.6], [math.pi, 0.6, 0.6])
        rotation_enu_flu = ENU_NED @ eul2rotm_zyx(euler) @ FLU_FRD
        quaternion = matrix_to_quaternion(rotation_enu_flu)
        assert np.allclose(gazebo_orientation_to_ned_euler(*quaternion), euler, atol=1e-9)


def test_north_facing_drone_has_gazebo_yaw_90_degrees():
    roll, pitch, yaw = ned_euler_to_gazebo_rpy([0.0, 0.0, 0.0])
    assert abs(roll) < 1e-12 and abs(pitch) < 1e-12 and math.isclose(yaw, math.pi / 2)


def test_gazebo_cloud_reordering_matches_matlab_order(scene):
    # Build a Gazebo-style packed cloud (rows = elevation, azimuth fastest) from the MATLAB-order one.
    cloud, _, _ = virtual_lidar(scene, [12.0, 14.0, -7.0], [0.3, 0.0, 0.0])
    gazebo_packed = cloud.reshape(COLUMNS, ROWS, 3).transpose(1, 0, 2).reshape(-1, 3)
    restored = gazebo_cloud_to_matlab_order(gazebo_packed)
    assert np.array_equal(np.isnan(restored), np.isnan(cloud))
    assert np.allclose(np.nan_to_num(restored), np.nan_to_num(cloud))


def test_gust_shape_is_one_minus_cosine():
    gust = Gust(start_s=2.0, duration_s=2.0, peak_mps=6.0, heading_deg=90.0)
    assert np.allclose(gust.velocity_ned(1.9), 0.0)
    assert np.allclose(gust.velocity_ned(3.0), [0.0, 6.0, 0.0], atol=1e-12)  # peak, blowing east
    assert np.allclose(gust.velocity_ned(4.1), 0.0)


def test_wind_is_reproducible_and_bounded():
    first = WindField(PRESETS["medium"], seed=4, stop_time_s=60)
    second = WindField(PRESETS["medium"], seed=4, stop_time_s=60)
    other = WindField(PRESETS["medium"], seed=5, stop_time_s=60)
    assert np.array_equal(first.velocity, second.velocity)
    assert not np.array_equal(first.velocity, other.velocity)
    assert len(first.gusts) == 4
    assert first.describe()["max_speed_mps"] < 2.0 + 4 * 6.0 + 5.0
    calm = WindField(PRESETS["calm"], seed=4, stop_time_s=60)
    assert np.all(calm.velocity == 0)


def test_drag_opposes_relative_motion():
    drag = DragModel()
    force = drag.force_ned([0, 0, 0], [2.5, 0, 0])
    assert force[0] < 0 and force[1] == 0
    force = drag.force_ned([0, 6, 0], [0, 0, 0])
    assert math.isclose(force[1], 0.5 * 1.225 * 0.06 * 36)


def test_signed_distance_and_scoring(scene):
    # Building C spans N 13.5-16.5, E 13.5-16.5: a point 2 m east of its east face.
    assert math.isclose(signed_box_distance(np.array([[15.0, 18.5, -7.0]]), scene)[0], 2.0)
    assert signed_box_distance(np.array([[15.0, 15.0, -7.0]]), scene)[0] == -1.5  # inside C
    options = default_options()
    time = np.linspace(0, 10, 101)
    path = np.linspace(options.start_state[:3], options.goal_state[:3], 101)
    metrics = evaluate_axis_run(time, path, options, scene)
    assert metrics["goalReachedAtTermination"] and metrics["collisionOccurred"]  # straight line hits a building
    assert route_deviation(path, np.vstack((path[0], path[-1]))).max() < 1e-9


def test_command_mapper_reaches_requested_average_acceleration():
    mapper = CommandMapper(gain_per_s=1.8, frame="world")
    dt, velocity, accel = 0.05, np.array([1.0, -0.5, 0.2]), np.array([0.8, 0.3, -0.4])
    setpoint_ned = ENU_NED @ mapper.velocity_setpoint(velocity, accel, dt, yaw_enu=0.0)
    # First-order response over one tick: v(dt) = sp - (sp - v0) * exp(-k dt)
    end = setpoint_ned - (setpoint_ned - velocity) * math.exp(-1.8 * dt)
    assert np.allclose((end - velocity) / dt, accel)


def test_world_file_is_valid_xml(scene):
    root = ElementTree.fromstring(world_sdf(scene, [0.0, 0.0, -7.0]))
    world = root.find("world")
    names = [model.get("name") for model in world.findall("model")]
    assert names.count("ares_quad") == 1 and sum(name.startswith("building_") for name in names) == 6
    lidar = world.find(".//sensor[@type='gpu_lidar']")
    assert lidar.find("lidar/scan/horizontal/samples").text == "716"
    assert lidar.find("lidar/scan/vertical/samples").text == "15"
    assert float(lidar.find("lidar/range/max").text) == 7.0


def test_matrix_covers_the_five_scenarios():
    matrix = load_matrix()
    ids = [scenario.id for scenario in matrix.scenarios]
    for prefix in ("S0", "S1", "S2", "S3", "S4", "S5"):
        assert any(item.startswith(prefix) for item in ids)
    winds = {scenario.wind.name for scenario in matrix.scenarios}
    assert winds == {"calm", "light", "medium"}
