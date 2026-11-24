"""Python port vs MATLAB: every function must reproduce the MATLAB outputs.

The reference values come from ``matlab/exportParityFixtures.m``, which runs
Atharva's real ``src/+ares/+navigation`` code. Regenerate them after any change
to that MATLAB code; a failure here means the port is out of date.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace

import numpy as np
import pytest

from ares_gz.navigation import (
    MapCorrectionController, correct_motion, default_options, map_clearance, outage_schedule,
    plan_path, project_control, segment_free, validate_options, virtual_lidar,
)
from ares_gz.navigation.matlab_rng import MatlabMT19937
from ares_gz.navigation.options import NavigationOptions
from ares_gz.paths import FIXTURES_ROOT
from ares_gz.scene import SceneMap, read_initializer, scene_map

FIXTURE = FIXTURES_ROOT / "matlab_parity.json"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="run matlab/exportParityFixtures.m first")


@pytest.fixture(scope="module")
def data():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def scene():
    return scene_map()


def rows(value, width=3) -> np.ndarray:
    """MATLAB jsonencode flattens 1-row matrices and drops empty ones; restore (n, width)."""
    if value is None:
        return np.zeros((0, width))
    return np.asarray(value, dtype=float).reshape(-1, width)


def vector(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float))


def options_from_matlab(struct: dict) -> NavigationOptions:
    known = {item.name for item in fields(NavigationOptions)}
    values = {key: value for key, value in struct.items() if key in known}
    if "OutageIntervalsS" in values:
        values["OutageIntervalsS"] = rows(values["OutageIntervalsS"], 2)
    return validate_options(values)


# --- configuration ---------------------------------------------------------------

def test_default_options_match(data):
    matlab = data["DefaultOptions"]
    python = default_options().to_dict()
    assert set(matlab) <= set(python), f"Options missing from the port: {set(matlab) - set(python)}"
    for key, value in matlab.items():
        actual = python[key]
        if isinstance(value, str):
            assert actual == value, key
        elif key == "OutageIntervalsS":
            assert rows(value, 2).size == 0 and len(actual) == 0
        else:
            np.testing.assert_allclose(np.asarray(actual, dtype=float).ravel(),
                                       np.asarray(value, dtype=float).ravel(), err_msg=key)


def test_scene_matches_initializer(data, scene):
    positions, width, height = read_initializer()
    np.testing.assert_array_equal(positions, rows(data["Scene"]["ObstaclePositions"], 2))
    assert width == data["Scene"]["ObstacleWidth"] and height == data["Scene"]["ObstacleHeight"]
    np.testing.assert_array_equal(scene.box_min, rows(data["Scene"]["BoxMin"]))
    np.testing.assert_array_equal(scene.box_max, rows(data["Scene"]["BoxMax"]))


# --- geometry and QP ------------------------------------------------------------

def test_map_clearance(data, scene):
    for case in data["MapClearance"]:
        distance, normal = map_clearance(case["position"], scene.box_min, scene.box_max)
        np.testing.assert_allclose(distance, vector(case["distance"]), atol=1e-12)
        np.testing.assert_allclose(normal, rows(case["normal"]), atol=1e-12)


def test_segment_free(data, scene):
    bounds = default_options().bounds
    for case in data["SegmentFree"]:
        assert segment_free(case["first"], case["last"], scene.box_min, scene.box_max, case["padding"], bounds) == bool(case["free"])


def test_project_control(data):
    for index, case in enumerate(data["ProjectControl"]):
        solution, feasible = project_control(case["request"], rows(case["A"]), vector(case["b"]), case["lower"], case["upper"])
        assert feasible == bool(case["feasible"]), index
        np.testing.assert_allclose(solution, vector(case["solution"]), atol=1e-9, err_msg=str(index))


def test_correct_motion(data, scene):
    options = default_options()
    for index, case in enumerate(data["CorrectMotion"]):
        acceleration, info = correct_motion(case["position"], case["velocity"], case["nominal"], case["previous"],
                                            scene.box_min, scene.box_max, options, rows(case["measured"]))
        assert info.Feasible == bool(case["feasible"]), index
        np.testing.assert_allclose(acceleration, vector(case["acceleration"]), atol=1e-9, err_msg=str(index))
        assert math.isclose(info.MinimumH, case["minimumH"], abs_tol=1e-12)
        assert math.isclose(info.MinimumPsi1, case["minimumPsi1"], abs_tol=1e-12)
        assert math.isclose(info.MaximumViolation, case["maximumViolation"], rel_tol=1e-9, abs_tol=1e-9)


# --- planner, schedule, lidar ---------------------------------------------------------

def test_plan_path(data, scene):
    for case in data["PlanPath"]:
        options = validate_options({"StartState": case["start"], "GoalState": case["goal"]})
        plan = plan_path(scene, options)
        np.testing.assert_allclose(plan.WaypointsNedM, rows(case["waypoints"]), atol=1e-9)
        np.testing.assert_allclose(plan.RawWaypointsNedM, rows(case["rawWaypoints"]), atol=1e-9)
        assert math.isclose(plan.LengthM, case["lengthM"], rel_tol=1e-12)
        assert plan.ExpandedNodes == case["expandedNodes"]


def test_random_stream(data):
    draws = data["RandStream"]
    stream = MatlabMT19937(draws["seed"])
    np.testing.assert_array_equal([stream.rand() for _ in range(10)], draws["rand"])
    stream = MatlabMT19937(draws["seed"])
    imax = [1, 2, 3, 7, 10, 11, 20, 100, 1, 5]
    assert [stream.randi(value) for value in imax] == list(vector(draws["randi"]).astype(int))


def test_outage_schedule(data):
    for case in data["OutageSchedule"]:
        overrides = dict(case["overrides"] or {})
        overrides["Seed"] = case["seed"]
        if "OutageIntervalsS" in overrides:
            overrides["OutageIntervalsS"] = rows(overrides["OutageIntervalsS"], 2)
        times, enabled, intervals = outage_schedule(validate_options(overrides))
        np.testing.assert_allclose(times, vector(case["times"]), atol=1e-12, err_msg=str(overrides))
        np.testing.assert_array_equal(enabled, vector(case["enabled"]), err_msg=str(overrides))
        np.testing.assert_allclose(intervals, rows(case["intervals"], 2), atol=1e-12, err_msg=str(overrides))


def test_virtual_lidar(data, scene):
    for case in data["VirtualLidar"]:
        _, ranges, hits = virtual_lidar(scene, case["position"], case["eulerZYX"])
        np.testing.assert_array_equal(hits, np.asarray(case["hits"], dtype=bool))
        np.testing.assert_allclose(ranges, vector(case["ranges"]), atol=1e-9)


# --- the full controller ------------------------------------------------------------

def test_controller_trace(data, scene):
    recorded = data["Controller"]
    options = options_from_matlab(recorded["options"])
    controller = MapCorrectionController(scene, options, rows(recorded["route"]))
    seen = replace(scene,
                   box_min=np.vstack((scene.box_min, rows(recorded["extraBoxMin"]))),
                   box_max=np.vstack((scene.box_max, rows(recorded["extraBoxMax"]))))
    for tick in recorded["trace"]:
        points, _, _ = virtual_lidar(seen, tick["position"], tick["eulerZYX"])
        out = controller.step(points, tick["position"], tick["velocity"], tick["eulerZYX"],
                              bool(tick["enabled"]), tick["time"])
        label = f"t={tick['time']:.2f}"
        assert out.info.Feasible == bool(tick["feasible"]), label
        np.testing.assert_allclose(out.acceleration_ned, vector(tick["acceleration"]), atol=1e-7, err_msg=label)
        np.testing.assert_allclose(out.nominal_ned, vector(tick["nominal"]), atol=1e-7, err_msg=label)
        np.testing.assert_allclose(out.reference_ned, vector(tick["reference"]), atol=1e-9, err_msg=label)
        np.testing.assert_allclose(out.command, vector(tick["command"]), atol=1e-5, err_msg=label)
        assert out.waypoint_index == int(tick["waypointIndex"]), label
        assert out.goal_reached == bool(tick["goal"]) and out.stop == bool(tick["stop"]), label
