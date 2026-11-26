"""Unit tests for the controller-side pieces (no MATLAB or Gazebo needed)."""

from __future__ import annotations

import math

import numpy as np

from ares_gz.controllers.pid import DiscretePID, PidCascade, yaw_rate_command
from ares_gz.scenarios import load_matrix


def test_discrete_pid_matches_forward_euler_difference_equation():
    p, d, n, ts = 6.0, 1.5, 10.0, 0.01
    pid = DiscretePID(p, 0.0, d, n, limit=1e9, ts=ts)
    errors = [1.0, 0.8, 0.5, 0.5, -0.2]
    s = 0.0
    for e in errors:
        expected = p * e + d * n * (e - s)  # filter state starts at 0 (Simulink IC)
        assert math.isclose(pid.step(e), expected, rel_tol=1e-12)
        s += ts * n * (e - s)


def test_pid_output_saturates():
    pid = DiscretePID(6.0, 0.0, 1.5, 10.0, limit=3.0, ts=0.01)
    assert pid.step(10.0) == 3.0
    assert pid.step(-10.0) == -3.0


def test_cascade_limits_match_the_model():
    cascade = PidCascade(0.01)
    acceleration, velocity_command = cascade.step([100.0, -100.0, 100.0], [0, 0, 0], [0, 0, 0])
    assert np.allclose(np.abs(velocity_command), 3.0)
    assert np.allclose(np.abs(acceleration), [1.0, 1.0, 4.0])


def test_yaw_law_is_unwrapped_gain_two_with_half_radian_limit():
    assert math.isclose(yaw_rate_command(0.1, 0.0), 0.2)
    assert yaw_rate_command(3.0, -3.0) == 0.5  # no angle wrapping, as in the Simulink Sum block
    assert yaw_rate_command(float("nan"), 0.1) == -0.2  # NaN desired yaw -> 0


def test_two_start_points_at_the_same_altitude_and_clear_of_buildings():
    from ares_gz.metrics import signed_box_distance
    from ares_gz.scene import scene_map

    matrix = load_matrix()
    assert len(matrix.starts) == 2
    altitude = matrix.mission["StartState"][2]
    scene = scene_map()
    for start in matrix.starts:
        state = start.state(altitude)
        assert state[2] == -7 and (state[0], state[1]) != (0.0, 0.0)
        assert -5 <= state[0] <= 40 and -5 <= state[1] <= 40  # CNN training domain
        assert signed_box_distance(np.array(state[:3]), scene)[0] - 0.1 >= 1.0
    assert len({(s.north, s.east) for s in matrix.starts}) == 2


def test_videos_are_sorted_by_experiment_and_pass_fail(tmp_path):
    import json

    from ares_gz.replay import organize_videos, video_category

    assert video_category("S1_intermittent") == "1_intermittent_lidar_failure"
    assert video_category("S5_gusts_medium_blackout") == "5_wind_gusts_plus_complete_blackout"
    runs = [("S2_blackout", "W1", 1, False), ("S4_gusts_light_intermittent", "W2", 3, False),
            ("S0_baseline", "W2", 1, True), ("S0_baseline", "W2", 11, False)]
    videos = tmp_path / "videos"
    videos.mkdir()
    for scenario, start, seed, success in runs:
        run = tmp_path / scenario / start / f"seed{seed:04d}_abc"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps(
            {"scenario": scenario, "start_id": start, "seed": seed, "mission_success": success}))
        (videos / f"{scenario}_{start}_seed{seed}_overview.mp4").write_bytes(b"x")
    assert organize_videos(videos, tmp_path) == 4
    assert (videos / "2_complete_lidar_blackout" / "FAILED" / "S2_blackout_W1_seed1_overview.mp4").exists()
    assert (videos / "4_wind_gusts_plus_intermittent_lidar" / "FAILED" /
            "S4_gusts_light_intermittent_W2_seed3_overview.mp4").exists()
    assert (videos / "0_control_healthy_lidar" / "PASSED" / "S0_baseline_W2_seed1_overview.mp4").exists()
    # seed1 must not be confused with seed11
    assert (videos / "0_control_healthy_lidar" / "FAILED" / "S0_baseline_W2_seed11_overview.mp4").exists()
    assert not list(videos.glob("*.mp4"))
    assert organize_videos(videos, tmp_path) == 0  # idempotent


def test_matrix_names_the_controller_and_model():
    matrix = load_matrix()
    assert matrix.controller["name"] in ("map-temporal-cnn", "map-correction")
    assert matrix.controller["model_path"].endswith(".pt") or matrix.controller["model_path"].endswith(".mat")
