"""Behavioral checks for one-flight splitting and synthetic LiDAR outages.

Run with: python -m unittest discover -s tests/python -v
The fixtures are synthetic compact DataFrames; no simulation files are read.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from lidar_reconstruction_data import (
    introduce_lidar_perturbations,
    make_saits_windows,
    split_train_val_test,
)


GRID_SIZE = (15, 716)
BEAM_COUNT = 10740
DENSE_COLUMNS = (
    "lidar_ranges_m", "lidar_valid_return_mask", "lidar_availability_mask", "lidar_missing_mask",
)
DISTANCE_COLUMNS = [f"sector_{index:02d}_distance_m" for index in range(7)]
FRACTION_COLUMNS = [f"sector_{index:02d}_return_fraction" for index in range(7)]
TARGET_COLUMNS = DISTANCE_COLUMNS + FRACTION_COLUMNS


def compact_frame(count=302, *, start_time=0.0, no_returns=False):
    """Independent cell arrays expose accidental writes to the clean source."""
    time = start_time + np.arange(count, dtype=float) * 0.1
    ranges = np.full(BEAM_COUNT, 7.0 if no_returns else 3.0)
    valid = np.full(BEAM_COUNT, not no_returns, dtype=bool)
    data = {
        "run_id": ["single-fixture-run"] * count,
        "time_s": time,
        "lidar_ranges_m": [ranges.copy() for _ in range(count)],
        "lidar_valid_return_mask": [valid.copy() for _ in range(count)],
        "lidar_availability_mask": [np.ones(BEAM_COUNT, dtype=bool) for _ in range(count)],
        "lidar_missing_mask": [np.zeros(BEAM_COUNT, dtype=bool) for _ in range(count)],
        "lidar_enabled": np.ones(count, dtype=bool),
    }
    for prefix, components in (
        ("position_ned_m", ("north", "east", "down")),
        ("velocity_ned_mps", ("north", "east", "down")),
        ("attitude_ned_rpy_rad", ("roll", "pitch", "yaw")),
        ("angular_velocity_body_radps", ("roll", "pitch", "yaw")),
    ):
        for index, component in enumerate(components):
            data[f"{prefix}_{component}"] = (index + 1) * 0.01 * np.arange(count)
    data["position_ned_m_down"] = np.full(count, -7.0)
    # These healthy-policy outputs must never enter reconstruction predictors.
    data["action_current_roll_rad"] = 1_000_000.0 + time
    data["action_previous_roll_rad"] = -1_000_000.0 - time
    data["reference_desired_position_ned_m_north"] = 9_000_000.0 + time
    data["obstacle_avoidance_status"] = np.arange(count) % 4
    return pd.DataFrame(data)


def contiguous_segments(mask):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not indices.size:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


class TestChronologicalFlightSplit(unittest.TestCase):
    def test_splits_one302_row_flight_before_corruption(self):
        source = compact_frame()
        train, validation, test = split_train_val_test(source)
        self.assertEqual([len(part) for part in (train, validation, test)], [181, 60, 61])
        np.testing.assert_array_equal(train.time_s, source.time_s.iloc[:181])
        np.testing.assert_array_equal(validation.time_s, source.time_s.iloc[181:241])
        np.testing.assert_array_equal(test.time_s, source.time_s.iloc[241:])
        self.assertLess(train.time_s.max(), validation.time_s.min())
        self.assertLess(validation.time_s.max(), test.time_s.min())
        self.assertEqual(sum(map(len, (train, validation, test))), len(source))

    def test_rejects_multiple_flights_and_nonmonotone_time(self):
        source = compact_frame(30)
        source.loc[15:, "run_id"] = "other-flight"
        with self.assertRaises(ValueError):
            split_train_val_test(source)
        source = compact_frame(30)
        source.loc[10, "time_s"] = source.loc[9, "time_s"] - 0.1
        with self.assertRaises(ValueError):
            split_train_val_test(source)

    def test_rejects_perturb_then_split_and_invalid_proportions(self):
        source = compact_frame(30)
        source.attrs["synthetic_lidar_perturbation"] = {"seed": 42}
        with self.assertRaises(ValueError):
            split_train_val_test(source)
        source.attrs.clear()
        for fractions in ((0.8, 0.2, 0.2), (-0.1, 0.5, 0.6), (1.0, 0.0, 0.0)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                split_train_val_test(source, train_size=fractions[0],
                                     val_size=fractions[1], test_size=fractions[2])


class TestSyntheticOutages(unittest.TestCase):
    def test_shapes_masks_and_separation_of_truth(self):
        source = compact_frame()
        result = introduce_lidar_perturbations(source)
        self.assertEqual(result["X"].shape, (302, 31))
        self.assertEqual(result["y"].shape, (302, 14))
        self.assertEqual(list(result["target_columns"]), TARGET_COLUMNS)
        self.assertEqual(list(result["y"].columns), TARGET_COLUMNS)
        outage = np.asarray(result["outage_mask"], dtype=bool)
        self.assertGreaterEqual(np.count_nonzero(outage[49:]), 5)
        expected_target_mask = np.broadcast_to(outage[:, None], (302, 14))
        np.testing.assert_array_equal(np.asarray(result["target_mask"]), expected_target_mask)
        expected_score_mask = expected_target_mask.copy()
        expected_score_mask[:49] = False
        np.testing.assert_array_equal(np.asarray(result["score_mask"]), expected_score_mask)
        self.assertEqual(np.asarray(result["target_mask"]).dtype, np.dtype(bool))
        self.assertEqual(np.asarray(result["score_mask"]).dtype, np.dtype(bool))
        self.assertTrue(np.isnan(result["X"].iloc[outage, :14].to_numpy()).all())
        self.assertTrue(np.isfinite(result["X"].iloc[~outage, :14].to_numpy()).all())
        self.assertTrue(np.isfinite(result["X"].iloc[:, 14:].to_numpy()).all())
        self.assertTrue(np.isfinite(result["y"].to_numpy()).all())
        np.testing.assert_allclose(result["y"].iloc[:, :7], 3.0)
        np.testing.assert_allclose(result["y"].iloc[:, 7:], 1.0)

    def test_seeded_bursts_fit_and_keep_healthy_gaps(self):
        source = compact_frame()
        first = introduce_lidar_perturbations(source, seed=513, n_outages=2,
                                              duration_s=(1.5, 3.0))
        second = introduce_lidar_perturbations(source, seed=513, n_outages=2,
                                               duration_s=(1.5, 3.0))
        np.testing.assert_array_equal(first["outage_mask"], second["outage_mask"])
        pd.testing.assert_frame_equal(first["X"], second["X"])
        segments = contiguous_segments(first["outage_mask"])
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(first["outage_intervals"]), 2)
        for segment in segments:
            self.assertGreaterEqual(len(segment) * 0.1, 1.5 - 1e-9)
            self.assertLessEqual(len(segment) * 0.1, 3.0 + 1e-9)
            self.assertGreaterEqual(source.time_s.iloc[segment[0]], 1.0 - 1e-9)
            self.assertLessEqual(segment[-1], len(source) - 1)
        self.assertGreaterEqual((segments[1][0] - segments[0][-1] - 1) * 0.1,
                                0.5 - 1e-9)

    def test_corruption_does_not_mutate_cell_array_truth(self):
        source = compact_frame()
        source_snapshot = {name: np.stack(source[name]) for name in DENSE_COLUMNS}
        result = introduce_lidar_perturbations(source)
        observed = result["observed_df"]
        outage = np.asarray(result["outage_mask"], dtype=bool)
        first_missing = int(np.flatnonzero(outage)[0])
        for name in DENSE_COLUMNS:
            np.testing.assert_array_equal(np.stack(source[name]), source_snapshot[name])
            self.assertFalse(np.shares_memory(observed[name].iloc[first_missing],
                                             source[name].iloc[first_missing]))
        for row in np.flatnonzero(outage):
            self.assertTrue(np.isnan(observed.lidar_ranges_m.iloc[row]).all())
            self.assertFalse(observed.lidar_valid_return_mask.iloc[row].any())
            self.assertFalse(observed.lidar_availability_mask.iloc[row].any())
            self.assertTrue(observed.lidar_missing_mask.iloc[row].all())
        np.testing.assert_array_equal(observed.lidar_enabled, source.lidar_enabled)
        np.testing.assert_array_equal(observed.synthetic_lidar_available, ~outage)
        observed.lidar_ranges_m.iloc[first_missing][0] = 999.0
        self.assertEqual(source.lidar_ranges_m.iloc[first_missing][0], 3.0)
        self.assertEqual(result["y"].iloc[first_missing, 0], 3.0)

    def test_no_return_is_observed_healthy_data_not_an_outage(self):
        source = compact_frame(no_returns=True)
        clean = introduce_lidar_perturbations(source, n_outages=0)
        self.assertFalse(np.asarray(clean["outage_mask"]).any())
        self.assertFalse(np.asarray(clean["target_mask"]).any())
        self.assertFalse(np.asarray(clean["score_mask"]).any())
        self.assertTrue(np.isfinite(clean["X"].iloc[:, :14].to_numpy()).all())
        np.testing.assert_allclose(clean["y"].iloc[:, :7], 7.0)
        np.testing.assert_allclose(clean["y"].iloc[:, 7:], 0.0)
        corrupted = introduce_lidar_perturbations(source)
        outage = np.asarray(corrupted["outage_mask"], dtype=bool)
        self.assertTrue(np.isnan(corrupted["X"].iloc[outage, :14].to_numpy()).all())
        pd.testing.assert_frame_equal(corrupted["y"], clean["y"])
        self.assertEqual(corrupted["metadata"]["scored_target_content"], "no-return-only")
        self.assertEqual(corrupted["metadata"]["scored_endpoints_with_returns"], 0)
        self.assertEqual(len(corrupted["metadata"]["constant_scored_target_columns"]), 14)

    def test_excludes_healthy_policy_outputs_from_predictors(self):
        source = compact_frame()
        first = introduce_lidar_perturbations(source)
        for name in ("action_current_roll_rad", "action_previous_roll_rad",
                     "reference_desired_position_ned_m_north", "obstacle_avoidance_status"):
            source[name] = np.arange(len(source)) * -987654321.0
        second = introduce_lidar_perturbations(source)
        pd.testing.assert_frame_equal(first["X"], second["X"])
        pd.testing.assert_frame_equal(first["y"], second["y"])
        for name in first["X"].columns:
            self.assertFalse(name.lower().startswith(("action_", "reference_", "obstacle_avoidance")))

    def test_sectors_follow_body_bearing_and_matlab_channel_order(self):
        source = compact_frame(12)
        sensor_azimuth = np.linspace(-179.0, 179.0, 716)
        body_bearing = -sensor_azimuth
        edges = np.linspace(-60.0, 60.0, 8)
        ranges = np.full(GRID_SIZE, 7.0)
        valid = np.zeros(GRID_SIZE, dtype=bool)
        for sector in range(7):
            columns = ((body_bearing >= edges[sector]) &
                       (body_bearing < edges[sector + 1]))
            if sector == 6:
                columns |= body_bearing == edges[-1]
            ranges[:, columns] = 1.0 + 0.5 * sector
            valid[:, columns] = True
        for row in range(len(source)):
            source.at[row, "lidar_ranges_m"] = ranges.reshape(-1, order="F").copy()
            source.at[row, "lidar_valid_return_mask"] = valid.reshape(-1, order="F").copy()
        result = introduce_lidar_perturbations(source, n_outages=0, window_size=5)
        np.testing.assert_allclose(result["y"].iloc[:, :7],
                                   np.broadcast_to(1.0 + 0.5 * np.arange(7), (12, 7)))
        np.testing.assert_allclose(result["y"].iloc[:, 7:], 1.0)
        self.assertEqual(list(result["metadata"]["sector_beam_counts"]),
                         [510, 525, 510, 510, 510, 525, 510])

    def test_rejects_infeasible_outages_and_duplicate_time(self):
        source = compact_frame(60)
        with self.assertRaisesRegex(ValueError, "min_scored_steps"):
            introduce_lidar_perturbations(source, duration_s=(0.5, 0.5),
                                         window_size=10, min_scored_steps=10, warmup_s=0)
        with self.assertRaises(ValueError):
            introduce_lidar_perturbations(source, n_outages=10, duration_s=(2.0, 3.0))
        source.loc[10, "time_s"] = source.loc[9, "time_s"]
        with self.assertRaises(ValueError):
            introduce_lidar_perturbations(source, n_outages=0)


class TestCausalSaitsWindows(unittest.TestCase):
    def test_windows_end_at_masked_target_and_contain_only_past_rows(self):
        prepared = introduce_lidar_perturbations(compact_frame())
        windows = make_saits_windows(prepared)
        positions = np.asarray(windows["source_row_positions"])
        expected_positions = np.flatnonzero(np.asarray(prepared["score_mask"]).any(axis=1))
        np.testing.assert_array_equal(positions, expected_positions)
        self.assertGreaterEqual(len(positions), 5)
        self.assertEqual(windows["X"].shape, (len(positions), 50, 31))
        self.assertEqual(windows["y"].shape, (len(positions), 14))
        self.assertEqual(windows["observed_mask"].dtype, np.dtype(bool))
        np.testing.assert_array_equal(windows["observed_mask"], ~np.isnan(windows["X"]))
        for index, endpoint in enumerate(positions):
            expected = prepared["X"].iloc[endpoint - 49:endpoint + 1].to_numpy()
            np.testing.assert_array_equal(windows["X"][index], expected)
            np.testing.assert_array_equal(windows["y"][index], prepared["y"].iloc[endpoint])
            self.assertEqual(windows["target_times_s"][index],
                             prepared["observed_df"].time_s.iloc[endpoint])
        np.testing.assert_array_equal(windows["target_feature_indices"], np.arange(14))

    def test_future_changes_cannot_change_an_earlier_prediction_window(self):
        source = compact_frame()
        prepared = introduce_lidar_perturbations(source)
        windows = make_saits_windows(prepared)
        endpoint = int(windows["source_row_positions"][0])
        self.assertLess(endpoint, len(source) - 1)
        for row in range(endpoint + 1, len(source)):
            source.at[row, "lidar_ranges_m"] = np.full(BEAM_COUNT, 0.25)
            source.at[row, "lidar_valid_return_mask"] = np.ones(BEAM_COUNT, dtype=bool)
        source.loc[endpoint + 1:, "position_ned_m_north"] += 10000
        changed = make_saits_windows(introduce_lidar_perturbations(source))
        np.testing.assert_array_equal(changed["source_row_positions"], windows["source_row_positions"])
        np.testing.assert_array_equal(changed["X"][0], windows["X"][0])
        np.testing.assert_array_equal(changed["y"][0], windows["y"][0])

    def test_partition_windows_never_borrow_training_rows(self):
        source = compact_frame()
        train, validation, test = split_train_val_test(source)
        training_last_time = train.time_s.iloc[-1]
        for part in (validation, test):
            prepared = introduce_lidar_perturbations(
                part, window_size=10, warmup_s=0.5, duration_s=(1.0, 1.5),
                min_scored_steps=3)
            windows = make_saits_windows(prepared)
            self.assertGreaterEqual(len(windows["source_row_positions"]), 3)
            self.assertTrue((np.asarray(windows["source_row_positions"]) >= 9).all())
            self.assertTrue((np.asarray(windows["target_times_s"]) > training_last_time).all())
            for index, endpoint in enumerate(windows["source_row_positions"]):
                np.testing.assert_array_equal(windows["X"][index],
                    prepared["X"].iloc[endpoint - 9:endpoint + 1].to_numpy())


if __name__ == "__main__":
    unittest.main()
