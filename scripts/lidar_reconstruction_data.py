"""One-episode SAITS preparation: chronological splits and known LiDAR loss.

NumPy/pandas only. Defaults match the ARES 15x716 scan, sensor mount [0,0,180]
intrinsic ZYX degrees, seven body-forward sectors, and 10 Hz telemetry.
These utilities prepare inputs/labels; they do not train a model.
"""
from __future__ import annotations

import copy
import json
from numbers import Integral

import numpy as np
import pandas as pd

GRID = (15, 716)
BEAMS = int(np.prod(GRID))
MAX_RANGE_M = 7.0
LIDAR_COLUMNS = (
    "lidar_ranges_m", "lidar_valid_return_mask",
    "lidar_availability_mask", "lidar_missing_mask",
)
VECTOR_COMPONENTS = {
    "action_current": ("roll_rad", "pitch_rad", "yaw_rate_radps", "thrust_n"),
    "action_previous": ("roll_rad", "pitch_rad", "yaw_rate_radps", "thrust_n"),
    "angular_velocity_body_radps": ("roll", "pitch", "yaw"),
    "attitude_ned_rpy_rad": ("roll", "pitch", "yaw"),
    "position_ned_m": ("north", "east", "down"),
    "velocity_ned_mps": ("north", "east", "down"),
    "goal_position_ned_m": ("north", "east", "down"),
    "reference_desired_position_ned_m": ("north", "east", "down"),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=1):
    _require(isinstance(value, Integral) and not isinstance(value, bool)
             and value >= minimum, f"{name} must be an integer >= {minimum}.")
    return int(value)


def make_table(episode, *, expand_lidar=False) -> pd.DataFrame:
    """The notebook make_table function; default LiDAR columns hold scan arrays."""
    count = len(episode["time_s"])
    blocks, metadata = [], {}
    for name, value in episode.items():
        values = np.asarray(value)
        if values.ndim == 0:
            metadata[name] = values.item()
            continue
        _require(values.shape[0] == count, f"Sample count mismatch: {name}")
        if values.ndim == 1:
            blocks.append(pd.DataFrame({name: values.copy()}))
        elif values.ndim == 2:
            if name.startswith("lidar_") and not expand_lidar:
                blocks.append(pd.DataFrame({name: list(values.copy())}))
            else:
                components = VECTOR_COMPONENTS.get(name, range(values.shape[1]))
                _require(len(components) == values.shape[1], f"Invalid width: {name}")
                blocks.append(pd.DataFrame(values, copy=True,
                    columns=[f"{name}_{component}" for component in components]))
        else:
            raise ValueError(f"Unsupported array shape: {name}: {values.shape}")
    frame = pd.concat(blocks, axis=1)
    if "run_id" in metadata:
        frame.insert(0, "run_id", metadata["run_id"])
    frame.attrs.update(metadata)
    if "metadata_json" in metadata:
        frame.attrs["metadata"] = json.loads(metadata["metadata_json"])
    return frame


def _timeline(frame):
    _require(isinstance(frame, pd.DataFrame) and len(frame) >= 2,
             "Supply an episode DataFrame with at least two rows.")
    _require(frame.index.is_unique and frame.columns.is_unique,
             "DataFrame row labels and column names must be unique.")
    _require("run_id" in frame and frame["run_id"].notna().all()
             and frame["run_id"].nunique() == 1,
             "This function is for one episode; run_id must identify one run.")
    _require("time_s" in frame, "Missing time_s column.")
    time = frame["time_s"].to_numpy(dtype=float)
    _require(np.isfinite(time).all() and np.all(np.diff(time) > 0),
             "time_s must be finite and strictly increasing in row order.")
    step = float(np.median(np.diff(time)))
    _require(np.allclose(np.diff(time), step, rtol=0, atol=1e-6),
             "This experiment requires regularly sampled scans.")
    return time, step


def _copy_frame(frame):
    result = frame.copy(deep=True)
    result.attrs = copy.deepcopy(frame.attrs)
    # pandas deep=True does not copy NumPy arrays nested inside object cells.
    for name in LIDAR_COLUMNS:
        if name in result:
            result[name] = [np.array(value, copy=True) for value in frame[name]]
    return result


def split_train_val_test(df: pd.DataFrame, *, train_size=0.6,
                         val_size=0.2, test_size=0.2):
    """Return chronological (train_df, val_df, test_df), before perturbations.

    302 rows -> 181/60/61. Original times, sample indices and row labels survive.
    Construct all history windows independently inside the returned partitions.
    """
    _timeline(df)
    _require(not df.attrs.get("synthetic_lidar_perturbation", False)
             and "synthetic_lidar_available" not in df,
             "Split the clean episode before generating synthetic outages.")
    ratios = np.asarray([train_size, val_size, test_size], dtype=float)
    _require(np.isfinite(ratios).all() and (ratios > 0).all()
             and np.isclose(ratios.sum(), 1.0, rtol=0, atol=1e-12),
             "Positive train/validation/test fractions must sum to one.")
    train_end = int(np.floor(len(df) * ratios[0]))
    val_end = int(np.floor(len(df) * (ratios[0] + ratios[1])))
    _require(0 < train_end < val_end < len(df), "Each split must contain at least one row.")
    parts = []
    for name, start, end in (("train", 0, train_end),
                              ("validation", train_end, val_end),
                              ("test", val_end, len(df))):
        part = _copy_frame(df.iloc[start:end])
        part.attrs["split_name"] = name
        part.attrs["source_row_interval"] = (start, end)
        parts.append(part)
    return tuple(parts)


def _clean_lidar(frame):
    _require(set(LIDAR_COLUMNS).issubset(frame.columns),
             "Use make_table(episode) with expand_lidar=False; four scan-array columns are required.")
    aliases = [name for name in frame.columns
               if name.startswith(tuple(key + "_" for key in LIDAR_COLUMNS))]
    _require(not aliases, "Remove expanded LiDAR aliases and use the compact DataFrame.")
    arrays = {}
    for name in LIDAR_COLUMNS:
        values = np.stack(frame[name].to_numpy())
        _require(values.shape == (len(frame), BEAMS),
                 f"{name} must contain one {BEAMS}-element vector per row.")
        if name != "lidar_ranges_m":
            _require(np.isin(values, [0, 1]).all(), f"Nonbinary mask: {name}")
            values = values.astype(bool)
        elif values.dtype.kind != "f":
            values = values.astype(float)
        arrays[name] = values
    ranges, valid = arrays["lidar_ranges_m"], arrays["lidar_valid_return_mask"]
    _require(np.isfinite(ranges).all() and ((ranges >= 0) & (ranges <= MAX_RANGE_M)).all()
             and (ranges[valid] > 0).all() and (ranges[~valid] == MAX_RANGE_M).all(),
             "Expected clean finite ranges and maximum-range encoding for no returns.")
    _require(arrays["lidar_availability_mask"].all()
             and not arrays["lidar_missing_mask"].any(),
             "Start from fully available clean LiDAR, not already degraded data.")
    if "lidar_enabled" in frame:
        _require(frame["lidar_enabled"].eq(True).all(), "The source episode must keep LiDAR enabled.")
    return arrays


def _sector_targets(ranges, valid, sectors, half_angle):
    _require(np.isfinite(half_angle) and 0 < half_angle <= 179,
             "forward_half_angle_deg must be between 0 and 179.")
    # MATLAB RayGenerator3D uses floor(span/resolution) points and linspace.
    sensor_azimuth = np.linspace(-179.0, 179.0, GRID[1])
    # Sensor FLU -> body FRD under the recorded 180-degree roll mounting.
    body_bearing = -sensor_azimuth
    edges = np.linspace(-half_angle, half_angle, sectors + 1)
    scan_ranges = ranges.reshape(len(ranges), GRID[1], GRID[0]).transpose(0, 2, 1)
    scan_valid = valid.reshape(len(valid), GRID[1], GRID[0]).transpose(0, 2, 1)
    distances, fractions, beam_counts = [], [], []
    for index in range(sectors):
        columns = (body_bearing >= edges[index]) & (
            body_bearing <= edges[index + 1] if index == sectors - 1
            else body_bearing < edges[index + 1])
        _require(columns.any(), "A sector contains no beams; reduce the sector count.")
        hits = scan_valid[:, :, columns]
        nearest = np.where(hits, scan_ranges[:, :, columns], np.inf).min(axis=(1, 2))
        nearest[~np.isfinite(nearest)] = MAX_RANGE_M
        distances.append(nearest)
        fractions.append(hits.mean(axis=(1, 2)))
        beam_counts.append(int(hits.shape[1] * hits.shape[2]))
    values = np.column_stack(distances + fractions)
    names = ([f"sector_{index:02d}_distance_m" for index in range(sectors)]
             + [f"sector_{index:02d}_return_fraction" for index in range(sectors)])
    return values, names, edges, beam_counts


def _motion(frame, prefix, components):
    names = [f"{prefix}_{component}" for component in components]
    _require(set(names).issubset(frame.columns), f"Missing independent motion columns: {names}")
    values = frame[names].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"Nonfinite independent motion: {prefix}")
    return values


def introduce_lidar_perturbations(
    df: pd.DataFrame, *, seed=42, n_outages=1, duration_s=(1.5, 3.0),
    warmup_s=3.0, min_healthy_gap_s=2.0, window_size=50,
    min_scored_steps=5, n_sectors=7, forward_half_angle_deg=60.0,
):
    """Inject full-scan loss into ONE already split partition; return X/y/masks.

    X: 14 currently observed sector features + 17 independent context features.
    y: 14 clean sector targets, never used as outage inputs.
    Missing LiDAR inputs are NaN; no-return healthy sectors remain [7 m,0].
    Intervals are contiguous, fit entirely inside the partition, and contain
    scored endpoints after enough history for the configured window size.
    Nothing in the original DataFrame or its nested array cells is modified.
    """
    time, dt = _timeline(df)
    _require(not df.attrs.get("synthetic_lidar_perturbation", False)
             and "synthetic_lidar_available" not in df, "Do not perturb an already perturbed frame.")
    count = len(df)
    n_outages = _integer(n_outages, "n_outages", minimum=0)
    seed = _integer(seed, "seed", minimum=0)
    window_size = _integer(window_size, "window_size")
    min_scored_steps = _integer(min_scored_steps, "min_scored_steps")
    n_sectors = _integer(n_sectors, "n_sectors")
    _require(window_size <= count, "This split is shorter than window_size; use larger partitions or a shorter history.")
    duration = np.asarray(duration_s, dtype=float)
    _require(duration.shape == (2,) and np.isfinite(duration).all()
             and 0 < duration[0] <= duration[1], "duration_s must be a positive (minimum, maximum) pair.")
    _require(np.isfinite(warmup_s) and warmup_s >= 0
             and np.isfinite(min_healthy_gap_s) and min_healthy_gap_s >= 0,
             "Healthy warmup and gap durations must be nonnegative.")
    low = int(np.ceil(duration[0] / dt - 1e-8))
    high = int(np.floor(duration[1] / dt + 1e-8))
    _require(1 <= low <= high, "No whole scan interval fits duration_s.")
    _require(n_outages == 0 or min_scored_steps <= low,
             "min_scored_steps must not exceed the shortest allowed outage in scans.")
    warmup = max(1, int(np.ceil(warmup_s / dt - 1e-8)))
    healthy_gap = max(1, int(np.ceil(min_healthy_gap_s / dt - 1e-8)))
    source = _clean_lidar(df)
    truth, target_columns, edges, beam_counts = _sector_targets(
        source["lidar_ranges_m"], source["lidar_valid_return_mask"],
        n_sectors, forward_half_angle_deg)
    position = _motion(df, "position_ned_m", ("north", "east", "down"))
    velocity = _motion(df, "velocity_ned_mps", ("north", "east", "down"))
    attitude = _motion(df, "attitude_ned_rpy_rad", ("roll", "pitch", "yaw"))
    angular = _motion(df, "angular_velocity_body_radps", ("roll", "pitch", "yaw"))
    outage = np.zeros(count, dtype=bool)
    intervals = []
    rng = np.random.default_rng(seed)
    if n_outages:
        lengths = rng.integers(low, high + 1, size=n_outages)
        first_start = max(warmup, window_size - 1 + min_scored_steps - int(lengths[0]))
        required = first_start + int(lengths.sum()) + healthy_gap * (n_outages - 1)
        _require(required <= count,
                 "Outages/history do not fit this split; reduce n_outages or window_size.")
        # Randomly place slack before, between, and after complete intervals.
        extra = rng.multinomial(count - required, np.full(n_outages + 1, 1 / (n_outages + 1)))
        start = first_start + int(extra[0])
        for number, length in enumerate(lengths):
            end = start + int(length)
            outage[start:end] = True
            intervals.append({"start_row": start, "end_row_exclusive": end,
                              "start_time_s": float(time[start]),
                              "end_time_s": float(time[end - 1] + dt),
                              "duration_s": float(length * dt)})
            start = end + healthy_gap + int(extra[number + 1])
    available = ~outage
    age, displacement = np.zeros(count), np.zeros((count, 3))
    persistence = np.empty_like(truth)
    last_healthy = 0
    for row in range(count):
        if available[row]:
            last_healthy = row
        age[row] = time[row] - time[last_healthy]
        displacement[row] = position[row] - position[last_healthy]
        persistence[row] = truth[last_healthy]
    observed_targets = truth.copy()
    observed_targets[outage] = np.nan
    orientation = np.column_stack([
        value for axis in range(3)
        for value in (np.sin(attitude[:, axis]), np.cos(attitude[:, axis]))])
    context = np.column_stack([displacement, velocity, orientation, angular, available, age])
    context_columns = (
        [f"delta_{axis}_since_lidar_m" for axis in ("north", "east", "down")]
        + [f"velocity_{axis}_mps" for axis in ("north", "east", "down")]
        + [f"{function}_{axis}" for axis in ("roll", "pitch", "yaw") for function in ("sin", "cos")]
        + [f"body_{axis}_rate_radps" for axis in ("roll", "pitch", "yaw")]
        + ["lidar_available", "outage_age_s"])
    X = pd.DataFrame(np.column_stack([observed_targets, context]), index=df.index,
                     columns=target_columns + context_columns)
    y = pd.DataFrame(truth, index=df.index, columns=target_columns)
    target_mask = pd.DataFrame(np.repeat(outage[:, None], len(target_columns), axis=1),
                               index=df.index, columns=target_columns)
    score_mask = target_mask.copy()
    score_mask.iloc[:window_size - 1] = False
    if n_outages:
        first_interval = intervals[0]
        scored_first = int(score_mask.iloc[
            first_interval["start_row"]:first_interval["end_row_exclusive"], 0].sum())
        _require(scored_first >= min_scored_steps,
                 "The requested number of scored outage steps could not be generated.")
    observed = _copy_frame(df)
    for name in LIDAR_COLUMNS:
        values = source[name].copy()
        values[outage] = (np.nan if name == "lidar_ranges_m"
                          else name == "lidar_missing_mask")
        observed[name] = list(values)
    observed["synthetic_lidar_available"] = available
    observed["synthetic_outage_age_s"] = age
    observed.attrs["synthetic_lidar_perturbation"] = True
    metadata = {"seed": int(seed), "n_outages": n_outages, "duration_s": duration.tolist(),
                "warmup_s": float(warmup_s), "min_healthy_gap_s": float(min_healthy_gap_s),
                "scan_period_s": dt, "n_sectors": n_sectors,
                "forward_half_angle_deg": float(forward_half_angle_deg),
                "window_size": window_size, "min_scored_steps": min_scored_steps,
                "sector_body_bearing_edges_deg": edges.tolist(),
                "sector_beam_counts": beam_counts, "sensor_mount_zyx_deg": [0, 0, 180],
                "target_units": ["m"] * n_sectors + ["fraction"] * n_sectors,
                "target_count": len(target_columns), "input_count": X.shape[1],
                "perturbation_type": "known-complete-lidar-outage",
                "split_name": df.attrs.get("split_name", "unspecified"),
                "raw_lidar_enabled_is_provenance": True,
                "window_endpoint_sampling": "at-least-min_scored_steps-per-first-outage"}
    scored = score_mask.any(axis=1).to_numpy()
    scored_truth = truth[scored]
    metadata["scored_endpoint_count"] = int(scored.sum())
    metadata["scored_endpoints_with_returns"] = int(
        np.count_nonzero(np.any(scored_truth[:, n_sectors:] > 0, axis=1)))
    metadata["distinct_scored_target_rows"] = int(len(np.unique(scored_truth, axis=0)))
    metadata["scored_target_content"] = (
        "no-scored-targets" if not scored.any() else
        "no-return-only" if not np.any(scored_truth[:, n_sectors:] > 0) else
        "contains-returns")
    metadata["constant_scored_target_columns"] = (
        [name for name, spread in zip(target_columns, np.ptp(scored_truth, axis=0)) if spread == 0]
        if scored.any() else [])
    observed.attrs["synthetic_perturbation_configuration"] = copy.deepcopy(metadata)
    return {"observed_df": observed, "X": X, "y": y,
            "observed_mask": X.notna(), "target_mask": target_mask, "score_mask": score_mask,
            "outage_mask": pd.Series(outage, index=df.index, name="outage_active"),
            "outage_intervals": pd.DataFrame(intervals, columns=[
                "start_row", "end_row_exclusive", "start_time_s", "end_time_s", "duration_s"]),
            "target_columns": target_columns, "predictor_columns": list(X.columns),
            "window_size": window_size,
            "persistence_y": pd.DataFrame(persistence, index=df.index, columns=target_columns),
            "metadata": metadata}


def make_saits_windows(prepared):
    """Past-only windows, one scored outage endpoint per example.

    Pass only X/observed_mask to model inference. y and target_mask are labels.
    SAITS imputed output is [N,window,features]; score [:,-1,target_feature_indices].
    Do not score earlier rows in a window, since attention can see their future.
    """
    size = prepared["window_size"]
    endpoints = np.flatnonzero(prepared["score_mask"].any(axis=1).to_numpy())
    _require(len(endpoints) > 0, "There are no scored outage endpoints with the requested history.")
    values = prepared["X"].to_numpy(copy=True)
    windows = np.stack([values[end - size + 1:end + 1] for end in endpoints])
    time = prepared["observed_df"]["time_s"].to_numpy()
    return {"X": windows, "observed_mask": ~np.isnan(windows),
            "y": prepared["y"].iloc[endpoints].to_numpy(copy=True),
            "target_mask": prepared["target_mask"].iloc[endpoints].to_numpy(copy=True),
            "target_times_s": time[endpoints].copy(), "source_row_positions": endpoints,
            "feature_names": list(prepared["X"].columns),
            "target_columns": list(prepared["target_columns"]),
            "target_feature_indices": np.arange(len(prepared["target_columns"])),
            "persistence_y": prepared["persistence_y"].iloc[endpoints].to_numpy(copy=True)}
