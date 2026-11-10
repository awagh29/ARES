"""Real-flight corpus and causal fault augmentation for two-stage training.

The production builder accepts the verified 80-run ARES export, never synthetic
trajectories or labels. Whole-run splits are frozen before augmentation. Scene
geometry is simulator ground truth; do not feed ``Corpus.scene`` to an online
policy, state encoder, learned barrier, or calibration model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_BATCH_ID = "c3eaab07-d882-4825-98a5-84c278cf8935"
DEFAULT_PROFILES = ("mixed", "healthy")
TARGET_NAMES = tuple([f"sector_{i:02d}_distance_m" for i in range(7)]
                     + [f"sector_{i:02d}_return_fraction" for i in range(7)])
CONTEXT_NAMES = (
    "delta_north_since_lidar_m", "delta_east_since_lidar_m", "delta_down_since_lidar_m",
    "velocity_north_mps", "velocity_east_mps", "velocity_down_mps",
    "sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "sin_yaw", "cos_yaw",
    "body_roll_rate_radps", "body_pitch_rate_radps", "body_yaw_rate_radps",
    "lidar_available", "outage_age_s",
)
FEATURE_NAMES = TARGET_NAMES + CONTEXT_NAMES
STATE_FEATURE_NAMES = ("position_north_m", "position_east_m", "position_down_m") + FEATURE_NAMES


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed, run_id, profile, variant):
    value = f"{int(seed)}:{run_id}:{profile}:{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


@dataclass
class Trajectory:
    """One measured trajectory; initial_state is [p,v,rpy,omega,F=0].

    Interior thrust state F was not recorded and is deliberately not invented.
    The pretraining encoder state is instead [absolute_position,current31].
    """
    run_id: str
    run_index: int
    axis: str
    split: str
    position: np.ndarray
    velocity: np.ndarray
    rpy: np.ndarray
    omega: np.ndarray
    clean_lidar: np.ndarray
    commands: np.ndarray
    times: np.ndarray
    initial_state: np.ndarray
    goal: np.ndarray
    source_npz_sha256: str
    mission_success: bool
    safety_pass: bool
    collision: bool

    @property
    def contains_returns(self):
        return np.any(self.clean_lidar[:, 7:] > 0, axis=1)

    @property
    def state12(self):
        return np.column_stack((self.position, self.velocity, self.rpy, self.omega))


@dataclass
class Corpus:
    trajectories: list[Trajectory]
    scene: dict
    splits: dict[str, list[str]]
    records: list[dict]
    diagnostics: dict
    batch_id: str
    seed: int

    def by_split(self, split):
        _require(split in self.splits, f"Unknown split: {split}")
        return [item for item in self.trajectories if item.split == split]

    @property
    def feature_names(self):
        return FEATURE_NAMES

    @property
    def target_names(self):
        return TARGET_NAMES


def split_run_records(records, train_fraction=0.7, val_fraction=0.15, seed=42):
    """Freeze whole IDs, stratified by axis and starting-position range.

    Five contiguous start ranges per axis are used when counts permit. Exact
    axis-wide quotas use largest remainders (80 runs -> 56/12/12 by default).
    Assignment never inspects returns, safety outcomes, or a model score.
    """
    fractions = np.asarray([train_fraction, val_fraction,
                            1 - train_fraction - val_fraction], dtype=float)
    _require(np.isfinite(fractions).all() and (fractions > 0).all(),
             "Train, validation and test fractions must be positive and sum to one.")
    _require(len({item["run_id"] for item in records}) == len(records), "Run IDs must be unique.")
    rng = np.random.default_rng(seed)
    names = ("train", "validation", "test")
    assignments, strata = {}, {}
    axes = sorted({item["axis"] for item in records})
    _require(axes == ["East", "North"], "Expected both North and East trajectory groups.")
    for axis in axes:
        ordered = sorted((item for item in records if item["axis"] == axis),
                         key=lambda item: (float(item["axis_start_m"]), item["run_id"]))
        exact = len(ordered) * fractions
        quota = np.floor(exact).astype(int)
        quota[np.argsort(-(exact - quota), kind="stable")[:len(ordered) - quota.sum()]] += 1
        _require((quota > 0).all(), "Each axis needs enough trajectories for all three splits.")
        bins = np.array_split(np.arange(len(ordered)), min(5, int(quota.min())))
        allocation = np.floor(np.asarray([len(group) for group in bins])[:, None] * fractions).astype(int)
        remaining = quota - allocation.sum(axis=0)
        # Reserve at least one item from each range in each fold when possible.
        # Rounding remainder is allocated against fixed quotas, never target data.
        for bin_index in rng.permutation(len(bins)):
            free = len(bins[bin_index]) - allocation[bin_index].sum()
            for _ in range(int(free)):
                candidates = np.flatnonzero(remaining > 0)
                ideal = len(bins[bin_index]) * fractions - allocation[bin_index]
                choice = candidates[np.argmax(ideal[candidates])]
                allocation[bin_index, choice] += 1
                remaining[choice] -= 1
        _require(not remaining.any(), "Internal trajectory split allocation failed.")
        for bin_index, group in enumerate(bins):
            shuffled = rng.permutation(group)
            start = 0
            for fold, number in zip(names, allocation[bin_index]):
                for index in shuffled[start:start + number]:
                    run_id = ordered[int(index)]["run_id"]
                    assignments[run_id] = fold
                    strata[run_id] = f"{axis}:start_range_{bin_index}"
                start += number
    return assignments, strata


def _targets(ranges, valid):
    _require(ranges.ndim == 2 and ranges.shape[1] == 10740 and valid.shape == ranges.shape,
             "Expected recorded [T,10740] ranges and matching validity mask.")
    _require(np.isfinite(ranges).all() and ((ranges >= 0) & (ranges <= 7)).all()
             and (ranges[valid] > 0).all() and (ranges[~valid] == 7).all(),
             "Recorded clean range/validity encoding is invalid.")
    grid = ranges.reshape(len(ranges), 716, 15).transpose(0, 2, 1)
    mask = valid.reshape(len(valid), 716, 15).transpose(0, 2, 1)
    bearing, edges = -np.linspace(-179., 179., 716), np.linspace(-60., 60., 8)
    output = np.empty((len(ranges), 14), dtype=np.float32)
    for sector in range(7):
        columns = (bearing >= edges[sector]) & (bearing <= edges[sector + 1]
            if sector == 6 else bearing < edges[sector + 1])
        hits = mask[:, :, columns]
        distance = np.where(hits, grid[:, :, columns], np.inf).min(axis=(1, 2))
        distance[~np.isfinite(distance)] = 7.
        output[:, sector] = distance
        output[:, sector + 7] = hits.mean(axis=(1, 2))
    return output


def _scene(manifest):
    static = manifest["static"]
    params = static["lidar_effective_block_parameters"]
    centers = np.asarray(static["obstacle_centers_ned_m"], dtype=np.float64)
    enu = np.asarray(static["obstacle_centers_enu_m"], dtype=np.float64)
    _require(centers.shape == enu.shape and centers.shape[1] == 2
             and np.array_equal(centers, enu[:, [1, 0]]), "Obstacle ENU/NED geometry differs.")
    width, height = float(static["obstacle_width_m"]), float(static["obstacle_height_m"])
    _require(width > 0 and height > 0 and str(params["HasNoise"]["value"]).lower() == "off",
             "This physical simulator requires positive boxes and the recorded noise-off sensor.")
    _require(params["AzimuthLimits"]["value"] == [-179, 179]
             and params["ElevationLimits"]["value"] == [-15, 15]
             and params["AzimuthResolution"]["value"] == .5
             and params["ElevationResolution"]["value"] == 2
             and params["MaxRange"]["value"] == 7
             and params["SampleTime"]["value"] == .1, "Recorded sensor geometry changed.")
    centers = np.unique(centers, axis=0)
    minimum = np.column_stack((centers - width / 2, np.full(len(centers), -height)))
    maximum = np.column_stack((centers + width / 2, np.zeros(len(centers))))
    return {
        "schema_version": "ares-static-box-lidar-scene-v1",
        "box_min_ned_m": minimum.tolist(), "box_max_ned_m": maximum.tolist(),
        "obstacle_centers_ned_m": centers.tolist(), "box_width_m": width, "box_height_m": height,
        "azimuth_limits_deg": [-179., 179.], "azimuth_count": 716,
        "elevation_limits_deg": [-15., 15.], "elevation_count": 15,
        "mounting_translation_body_m": [0., 0., -.4],
        "sensor_to_body_rotation": [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        "sector_edges_body_deg": np.linspace(-60., 60., 8).tolist(),
        "maximum_range_m": 7., "sample_period_s": .1, "has_noise": False,
        "vehicle_radius_m": float(manifest["configuration"]["vehicle"]["radiusM"]),
        "geometry_usage": "simulator truth and evaluation only; never online policy/barrier input",
        "ground_mesh_present": False,
    }


def build_dataset(repo_root, batch_id=DEFAULT_BATCH_ID, train_fraction=.7,
                  val_fraction=.15, seed=42, *, cache_path=None):
    """Load and verify all 80 actual episodes, freezing three whole-run folds.

    Optional ``cache_path`` writes a new compact NPZ with clean14/state arrays,
    run offsets, scene and split provenance. Existing caches are never replaced.
    The corpus itself remains in memory; the full beam arrays are streamed out.
    """
    from colab_episode_starter import EpisodeDataset

    root = Path(repo_root).expanduser().resolve()
    dataset = EpisodeDataset(root, str(batch_id), verify_hashes=True, expected_runs=80)
    records, manifests = [], {}
    for row in dataset.runs.itertuples():
        path = root / "data" / "raw" / row.RunId / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        _require(str(manifest["runId"]) == row.RunId, "Raw manifest run identity differs from index.")
        start = np.asarray(manifest["configuration"]["mission"]["initialPositionNed"], dtype=float)
        _require(start.shape == (3,) and np.isfinite(start).all(), "Invalid trajectory starting position.")
        records.append({"run_id": row.RunId, "run_index": int(row.RunIndex), "axis": row.Axis,
            "axis_start_m": float(start[0 if row.Axis == "North" else 1]),
            "source_npz_sha256": dataset._reports[row.RunId]["npzSha256"],
            "source_manifest_sha256": _sha256(path), "mission_success": bool(row.MissionSuccess),
            "safety_pass": bool(row.SafetyChecksPassed), "collision": bool(row.CollisionOccurred)})
        manifests[row.RunId] = manifest
    assignment, strata = split_run_records(records, train_fraction, val_fraction, seed)
    scene = _scene(manifests[records[0]["run_id"]])
    trajectories = []
    wanted = ("lidar_ranges_m", "lidar_valid_return_mask", "lidar_availability_mask",
        "lidar_missing_mask", "lidar_enabled", "position_ned_m", "velocity_ned_mps",
        "attitude_ned_rpy_rad", "angular_velocity_body_radps", "action_current")
    for record in records:
        run_id = record["run_id"]
        manifest = manifests[run_id]
        _require(_scene(manifest) == scene, "All trajectories must share the same recorded physical scene.")
        episode = dataset.load_episode(run_id, fields=wanted)
        _require(episode["lidar_enabled"].all() and episode["lidar_availability_mask"].all()
                 and not episode["lidar_missing_mask"].any(), "Pretraining source must be measured healthy LiDAR.")
        clean = _targets(episode["lidar_ranges_m"], episode["lidar_valid_return_mask"].astype(bool))
        position, velocity, rpy, omega, commands = [np.asarray(episode[name], dtype=np.float32)
            for name in ("position_ned_m", "velocity_ned_mps", "attitude_ned_rpy_rad",
                         "angular_velocity_body_radps", "action_current")]
        _require(all(np.isfinite(value).all() for value in (position, velocity, rpy, omega, commands)),
                 "Native state or command contains unavailable values.")
        initial = np.concatenate((position[0], velocity[0], rpy[0], omega[0], np.zeros(1, np.float32)))
        _require(np.allclose(initial[3:], 0, rtol=0, atol=1e-7),
                 "Initial velocity/attitude/angular rate must match the known zero-state reset.")
        goal = np.asarray(manifest["configuration"]["mission"]["waypointsNed"][-1], dtype=np.float32)
        _require(np.array_equal(goal, np.array([30, 35, -7], np.float32)), "Mission goal differs from corpus contract.")
        record.update(split=assignment[run_id], start_range_stratum=strata[run_id],
                      scan_count=len(clean), contains_returns=bool(np.any(clean[:, 7:] > 0)),
                      initial_state=initial.tolist(), goal=goal.tolist())
        trajectories.append(Trajectory(run_id, record["run_index"], record["axis"], assignment[run_id],
            position, velocity, rpy, omega, clean, commands,
            np.asarray(episode["time_s"], dtype=np.float64), initial, goal,
            record["source_npz_sha256"], record["mission_success"], record["safety_pass"], record["collision"]))
    splits = {name: [item.run_id for item in trajectories if item.split == name]
              for name in ("train", "validation", "test")}
    diagnostics = {name: _coverage([item for item in trajectories if item.split == name]) for name in splits}
    diagnostics["split_policy"] = "whole run IDs; axis/start-range stratified; no target/outcome-driven split search"
    diagnostics["feature_encoder_excludes"] = ["goal", "commands", "scene_geometry", "clean_hidden_lidar"]
    corpus = Corpus(trajectories, scene, splits, records, diagnostics, str(batch_id), int(seed))
    if cache_path is not None:
        _write_cache(corpus, cache_path)
    return corpus


def _coverage(items):
    values = np.concatenate([item.clean_lidar for item in items])
    return {"trajectory_count": len(items), "axis_counts": {axis: sum(item.axis == axis for item in items)
            for axis in ("North", "East")}, "scan_count": len(values),
        "scans_with_returns": int(np.any(values[:, 7:] > 0, axis=1).sum()),
        "trajectories_with_returns": sum(bool(item.contains_returns.any()) for item in items),
        "constant_target_columns": [name for name, span in zip(TARGET_NAMES, np.ptp(values, axis=0)) if span == 0],
        "coverage_is_diagnostic_not_split_selection": True}


def _write_cache(corpus, path):
    path = Path(path)
    _require(not path.exists(), f"Cache already exists and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": "ares-safe-real-corpus-v1", "batch_id": corpus.batch_id,
        "seed": corpus.seed, "scene": corpus.scene, "runs": corpus.records,
        "splits": corpus.splits, "diagnostics": corpus.diagnostics,
        "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES}
    arrays = {name: np.concatenate([getattr(item, name) for item in corpus.trajectories])
              for name in ("position", "velocity", "rpy", "omega", "clean_lidar", "commands", "times")}
    arrays["trajectory_offsets"] = np.concatenate(([0], np.cumsum([len(item.times) for item in corpus.trajectories])))
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        _require(not path.exists(), "Cache destination appeared while exporting.")
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _profile_inputs(trajectory, profile, seed, variant=0):
    _require(profile in ("mixed", "short", "long", "partial", "healthy"), "Unknown fault augmentation profile.")
    count = len(trajectory.times)
    dt = float(np.median(np.diff(trajectory.times)))
    _require(np.allclose(np.diff(trajectory.times), .1, rtol=0, atol=1e-6), "Expected 10 Hz trajectory samples.")
    rng = np.random.default_rng(_seed(seed, trajectory.run_id, profile, variant))
    observed = np.ones((count, 31), dtype=bool)
    intervals = []
    modes = ["short", "long", "partial"] if profile == "mixed" else ([] if profile == "healthy" else [profile])
    minimum = {"short": 15, "long": 50, "partial": 15}
    maximum = {"short": 30, "long": 70, "partial": 30}
    if modes:
        required = sum(minimum[mode] for mode in modes) + 10 * (len(modes) - 1) + 2
        _require(count >= required, f"Trajectory is too short for the {profile} profile.")
        start = int(rng.integers(1, min(30, count - required + 1) + 1))
        for index, mode in enumerate(modes):
            remaining = sum(minimum[next_mode] for next_mode in modes[index + 1:])
            remaining += 10 * (len(modes) - index - 1) + 1
            upper = min(maximum[mode], count - start - remaining)
            _require(upper >= minimum[mode], "Fault profile does not fit the trajectory.")
            length = int(rng.integers(minimum[mode], upper + 1))
            end = start + length
            sectors = np.arange(7) if mode != "partial" else np.sort(rng.choice(7, int(rng.integers(1, 7)), replace=False))
            hidden = np.concatenate((sectors, sectors + 7))
            observed[start:end, hidden] = False
            intervals.append({"mode": mode, "start_row": start, "end_row_exclusive": end,
                              "duration_s": length * dt, "hidden_target_indices": hidden.tolist()})
            if index + 1 < len(modes):
                # Keep enough recovery and room for every remaining mode.
                max_gap = min(20, count - end - sum(minimum[m] for m in modes[index + 1:])
                              - 10 * (len(modes) - index - 2) - 1)
                start = end + int(rng.integers(10, max_gap + 1))
    available = observed[:, :14].all(axis=1)
    last = np.maximum.accumulate(np.where(available, np.arange(count), 0))
    displacement = trajectory.position - trajectory.position[last]
    age = trajectory.times - trajectory.times[last]
    orientation = np.stack((np.sin(trajectory.rpy), np.cos(trajectory.rpy)), axis=-1).reshape(count, 6)
    context = np.column_stack((displacement, trajectory.velocity, orientation, trajectory.omega, available, age))
    features = np.column_stack((trajectory.clean_lidar, context)).astype(np.float32)
    features[~observed] = np.nan
    return features, observed, {"profile": profile, "variant": variant, "intervals": intervals,
        "complete_availability_definition": "all fourteen sector features measured",
        "last_healthy_update": "measured complete scan only; never reconstructed or oracle outage data"}


def _select(corpus, split, run_indices=None):
    items = corpus.by_split(split)
    if run_indices is not None:
        indices = [int(value) for value in run_indices]
        allowed = {item.run_index for item in items}
        _require(len(indices) == len(set(indices)) and set(indices).issubset(allowed),
                 "Requested run indices must be unique and belong to the named outer split.")
        items = [item for item in items if item.run_index in set(indices)]
    _require(bool(items), f"No trajectories selected from split {split}.")
    return items


def make_pretraining_windows(corpus, split, window_size=50, seed=42,
                             profiles=DEFAULT_PROFILES, *, stride=1, run_indices=None):
    """Return real-label NumPy tensors with causal cold-history padding.

    raw_history/observed: [N,50,31]; current_state: [N,34] position+current31.
    target: [N,14] recorded clean labels; target_mask marks hidden endpoint
    features. Healthy samples are retained for an explicit auxiliary/ORT loss.
    No motion feature is masked at real rows; only leading padding masks all.
    """
    _require(isinstance(window_size, int) and window_size > 1, "window_size must be an integer > 1.")
    _require(isinstance(stride, int) and stride > 0, "stride must be a positive integer.")
    profiles = tuple(profiles)
    _require(bool(profiles), "At least one pretraining profile is required.")
    chunks, schedules = [], []
    for trajectory in _select(corpus, split, run_indices):
        for variant, profile in enumerate(profiles):
            features, observed, schedule = _profile_inputs(trajectory, profile, seed, variant)
            count = len(features)
            endpoints = np.arange(0, count, stride)
            padded = np.concatenate((np.full((window_size - 1, 31), np.nan, np.float32), features))
            padded_mask = np.concatenate((np.zeros((window_size - 1, 31), bool), observed))
            history = np.lib.stride_tricks.sliding_window_view(padded, window_size, axis=0).transpose(0, 2, 1)[endpoints].copy()
            history_mask = np.lib.stride_tricks.sliding_window_view(padded_mask, window_size, axis=0).transpose(0, 2, 1)[endpoints].copy()
            current_state = np.column_stack((trajectory.position[endpoints], features[endpoints])).astype(np.float32)
            chunks.append({"raw_history": history, "observed": history_mask,
                "current_state": current_state,
                "current_state_observed": np.column_stack((np.ones((len(endpoints), 3), bool), observed[endpoints])),
                "position": trajectory.position[endpoints].copy(), "target": trajectory.clean_lidar[endpoints].copy(),
                "target_mask": ~observed[endpoints, :14], "clean_label_mask": np.ones((len(endpoints), 14), bool),
                "history_ready": endpoints >= window_size - 1,
                "run_index": np.full(len(endpoints), trajectory.run_index, np.int64),
                "row_index": endpoints.astype(np.int64), "time_s": trajectory.times[endpoints].copy(),
                "contains_returns": trajectory.contains_returns[endpoints],
                "profile_index": np.full(len(endpoints), variant, np.int64)})
            schedules.append({"run_id": trajectory.run_id, "run_index": trajectory.run_index, **schedule})
    _require(bool(chunks), f"No trajectories assigned to split {split}.")
    output = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    output["metadata"] = {"schema_version": "ares-safe-real-pretraining-v1", "split": split,
        "seed": int(seed), "profiles": profiles, "window_size": window_size, "stride": stride,
        "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES,
        "state_feature_names": STATE_FEATURE_NAMES, "schedules": schedules,
        "labels": "recorded clean LiDAR only; synthetic faults alter inputs and masks, never labels",
        "encoder_excluded_inputs": ["goal", "commands", "scene_geometry", "clean_hidden_targets"],
        "scored_examples_with_returns": int(output["contains_returns"].sum()),
        "hidden_target_count": int(output["target_mask"].sum()),
        "cold_history_examples": int((~output["history_ready"]).sum())}
    return output


@dataclass
class FeatureNormalizer:
    position_mean: np.ndarray
    position_std: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray
    training_run_ids: tuple[str, ...]
    fit_row_count: int
    seed: int
    profiles: tuple[str, ...]

    @classmethod
    def fit(cls, corpus, seed=42, profiles=DEFAULT_PROFILES, run_indices=None):
        """One contribution per native row/declared augmentation condition.

        Repeated overlapping windows and validation/test rows are excluded.
        Physical distance/fraction scaling is fixed, not estimated from labels.
        """
        positions, contexts, ids = [], [], []
        profiles = tuple(profiles)
        _require(bool(profiles), "Normalizer needs at least one declared fault profile.")
        for item in _select(corpus, "train", run_indices):
            ids.append(item.run_id)
            for variant, profile in enumerate(profiles):
                features, _, _ = _profile_inputs(item, profile, seed, variant)
                positions.append(item.position.astype(np.float64))
                contexts.append(features[:, 14:].astype(np.float64))
        _require(bool(contexts), "No training trajectories for normalization.")
        position, context = np.concatenate(positions), np.concatenate(contexts)
        _require(np.isfinite(position).all() and np.isfinite(context).all(), "Training motion context must be observed.")
        pstd, cstd = position.std(axis=0), context.std(axis=0)
        pstd[pstd < 1e-8] = 1.; cstd[cstd < 1e-8] = 1.
        return cls(position.mean(axis=0), pstd, context.mean(axis=0), cstd,
                   tuple(ids), len(context), int(seed), profiles)

    @property
    def history_mean(self):
        return np.concatenate((np.zeros(14), self.context_mean)).astype(np.float32)

    @property
    def history_std(self):
        return np.concatenate((np.full(7, 7.), np.ones(7), self.context_std)).astype(np.float32)

    @property
    def state_mean(self):
        return np.concatenate((self.position_mean, self.history_mean)).astype(np.float32)

    @property
    def state_std(self):
        return np.concatenate((self.position_std, self.history_std)).astype(np.float32)

    def transform_history_numpy(self, raw_history, observed=None):
        raw = np.asarray(raw_history, dtype=np.float32)
        mask = np.isfinite(raw) if observed is None else np.asarray(observed, dtype=bool)
        _require(raw.shape[-1] == 31 and mask.shape == raw.shape and np.isfinite(raw[mask]).all(),
                 "History/mask shapes must agree; observed values must be finite.")
        transformed = (np.where(mask, raw, 0) - self.history_mean) / self.history_std
        transformed[~mask] = 0
        return np.ascontiguousarray(transformed, dtype=np.float32)

    def transform_state_numpy(self, current_state, observed=None):
        state = np.asarray(current_state, dtype=np.float32)
        mask = np.isfinite(state) if observed is None else np.asarray(observed, dtype=bool)
        _require(state.shape[-1] == 34 and mask.shape == state.shape and np.isfinite(state[mask]).all(),
                 "Encoder input must be [absolute_position,current31], with a matching observation mask.")
        transformed = (np.where(mask, state, 0) - self.state_mean) / self.state_std
        transformed[~mask] = 0
        return np.ascontiguousarray(transformed, dtype=np.float32)

    @staticmethod
    def transform_targets_numpy(target):
        values = np.asarray(target, dtype=np.float32).copy()
        _require(values.shape[-1] == 14 and np.isfinite(values).all(), "Targets must have fourteen finite features.")
        values[..., :7] /= np.float32(7)
        return values

    @staticmethod
    def inverse_targets_numpy(target):
        values = np.asarray(target, dtype=np.float32).copy()
        _require(values.shape[-1] == 14, "Predictions must have fourteen features.")
        values[..., :7] *= np.float32(7)
        return values

    def to_dict(self):
        return {"schema_version": "ares-safe-feature-normalizer-v1", "feature_names": FEATURE_NAMES,
            "state_feature_names": STATE_FEATURE_NAMES, "target_names": TARGET_NAMES,
            "position_mean": self.position_mean.tolist(), "position_std": self.position_std.tolist(),
            "context_mean": self.context_mean.tolist(), "context_std": self.context_std.tolist(),
            "distance_scale_m": 7., "training_run_ids": self.training_run_ids,
            "normalization_source_run_ids": self.training_run_ids,
            "history_mean": self.history_mean.tolist(), "history_std": self.history_std.tolist(),
            "state_mean": self.state_mean.tolist(), "state_std": self.state_std.tolist(),
            "conditioning": "absolute_position and measured independent motion plus available/reconstructed LiDAR only; no goal, commands or map",
            "fit_row_count": self.fit_row_count, "seed": self.seed, "profiles": self.profiles,
            "fit_unit": "one native row per training trajectory and declared fault condition; no window duplication"}

    @classmethod
    def from_dict(cls, value):
        _require(value.get("schema_version") == "ares-safe-feature-normalizer-v1"
                 and tuple(value["feature_names"]) == FEATURE_NAMES
                 and tuple(value["state_feature_names"]) == STATE_FEATURE_NAMES
                 and value["distance_scale_m"] == 7., "Normalizer schema or feature order differs.")
        arrays = [np.asarray(value[name], dtype=np.float64) for name in
                  ("position_mean", "position_std", "context_mean", "context_std")]
        _require([array.shape for array in arrays] == [(3,), (3,), (17,), (17,)]
                 and all(np.isfinite(array).all() for array in arrays)
                 and (arrays[1] > 0).all() and (arrays[3] > 0).all(), "Invalid normalizer statistics.")
        return cls(*arrays, tuple(value["training_run_ids"]), int(value["fit_row_count"]),
                   int(value["seed"]), tuple(value["profiles"]))


def fit_normalizer(corpus, seed=42, profiles=DEFAULT_PROFILES, run_indices=None):
    return FeatureNormalizer.fit(corpus, seed=seed, profiles=profiles, run_indices=run_indices)
