"""Actual recorded-data pretraining and calibrated closed-loop fine-tuning.

No recorded future state/scan is used as a rollout transition. Only the
simulator's sensor and loss functions receive scene geometry. Test IDs never
enter fitting, empirical calibration, checkpoint selection, or early stopping.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .barrier import BarrierConfig, BarrierSafetyFilter
from .config import PipelineConfig
from .control import desired_yaw_rate, nominal_acceleration, smooth_reference
from .data import FeatureNormalizer, make_pretraining_windows
from .dynamics import GuidanceDynamics, load_calibration
from .geometry import TorchLidarScene
from .models import AugmentedSAITS
from validate_dynamics_calibration import verify_validation_report

SCHEMA = "ares-augmented-saits-barrier-training-v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(cpu_tree(payload), temporary)
    os.replace(temporary, path)


def source_signature():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths.append(Path(__file__).parents[1] / "saits_endpoint_model.py")
    return {path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
            for path in sorted(paths)}


def model_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": [numpy[0], torch.tensor(numpy[1].astype(np.int64)),
                      int(numpy[2]), int(numpy[3]), float(numpy[4])]}


def restore_rng(value):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])
    state = value["numpy"]
    np.random.set_state((state[0], state[1].cpu().numpy().astype(np.uint32), *state[2:]))


def device_for(config):
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab; CPU must be selected explicitly for verification.")
    return torch.device(config.device)


def amp_context(device, config):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == "cuda" and config.amp)


def experiment_metadata(corpus, config, calibration_path):
    """Reserve calibration inside TRAIN without creating a fourth outer fold."""
    rng = np.random.default_rng(config.seed + 917)
    calibration_indices, train_indices = [], []
    for axis in ("North", "East"):
        items = sorted((item for item in corpus.by_split("train") if item.axis == axis),
                       key=lambda item: item.run_index)
        shuffled = rng.permutation(len(items))
        count = max(1, int(round(len(items) * config.calibration_fraction_of_train)))
        count = min(count, len(items) - 1)
        calibration_indices.extend(items[index].run_index for index in shuffled[:count])
        train_indices.extend(items[index].run_index for index in shuffled[count:])
    if not train_indices or not calibration_indices:
        raise ValueError("Training needs disjoint fitting and empirical-calibration trajectories.")
    hashes = {item.run_id: item.source_npz_sha256 for item in corpus.trajectories}
    calibration = load_calibration(calibration_path)
    numerical_validation = verify_validation_report(calibration_path)
    return {"schema_version": SCHEMA, "configuration": config.to_dict(),
            "outer_splits": corpus.splits, "training_run_indices": sorted(train_indices),
            "calibration_run_indices_within_train": sorted(calibration_indices),
            "source_episode_sha256": hashes, "source_signature": source_signature(),
            "source_manifest_sha256": {item["run_id"]: item["source_manifest_sha256"] for item in corpus.records},
            "dynamics_calibration_sha256": calibration["calibration_sha256"],
            "dynamics_numerical_validation": numerical_validation,
            "dataset_diagnostics": corpus.diagnostics, "simulator_scene": corpus.scene,
            "online_obstacle_map": False, "safety_certified": False,
            "bounds_provenance": "empirical calibration and explicit software envelope; not a hardware certificate",
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None}


def immutable_signature(metadata):
    value = {name: metadata[name] for name in ("configuration", "outer_splits", "training_run_indices",
             "calibration_run_indices_within_train", "source_episode_sha256", "source_signature",
             "source_manifest_sha256", "simulator_scene", "dynamics_calibration_sha256")}
    # Epoch limits can be extended on resume; optimization/geometry stays fixed.
    value = json.loads(json.dumps(value))
    for name in ("pretrain_epochs", "finetune_epochs", "log_every", "checkpoint_every"):
        value["configuration"].pop(name)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class TensorNormalizer(nn.Module):
    def __init__(self, normalizer, device):
        super().__init__()
        for name in ("history_mean", "history_std", "state_mean", "state_std"):
            self.register_buffer(name, torch.as_tensor(getattr(normalizer, name), device=device))

    def history(self, values, mask):
        safe = torch.where(mask, values, torch.zeros_like(values))
        return ((safe - self.history_mean) / self.history_std).masked_fill(~mask, 0)

    def state(self, values, mask):
        safe = torch.where(mask, values, torch.zeros_like(values))
        return ((safe - self.state_mean) / self.state_std).masked_fill(~mask, 0)


def prepare_pack(corpus, split, normalizer, config, device, run_indices=None):
    windows = make_pretraining_windows(corpus, split, config.model.window_size, config.seed,
                                       run_indices=run_indices)
    arrays = {
        "history": normalizer.transform_history_numpy(windows["raw_history"], windows["observed"]),
        "history_mask": windows["observed"],
        "state": normalizer.transform_state_numpy(windows["current_state"], windows["current_state_observed"]),
        "state_mask": windows["current_state_observed"],
        "target": normalizer.transform_targets_numpy(windows["target"]),
        "target_mask": windows["target_mask"],
    }
    byte_count = sum(array.nbytes for array in arrays.values())
    if device.type == "cuda" and byte_count > torch.cuda.get_device_properties(device).total_memory // 3:
        raise MemoryError("Window tensors exceed the GPU budget; increase stride or implement streaming before training.")
    tensors = {name: torch.from_numpy(np.ascontiguousarray(array)).to(device) for name, array in arrays.items()}
    return {"tensors": tensors, "windows": windows, "count": len(windows["target"]),
            "gpu_resident_bytes": byte_count}


def reconstruction_loss(output, target, hidden, config, example_weights=None):
    weights = torch.where(hidden, torch.ones_like(target),
                          torch.full_like(target, config.observed_reconstruction_weight))
    if example_weights is not None:
        weights = weights * example_weights[:, None]
    def term(prediction):
        errors = F.smooth_l1_loss(prediction.float(), target.float(), beta=.1, reduction="none")
        return (errors * weights).sum() / weights.sum().clamp_min(1)
    return (term(output["normalized"]) + config.auxiliary_reconstruction_weight * .5
            * (term(output["stage1"]) + term(output["stage2"])))


def _forward(model, data, index, device, config):
    with amp_context(device, config):
        return model(data["history"][index], data["history_mask"][index],
                     data["state"][index], data["state_mask"][index])


@torch.inference_mode()
def evaluate_reconstruction(model, pack, config, device, save_path=None):
    model.eval()
    predictions = []
    for start in range(0, pack["count"], config.pretrain_batch_size):
        output = _forward(model, pack["tensors"], slice(start, start + config.pretrain_batch_size), device, config)
        predictions.append(output["physical"].float().cpu().numpy())
    prediction = np.concatenate(predictions)
    target = pack["windows"]["target"]
    mask = pack["windows"]["target_mask"]
    distance, fraction = np.abs(prediction[:, :7] - target[:, :7]), np.abs(prediction[:, 7:] - target[:, 7:])
    missing_distance, missing_fraction = mask[:, :7], mask[:, 7:]
    if not missing_distance.any() or not missing_fraction.any():
        raise ValueError("Evaluation must include actual hidden targets.")
    metrics = {"distance_mae_m": float(distance[missing_distance].mean()),
               "return_fraction_mae": float(fraction[missing_fraction].mean()),
               "selection_mae_normalized": float(np.abs(prediction / np.array([7.] * 7 + [1.] * 7)
                     - target / np.array([7.] * 7 + [1.] * 7))[mask].mean()),
               "examples": len(target), "hidden_target_count": int(mask.sum()),
               "hidden_examples_with_returns": int((mask.any(1) & (target[:, 7:] > 0).any(1)).sum()),
               "finite_predictions": bool(np.isfinite(prediction).all())}
    if not metrics["finite_predictions"]:
        raise FloatingPointError("Nonfinite reconstruction prediction.")
    seen = pack["windows"]["observed"][:, :, :14]
    last = seen.shape[1] - 1 - seen[:, ::-1].argmax(axis=1)
    baseline = np.take_along_axis(pack["windows"]["raw_history"][:, :, :14], last[:, None], axis=1)[:, 0]
    baseline = np.where(seen.any(axis=1), baseline, np.array([7.] * 7 + [0.] * 7))
    metrics.update({"persistence_distance_mae_m": float(np.abs(baseline[:, :7] - target[:, :7])[missing_distance].mean()),
                    "persistence_return_fraction_mae": float(np.abs(baseline[:, 7:] - target[:, 7:])[missing_fraction].mean()),
                    "persistence_scope": "most recent observed feature within the same window; no-return prior when none exists"})
    if save_path is not None:
        np.savez_compressed(save_path, prediction=prediction, target=target, target_mask=mask,
                            run_index=pack["windows"]["run_index"], row_index=pack["windows"]["row_index"],
                            time_s=pack["windows"]["time_s"])
    return metrics, prediction


def make_optimizer(parameters, learning_rate, config, device):
    kwargs = {"lr": learning_rate, "weight_decay": config.weight_decay}
    if device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (TypeError, NotImplementedError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def _restore_stage(path, stage, signature, model, optimizer, scheduler, scaler, barrier=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != SCHEMA or checkpoint["stage"] != stage or checkpoint["signature"] != signature:
        raise ValueError("Resume source/data/calibration/configuration changed; use a new experiment directory.")
    model.load_state_dict(checkpoint["model"])
    if barrier is not None:
        barrier.load_state_dict(checkpoint["barrier"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"])
    return checkpoint


def train_stage_one(model, train_pack, validation_pack, normalizer, output_dir, config, signature, *, resume=False):
    """No calibration or test labels enter gradient fitting/model selection."""
    device = next(model.parameters()).device
    folder = Path(output_dir) / "stage1"
    folder.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(model.parameters(), config.pretrain_learning_rate, config, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and config.amp)
    history, best, stale, start = [], math.inf, 0, 1
    runtime_sources = source_signature()
    last = folder / "last.pt"
    if last.exists():
        if not resume:
            raise FileExistsError("Stage 1 exists; resume explicitly or select a new output directory.")
        old = _restore_stage(last, "reconstruction", signature, model, optimizer, scheduler, scaler)
        if config.pretrain_epochs < old["epoch"]:
            raise ValueError("Cannot reduce the epoch limit below a saved training epoch.")
        history, best, stale, start = old["history"], old["best_score"], old["stale"], old["epoch"] + 1
    for epoch in range(start, config.pretrain_epochs + 1):
        if stale >= config.patience:
            break
        started = time.perf_counter()
        model.train()
        order = torch.randperm(train_pack["count"], device=device)
        losses, count = 0., 0
        for offset in range(0, len(order), config.pretrain_batch_size):
            indices = order[offset:offset + config.pretrain_batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, train_pack["tensors"], indices, device, config)
            loss = reconstruction_loss(output, train_pack["tensors"]["target"][indices],
                                       train_pack["tensors"]["target_mask"][indices], config)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite Stage-1 loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm, error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            losses += float(loss.detach()) * len(indices)
            count += len(indices)
        metrics, _ = evaluate_reconstruction(model, validation_pack, config, device)
        score = metrics["selection_mae_normalized"]
        scheduler.step(score)
        improved = score < best - 1e-6
        best, stale = (score, 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch, "loss": losses / count, "validation": metrics,
                        "seconds": time.perf_counter() - started})
        payload = {"schema_version": SCHEMA, "stage": "reconstruction", "signature": signature,
                   "epoch": epoch, "model": model.state_dict(), "normalizer": normalizer.to_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "rng": rng_state(), "history": history,
                   "best_score": best, "stale": stale, "model_config": asdict(config.model)}
        payload["source_signature"] = runtime_sources
        if improved:
            save_checkpoint(folder / "best.pt", dict(payload, checkpoint_role="best", selected_weights=True))
        save_checkpoint(last, dict(payload, checkpoint_role="last", selected_weights=False))
        write_json(folder / "history.json", history)
        if epoch == 1 or epoch % config.log_every == 0:
            print(f"Stage 1 epoch {epoch}: loss={losses/count:.5f}, validation={score:.5f}", flush=True)
    if not (folder / "best.pt").exists():
        raise RuntimeError("No completed Stage-1 checkpoint exists.")
    selected = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"])
    return selected


@torch.inference_mode()
def calibrate_reconstruction(model, calibration_pack, config, device):
    """Empirical one-sided errors on reserved TRAIN trajectories, never test."""
    metrics, prediction = evaluate_reconstruction(model, calibration_pack, config, device)
    target = calibration_pack["windows"]["target"]
    hidden = calibration_pack["windows"]["target_mask"][:, :7]
    error = np.maximum(prediction[:, :7] - target[:, :7], 0)
    bounds, counts = [], []
    for sector in range(7):
        sample = error[hidden[:, sector], sector]
        if not len(sample):
            raise ValueError("Each sector requires hidden calibration examples.")
        estimate = np.quantile(sample, config.uncertainty_quantile, method="higher")
        bounds.append(float(max(config.minimum_range_bound_m, estimate + config.uncertainty_margin_m)))
        counts.append(int(len(sample)))
    return {"schema_version": "ares-empirical-reconstruction-bound-v1", "bounds_m": bounds,
            "model_state_sha256": model_state_sha256(model.state_dict()),
            "one_sided_error": "max(predicted_distance-clean_distance,0)",
            "quantile": config.uncertainty_quantile, "engineering_margin_m": config.uncertainty_margin_m,
            "sample_counts": counts, "reconstruction_metrics": metrics,
            "source_run_indices": sorted(set(calibration_pack["windows"]["run_index"].tolist())),
            "certified": False, "provenance": "finite route-level training calibration; not a uniform error certificate"}


def rollout_fault_plan(trajectories, steps, dt, seed, *, training=False):
    """Fixed evaluation faults, independent resampled training faults.

    An intact initial scan is retained. Long failures can remove every healthy
    LiDAR sample from history. Masks alter observations, never simulator truth.
    """
    plan = np.ones((len(trajectories), steps, 14), dtype=bool)
    schedules = []
    for row, trajectory in enumerate(trajectories):
        key = f"{seed}:{trajectory.run_id}:closed-loop".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        start = max(1, int(round(5 / dt)))
        intervals = []
        number = 0
        while start < steps:
            long = training and rng.random() < .25
            length = int(rng.integers(50, 71) if long else rng.integers(15, 31))
            end = min(steps, start + length)
            partial = training and not long and rng.random() < .25
            sectors = np.sort(rng.choice(7, 3, replace=False)) if partial else np.arange(7)
            indices = np.concatenate((sectors, sectors + 7))
            plan[row, start:end, indices] = False
            intervals.append({"start_step": start, "end_step_exclusive": end,
                              "hidden_sectors": sectors.tolist(), "long_outage": bool(long)})
            start = end + int(round(2 / dt))
            number += 1
        schedules.append({"run_id": trajectory.run_id, "intervals": intervals})
    return plan, schedules


def _current_features(state, clean_lidar, measured, last_position, last_time, time_s):
    available = measured.all(dim=1)
    last_position = torch.where(available[:, None], state[:, :3], last_position)
    now = state.new_full((len(state),), time_s)
    last_time = torch.where(available, now, last_time)
    orientation = torch.stack((state[:, 6:9].sin(), state[:, 6:9].cos()), dim=-1).reshape(len(state), 6)
    raw = torch.cat((torch.where(measured, clean_lidar, torch.zeros_like(clean_lidar)),
                     state[:, :3] - last_position, state[:, 3:6], orientation, state[:, 9:12],
                     available.float()[:, None], (now - last_time)[:, None]), dim=1)
    mask = torch.cat((measured, torch.ones(len(state), 17, dtype=torch.bool, device=state.device)), dim=1)
    current = torch.cat((state[:, :3], raw), dim=1)
    current_mask = torch.cat((torch.ones(len(state), 3, dtype=torch.bool, device=state.device), mask), dim=1)
    return raw, mask, current, current_mask, last_position, last_time


def run_rollout(model, barrier, dynamics, scene, trajectories, normalizer, config, *,
                optimizer=None, scaler=None, fault_seed=42, steps=None, mask_plan=None, collect_traces=False):
    """Closed loop with actual 13-state transitions and native-grid clearance.

    Training uses truncated BPTT, updating between chunks and detaching their
    initial conditions. Evaluation freezes weights for an entire trajectory.
    Failed filters terminate trajectories explicitly; computationally frozen
    inactive batch rows are not counted as simulated safe flight or completion.
    """
    device = next(model.parameters()).device
    training = optimizer is not None
    model.train(training)
    barrier.train(training)
    count = len(trajectories)
    total_steps = config.rollout_steps if steps is None else int(steps)
    initial = torch.as_tensor(np.stack([item.initial_state for item in trajectories]), device=device)
    goals = torch.as_tensor(np.stack([item.goal for item in trajectories]), device=device)
    state = initial.clone()
    initial_position = initial[:, :3].clone()
    history = torch.zeros(count, config.model.window_size, 31, device=device)
    history_mask = torch.zeros_like(history, dtype=torch.bool)
    last_position, last_time = state[:, :3].clone(), torch.zeros(count, device=device)
    previous_acceleration = torch.zeros(count, 2, device=device)
    previous_vertical = torch.zeros(count, device=device)
    active = torch.ones(count, dtype=torch.bool, device=device)
    initial_clearance = scene.clearance(state[:, :3]).detach()
    initial_truth_clearance_ok = initial_clearance >= barrier.config.clearance_m
    valid_initial = initial_truth_clearance_ok.clone()
    initial_cbf_valid = torch.zeros(count, dtype=torch.bool, device=device)
    initial_qp_feasible = torch.zeros_like(initial_cbf_valid)
    active &= valid_initial
    plan, schedules = rollout_fault_plan(trajectories, total_steps, config.control_dt, fault_seed, training=training)
    if mask_plan is not None:
        if np.asarray(mask_plan).shape != plan.shape:
            raise ValueError("Explicit fault plan shape differs from the rollout.")
        plan = np.asarray(mask_plan, dtype=bool)
    measured_plan = torch.as_tensor(plan, device=device)
    tensor_normalizer = TensorNormalizer(normalizer, device)
    scale = state.new_tensor([7.] * 7 + [1.] * 7)
    executed = torch.zeros(count, dtype=torch.int64, device=device)
    failures = ~valid_initial
    collisions = initial_clearance <= 0
    near_collision = initial_clearance < barrier.config.clearance_m
    altitude_failure = torch.zeros_like(collisions)
    outage_steps = torch.zeros(count, dtype=torch.int64, device=device)
    correction_total = torch.zeros(count, device=device)
    minimum_clearance = scene.clearance(state[:, :3]).detach()
    loss_total, loss_count = 0., 0
    traces = {name: [] for name in ("time_s", "state", "command", "nominal_acceleration",
                                  "filtered_acceleration", "prediction", "clean_lidar", "measured_mask",
                                  "feasible", "active", "minimum_clearance_m", "max_qp_violation", "k2")}
    context = nullcontext() if training else torch.no_grad()
    with context:
        for chunk_start in range(0, total_steps, config.gradient_horizon):
            if not bool(active.any().item()):
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            chunk_losses = []
            for step in range(chunk_start, min(total_steps, chunk_start + config.gradient_horizon)):
                if not bool(active.any().item()):
                    break
                time_s = step * config.control_dt
                clean = scene.features(state)  # truth belongs to the simulator, never the barrier
                measured = measured_plan[:, step]
                raw, mask, current, current_mask, last_position, last_time = _current_features(
                    state, clean, measured, last_position, last_time, time_s)
                history = torch.cat((history[:, 1:], raw[:, None]), dim=1)
                history_mask = torch.cat((history_mask[:, 1:], mask[:, None]), dim=1)
                with amp_context(device, config):
                    output = model(tensor_normalizer.history(history, history_mask), history_mask,
                                   tensor_normalizer.state(current, current_mask), current_mask)
                prediction = output["physical"].float()
                perceived = torch.where(measured, clean, prediction)
                reference = smooth_reference(initial_position, goals, time_s,
                                             config.reference_speed_mps, config.tracking_tube_radius_m)
                nominal = nominal_acceleration(state, perceived, reference, config)
                # Zero reconstruction uncertainty on measured sectors. Physical
                # sensor/beam/latency margins remain in the barrier definition.
                distance_known = measured[:, :7] & measured[:, 7:]
                uncertainty = torch.where(distance_known, torch.zeros(count, 7, device=device, dtype=torch.float64),
                                          barrier.calibration_bounds[None].expand(count, 7))
                filtered = barrier(state[:, :3], state[:, 3:6], state[:, 6:9], perceived,
                                   reference, previous_acceleration, config.control_dt, uncertainty,
                                   nominal_acceleration=nominal[:, :2])
                supervision_active = active.clone()
                if step == 0:
                    # Enroll only starts satisfying h>=0, psi1>=0, angular
                    # validity and a feasible hard QP. Invalid starts retain
                    # reconstruction supervision but execute zero plant steps.
                    initial_cbf_valid = filtered["cbf_valid"].detach()
                    initial_qp_feasible = filtered["feasible"].detach()
                    valid_initial &= initial_cbf_valid & initial_qp_feasible
                    failures |= ~valid_initial
                    active &= valid_initial
                horizontal = filtered["acceleration"].float()
                # This is an upstream software limit on the separate vertical
                # controller, matching the measured calibration ramp envelope.
                vertical = torch.maximum(previous_vertical - config.control_dt,
                                         torch.minimum(previous_vertical + config.control_dt, nominal[:, 2]))
                acceleration = torch.cat((horizontal, vertical[:, None]), dim=1)
                yaw_rate = desired_yaw_rate(state, goals, config.maximum_yaw_rate_radps)
                command = dynamics.acceleration_to_commands(acceleration, state[:, 8], yaw_rate)
                params = dynamics.parameters_dict
                command_valid = ((command[:, 0].abs() <= params["roll_limit_rad"] + 1e-6)
                    & (command[:, 1].abs() <= params["pitch_limit_rad"] + 1e-6)
                    & (command[:, 2].abs() <= params["yaw_rate_limit_radps"] + 1e-6)
                    & (command[:, 3] >= 0) & torch.isfinite(command).all(1))
                if not bool(command_valid.all().item()):
                    raise FloatingPointError("Allocation violates the declared attitude/yaw/thrust software envelope.")
                next_state, native_positions = dynamics(state.float(), command.float(), config.control_dt, return_trace=True)
                if not bool(torch.isfinite(next_state).all().item()):
                    raise FloatingPointError("The calibrated plant produced a nonfinite state.")
                clearance = scene.clearance(native_positions)
                smallest = clearance.amin(dim=1)
                reference_next = smooth_reference(initial_position, goals, time_s + config.control_dt,
                                                  config.reference_speed_mps, config.tracking_tube_radius_m)
                # Supervision is simulator truth at THIS rolled pose, never a
                # recorded future scan. Labels are detached external targets.
                rec = reconstruction_loss(output, (clean / scale).detach(), ~measured, config, supervision_active.float())
                tracking = ((next_state[:, :3] - reference_next["p"])**2).mean(1) / config.tracking_tube_radius_m**2
                correction = (horizontal - nominal[:, :2]).square().mean(1)
                residual = filtered["true_inequality_violation"].square().mean(1).clamp_max(1e8)
                initial_h = F.relu(-filtered["barrier_values"]).square().mean(1)
                initial_psi = F.relu(-filtered["psi1"]).square().mean(1)
                geometry = F.relu(barrier.config.clearance_m - clearance).square().mean(1)
                altitude = F.relu(5 + native_positions[:, :, 2]).square().mean(1)
                altitude += F.relu(-native_positions[:, :, 2] - 9).square().mean(1)
                weights = active.float()
                denominator = weights.sum().clamp_min(1)
                downstream = (config.tracking_weight * tracking + config.control_correction_weight * correction
                    + config.barrier_violation_weight * (residual + initial_h + initial_psi)
                    + config.geometry_violation_weight * (geometry + altitude)
                    + config.tracking_slack_weight * filtered["tracking_slack"].square().clamp_max(1e8))
                loss = config.reconstruction_weight * rec + (downstream * weights).sum() / denominator
                if step == 0:
                    # Rejected starts execute no motion. Still expose their
                    # actual hard-constraint/initial-condition residuals to
                    # learning: k2 can affect QP feasibility, while fixed
                    # geometry h/psi1 cannot be repaired by changing a model.
                    rejected = (supervision_active & ~active).float()
                    loss = loss + config.barrier_violation_weight * (
                        (residual + initial_h + initial_psi) * rejected).sum() / supervision_active.sum().clamp_min(1)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Nonfinite closed-loop loss; no checkpoint was selected.")
                chunk_losses.append(loss)
                with torch.no_grad():
                    was_active = active.clone()
                    failed_now = was_active & ~filtered["feasible"]
                    failures |= failed_now
                    collisions |= was_active & (smallest <= 0)
                    near_collision |= was_active & (smallest < barrier.config.clearance_m)
                    altitude_failure |= was_active & ((-native_positions[:, :, 2] < 5)
                                                      | (-native_positions[:, :, 2] > 9)).any(1)
                    minimum_clearance = torch.where(was_active, torch.minimum(minimum_clearance, smallest.detach()), minimum_clearance)
                    executed += was_active
                    outage_steps += was_active & ~measured.all(1)
                    correction_total += correction.detach() * was_active
                    trace_values = (("time_s", state.new_full((count,), time_s)), ("state", state),
                        ("command", command), ("nominal_acceleration", nominal),
                        ("filtered_acceleration", acceleration), ("prediction", prediction), ("clean_lidar", clean),
                        ("measured_mask", measured), ("feasible", filtered["feasible"]), ("active", was_active),
                        ("minimum_clearance_m", smallest), ("max_qp_violation", filtered["max_violation"]),
                        ("k2", filtered["k2"]))
                    if collect_traces:
                        for name, value in trace_values:
                            traces[name].append(value.detach())
                    active = was_active & filtered["feasible"] & (smallest > 0) & ~altitude_failure
                state = torch.where(was_active[:, None], next_state, state)
                previous_acceleration = horizontal
                previous_vertical = vertical
            if not chunk_losses:
                break
            chunk_loss = torch.stack(chunk_losses).mean()
            if training:
                if scaler is None:
                    raise ValueError("Training requires an explicit gradient scaler.")
                scaler.scale(chunk_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(barrier.parameters()),
                                         config.clip_grad_norm, error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
            loss_total += float(chunk_loss.detach()) * len(chunk_losses)
            loss_count += len(chunk_losses)
            state, history = state.detach(), history.detach()
            last_position, last_time = last_position.detach(), last_time.detach()
            previous_acceleration, previous_vertical = previous_acceleration.detach(), previous_vertical.detach()
    final_error = torch.linalg.vector_norm(state[:, :3] - goals, dim=1).detach()
    records = []
    for index, trajectory in enumerate(trajectories):
        failure = bool(failures[index])
        collision = bool(collisions[index])
        altitude_bad = bool(altitude_failure[index])
        reached = bool(final_error[index] <= 1)
        records.append({"run_id": trajectory.run_id, "run_index": trajectory.run_index,
            "recorded_safety_pass": trajectory.safety_pass, "recorded_collision": trajectory.collision,
            "initial_state_safe": bool(valid_initial[index]), "steps_simulated": int(executed[index]),
            "initial_truth_clearance_ok": bool(initial_truth_clearance_ok[index]),
            "initial_cbf_valid": bool(initial_cbf_valid[index]),
            "initial_qp_feasible": bool(initial_qp_feasible[index]),
            "duration_s": float(executed[index]) * config.control_dt, "outage_steps": int(outage_steps[index]),
            "filter_failed": failure, "collision": collision, "near_collision": bool(near_collision[index]),
            "altitude_failed": altitude_bad, "minimum_surface_clearance_m": float(minimum_clearance[index]),
            "final_goal_error_m": float(final_error[index]), "goal_reached": reached,
            "control_correction_mean_square": float(correction_total[index] / executed[index].clamp_min(1)),
            "mission_success": reached and not failure and not collision and not bool(near_collision[index]) and not altitude_bad,
            "termination": "invalid-initial-state" if not bool(valid_initial[index]) else
                "filter-infeasible-or-invalid" if failure else "collision" if collision else
                "altitude-failure" if altitude_bad else "rollout-horizon"})
    arrays = {name: torch.stack(values).cpu().numpy() for name, values in traces.items() if values}
    return {"loss": loss_total / max(loss_count, 1), "records": records, "traces": arrays,
            "fault_schedules": schedules, "control_step_s": config.control_dt,
            "truth_collision_step_s": config.plant_dt, "safety_certified": False}


def summarize_rollouts(results):
    records = [record for result in results for record in result["records"]]
    if not records:
        raise ValueError("No rollout trajectories were evaluated.")
    return {"trajectory_count": len(records),
            "loss": float(np.average([result["loss"] for result in results],
                                     weights=[len(result["records"]) for result in results])),
            "collision_rate": float(np.mean([item["collision"] for item in records])),
            "filter_failure_rate": float(np.mean([item["filter_failed"] for item in records])),
            "mission_success_rate": float(np.mean([item["mission_success"] for item in records])),
            "initial_unsafe_count": sum(not item["initial_state_safe"] for item in records),
            "outage_control_exercised_count": sum(item["outage_steps"] > 0 for item in records),
            "minimum_surface_clearance_m": min(item["minimum_surface_clearance_m"] for item in records),
            "mean_final_goal_error_m": float(np.mean([item["final_goal_error_m"] for item in records])),
            "safety_certified": False, "records": records}


def evaluate_closed_loop(model, barrier, dynamics, scene, trajectories, normalizer, config, *,
                         output_dir=None, split_name="validation", steps=None):
    if split_name not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test.")
    if any(item.split != split_name for item in trajectories):
        raise ValueError("Evaluation trajectory IDs do not belong to the requested outer split.")
    results = []
    folder = Path(output_dir) if output_dir is not None else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(trajectories), config.rollout_batch_size):
        group = trajectories[offset:offset + config.rollout_batch_size]
        result = run_rollout(model, barrier, dynamics, scene, group, normalizer, config,
                             fault_seed=config.seed + 9001, steps=steps, collect_traces=folder is not None)
        results.append(result)
        if folder:
            np.savez_compressed(folder / f"{split_name}_rollout_{offset:04d}.npz", **result["traces"])
    summary = summarize_rollouts(results)
    if folder:
        write_json(folder / f"{split_name}_closed_loop.json", summary)
    return summary


def train_stage_two(model, corpus, train_indices, calibration_pack, validation_pack, normalizer,
                    calibration_path, output_dir, config, signature, *, resume=False):
    """Fine-tune on training starts; select on validation routes, never test."""
    device = next(model.parameters()).device
    calibration = load_calibration(calibration_path)
    verify_validation_report(calibration_path)
    if calibration["tracking_error"].get("bound_dimension") != "horizontal-XY":
        raise ValueError("The obstacle barrier needs the measured horizontal tracking-error bound.")
    dynamics = GuidanceDynamics(calibration["parameters"], config.plant_dt).to(device)
    if config.compile_dynamics and device.type == "cuda":
        dynamics.prepare_compile(device)
    scene = TorchLidarScene(corpus.scene, device=device)
    fitting = [item for item in corpus.by_split("train") if item.run_index in train_indices]
    if len(fitting) != len(train_indices):
        raise ValueError("Stage-2 fitting IDs must belong to the training fold.")
    uncertainty = calibrate_reconstruction(model, calibration_pack, config, device)
    barrier_config = BarrierConfig(calibration["tracking_error"]["bound_mps2"], tuple(uncertainty["bounds_m"]))
    barrier = BarrierSafetyFilter(barrier_config).to(device)
    groups = [{"params": model.parameters(), "lr": config.finetune_learning_rate},
              {"params": barrier.parameters(), "lr": config.barrier_learning_rate}]
    optimizer = make_optimizer(groups, config.finetune_learning_rate, config, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and config.amp)
    folder = Path(output_dir) / "stage2"
    folder.mkdir(parents=True, exist_ok=True)
    last = folder / "last.pt"
    history, best, stale, start = [], math.inf, 0, 1
    runtime_sources = source_signature()
    if last.exists():
        if not resume:
            raise FileExistsError("Stage 2 exists; resume explicitly or choose a new output directory.")
        old = _restore_stage(last, "closed_loop", signature, model, optimizer, scheduler, scaler, barrier)
        if config.finetune_epochs < old["epoch"]:
            raise ValueError("Cannot reduce the epoch limit below a saved training epoch.")
        history, best, stale, start = old["history"], old["best_score"], old["stale"], old["epoch"] + 1
        uncertainty = old["uncertainty"]
        barrier.config.range_error_bound_m = tuple(uncertainty["bounds_m"])
    for epoch in range(start, config.finetune_epochs + 1):
        if stale >= config.patience:
            break
        started = time.perf_counter()
        order = torch.randperm(len(fitting)).tolist()
        training_results = []
        for offset in range(0, len(order), config.rollout_batch_size):
            group = [fitting[index] for index in order[offset:offset + config.rollout_batch_size]]
            training_results.append(run_rollout(model, barrier, dynamics, scene, group, normalizer, config,
                optimizer=optimizer, scaler=scaler, fault_seed=config.seed + epoch * 101))
        # Bind empirical errors to THIS candidate's weights before validation.
        # Calibration remains inside TRAIN and never differentiates its bounds.
        uncertainty = calibrate_reconstruction(model, calibration_pack, config, device)
        with torch.no_grad():
            barrier.calibration_bounds.copy_(torch.tensor(uncertainty["bounds_m"], device=device,
                                                          dtype=barrier.calibration_bounds.dtype))
        barrier.config.range_error_bound_m = tuple(uncertainty["bounds_m"])
        reconstruction, _ = evaluate_reconstruction(model, validation_pack, config, device)
        closed_loop = evaluate_closed_loop(model, barrier, dynamics, scene, corpus.by_split("validation"),
                                           normalizer, config, steps=config.evaluation_steps)
        score = (reconstruction["selection_mae_normalized"] + closed_loop["loss"]
                 + 100 * closed_loop["collision_rate"] + 10 * closed_loop["filter_failure_rate"])
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation score.")
        scheduler.step(score)
        improved = score < best - 1e-6
        best, stale = (score, 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch, "training": summarize_rollouts(training_results),
                        "validation_reconstruction": reconstruction, "validation_closed_loop": closed_loop,
                        "selection_score": score, "seconds": time.perf_counter() - started})
        payload = {"schema_version": SCHEMA, "stage": "closed_loop", "signature": signature,
            "epoch": epoch, "model": model.state_dict(), "barrier": barrier.state_dict(),
            "normalizer": normalizer.to_dict(), "model_config": asdict(config.model),
            "barrier_config": asdict(barrier.config), "config": config.to_dict(),
            "dynamics_calibration": calibration, "uncertainty": uncertainty,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng": rng_state(), "history": history,
            "best_score": best, "stale": stale, "safety_certified": False}
        payload["source_signature"] = runtime_sources
        if improved:
            save_checkpoint(folder / "best.pt", dict(payload, checkpoint_role="best", selected_weights=True))
        save_checkpoint(last, dict(payload, checkpoint_role="last", selected_weights=False))
        write_json(folder / "history.json", history)
        if epoch == 1 or epoch % config.log_every == 0:
            print(f"Stage 2 epoch {epoch}: score={score:.5f}, collision={closed_loop['collision_rate']:.3f}, "
                  f"filter failure={closed_loop['filter_failure_rate']:.3f}", flush=True)
    if not (folder / "best.pt").exists():
        raise RuntimeError("No completed Stage-2 checkpoint exists.")
    selected = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"])
    barrier = BarrierSafetyFilter(selected["barrier_config"]).to(device)
    barrier.load_state_dict(selected["barrier"])
    return selected, barrier, dynamics, scene
