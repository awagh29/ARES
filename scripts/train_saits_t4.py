"""T4-ready, endpoint-supervised SAITS training for the ARES single-episode data.

Requires PyTorch >=2.5, NumPy and pandas. Uses the SAITS architecture adaptation
in saits_endpoint_model.py; this is not the upstream stock MIT/ORT training loop.
Only train/validation enter train_saits. Call evaluate_saits on test afterward.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from torch.nn import functional as F

from lidar_reconstruction_data import make_saits_windows
from saits_endpoint_model import SAITSEndpointModel

FORMAT = "ares-saits-endpoint-training-v1"


def _source_signature():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_text(
        encoding="utf-8").encode("utf-8")).hexdigest()
        for name in ("train_saits_t4.py", "saits_endpoint_model.py",
                     "lidar_reconstruction_data.py")}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    device: str = "cuda"
    amp: bool = True
    seed: int = 42
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    n_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    auxiliary_weight: float = 0.1
    huber_delta: float = 0.1
    clip_grad_norm: float = 1.0
    patience: int = 25
    min_delta: float = 1e-5
    lr_patience: int = 8
    checkpoint_every: int = 10
    log_every: int = 10

    def validate(self):
        for name in ("epochs", "batch_size", "d_model", "n_heads", "d_ff", "n_layers",
                     "patience", "lr_patience", "checkpoint_every", "log_every"):
            value = getattr(self, name)
            _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                     f"{name} must be a positive integer.")
        _require(self.device in ("cuda", "cpu"), "device must be cuda or cpu.")
        _require(isinstance(self.seed, int) and 0 <= self.seed < 2**32, "Invalid seed.")
        _require(0 <= self.dropout < 1 and self.d_model % self.n_heads == 0
                 and self.d_model % 2 == 0, "Invalid dropout or attention dimensions.")
        for name in ("learning_rate", "huber_delta", "clip_grad_norm"):
            _require(math.isfinite(getattr(self, name)) and getattr(self, name) > 0,
                     f"{name} must be finite and positive.")
        for name in ("weight_decay", "auxiliary_weight", "min_delta"):
            _require(math.isfinite(getattr(self, name)) and getattr(self, name) >= 0,
                     f"{name} must be finite and nonnegative.")


@dataclass
class FeatureTransform:
    feature_names: list[str]
    target_columns: list[str]
    context_mean: np.ndarray
    context_std: np.ndarray
    distance_scale: float = 7.0

    @classmethod
    def fit(cls, train_prepared):
        frame = train_prepared["X"]
        targets = list(train_prepared["target_columns"])
        _require(len(targets) == 14 and list(frame.columns[:14]) == targets
                 and frame.shape[1] == 31, "Expected the 14-target/31-input sector schema.")
        # Unique chronological training rows, not duplicated overlapping windows.
        context = frame.iloc[:, 14:].to_numpy(dtype=np.float64)
        _require(len(context) > 0 and np.isfinite(context).all(), "Invalid training context.")
        mean, std = context.mean(axis=0), context.std(axis=0, ddof=0)
        std[std < 1e-8] = 1.0
        return cls(list(frame.columns), targets, mean, std)

    def transform_inputs(self, values):
        array = np.asarray(values, dtype=np.float32).copy()
        _require(array.shape[-1] == len(self.feature_names), "Input feature width changed.")
        array[..., :7] /= self.distance_scale
        array[..., 14:] = (array[..., 14:] - self.context_mean) / self.context_std
        return array

    def transform_targets(self, values):
        array = np.asarray(values, dtype=np.float32).copy()
        _require(array.shape[-1] == 14, "Target width must be 14.")
        array[..., :7] /= self.distance_scale
        return array

    def inverse_targets(self, values):
        array = np.asarray(values).copy()
        _require(array.shape[-1] == 14, "Target width must be 14.")
        array[..., :7] *= self.distance_scale
        return array

    def to_dict(self):
        return {"feature_names": self.feature_names, "target_columns": self.target_columns,
                "context_mean": self.context_mean.tolist(), "context_std": self.context_std.tolist(),
                "distance_scale": self.distance_scale}

    @classmethod
    def from_dict(cls, value):
        return cls(list(value["feature_names"]), list(value["target_columns"]),
                   np.asarray(value["context_mean"], dtype=np.float64),
                   np.asarray(value["context_std"], dtype=np.float64),
                   float(value["distance_scale"]))


@dataclass
class TrainingResult:
    model: SAITSEndpointModel
    transform: FeatureTransform
    config: TrainConfig
    history: pd.DataFrame
    best_epoch: int
    best_metric: float
    output_dir: Path
    device: torch.device
    run_id: str
    training_end_time_s: float
    validation_end_time_s: float


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path, value):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _save_history(folder, history):
    temporary = folder / "history.tmp.csv"
    pd.DataFrame(history).to_csv(temporary, index=False)
    os.replace(temporary, folder / "history.csv")


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "batch_generator": generator.get_state().cpu(),
            "numpy": {"algorithm": numpy_state[0],
                      "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]}}


def _restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    generator.set_state(state["batch_generator"].cpu())
    value = state["numpy"]
    np.random.set_state((value["algorithm"], value["keys"].cpu().numpy().astype(np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))


def _run_id(prepared):
    return str(prepared["observed_df"]["run_id"].iloc[0])


def _partition_times(prepared):
    frame = prepared["observed_df"]
    times = frame["time_s"].to_numpy(dtype=np.float64)
    _require(len(times) > 0 and np.isfinite(times).all()
             and (np.diff(times) > 0).all(), "Partition times must be finite and strictly increasing.")
    _require(frame["run_id"].notna().all() and frame["run_id"].nunique() == 1,
             "Each partition must contain exactly one episode.")
    _require(all(len(prepared[key]) == len(frame) for key in
                 ("X", "y", "target_mask", "score_mask", "persistence_y")),
             "Prepared partition lengths differ.")
    return times


def _signature(prepared):
    digest = hashlib.sha256()
    digest.update(json.dumps({"features": list(prepared["X"].columns),
                              "targets": list(prepared["target_columns"]),
                              "window_size": prepared["window_size"],
                              "run_id": _run_id(prepared)}, sort_keys=True).encode())
    for value in (prepared["X"].to_numpy(dtype=np.float64),
                  prepared["y"].to_numpy(dtype=np.float64),
                  prepared["target_mask"].to_numpy(dtype=np.float64),
                  prepared["score_mask"].to_numpy(dtype=np.float64),
                  prepared["persistence_y"].to_numpy(dtype=np.float64),
                  prepared["observed_df"]["time_s"].to_numpy(dtype=np.float64)):
        digest.update(np.isnan(value).tobytes())
        digest.update(np.nan_to_num(value, nan=0.0).tobytes())
    return digest.hexdigest()


def _bundle(prepared, transform, device):
    _partition_times(prepared)
    _require(list(prepared["X"].columns) == transform.feature_names
             and list(prepared["target_columns"]) == transform.target_columns,
             "Feature/target names changed between partitions.")
    all_targets = prepared["y"].to_numpy(dtype=np.float64)
    _require(all_targets.shape == (len(prepared["X"]), 14)
             and np.isfinite(all_targets).all(), "Invalid clean endpoint labels.")
    windows = make_saits_windows(prepared)
    X = transform.transform_inputs(windows["X"])
    y = transform.transform_targets(windows["y"])
    mask = np.asarray(windows["observed_mask"], dtype=bool)
    target_mask = np.asarray(windows["target_mask"], dtype=bool)
    _require(np.array_equal(mask, ~np.isnan(X)) and np.isfinite(X[mask]).all(),
             "Invalid observed inputs or observation mask.")
    _require(np.isfinite(y).all() and target_mask.shape == y.shape
             and target_mask.any(axis=1).all(), "Invalid clean endpoint labels or target mask.")
    _require(np.isfinite(X[:, :, 14:]).all() and np.isnan(X[:, -1, :14]).all(),
             "Expected observed motion context and hidden endpoint LiDAR.")
    arrays = {"X": np.nan_to_num(X, nan=0.0), "observed_mask": mask,
              "y": y, "target_mask": target_mask}
    if device.type == "cuda":
        required = sum(value.nbytes for value in arrays.values())
        _require(required < torch.cuda.get_device_properties(device).total_memory // 4,
                 "Prepared data is too large for GPU-resident batching; use a streaming trainer.")
    tensors = {key: torch.from_numpy(np.ascontiguousarray(value)).to(device)
               for key, value in arrays.items()}
    return {"tensors": tensors, "windows": windows, "count": len(y)}


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _endpoint_loss(output, target, mask, config):
    weights = mask.float()
    denominator = weights.sum().clamp_min(1)
    def term(prediction):
        errors = F.smooth_l1_loss(prediction.float(), target.float(),
                                  reduction="none", beta=config.huber_delta)
        return (errors * weights).sum() / denominator
    return term(output["prediction"]) + config.auxiliary_weight * 0.5 * (
        term(output["stage1"]) + term(output["stage2"]))


@torch.inference_mode()
def _predict(model, bundle, config, device):
    model.eval()
    data = bundle["tensors"]
    pieces = []
    for start in range(0, bundle["count"], config.batch_size):
        stop = start + config.batch_size
        with _autocast(device, config.amp and device.type == "cuda"):
            output = model(data["X"][start:stop], data["observed_mask"][start:stop])
        pieces.append(output["prediction"].float())
    prediction = torch.cat(pieces)
    _require(bool(torch.isfinite(prediction).all().item()), "Nonfinite model prediction.")
    return prediction


def _metrics(prediction, windows, transform):
    predicted = transform.inverse_targets(prediction)
    target = np.asarray(windows["y"], dtype=np.float64)
    baseline = np.asarray(windows["persistence_y"], dtype=np.float64)
    mask = np.asarray(windows["target_mask"], dtype=bool)
    distance_mask, fraction_mask = mask[:, :7], mask[:, 7:]
    distance_error = predicted[:, :7] - target[:, :7]
    fraction_error = predicted[:, 7:] - target[:, 7:]
    returning = distance_mask & (target[:, 7:] > 0)
    near = returning & (target[:, :7] < 3.0)
    result = {
        "scored_timestep_count": int(len(target)),
        "scored_distance_count": int(distance_mask.sum()),
        "distance_mae_m": float(np.abs(distance_error[distance_mask]).mean()),
        "distance_rmse_m": float(np.sqrt(np.square(distance_error[distance_mask]).mean())),
        "return_fraction_mae": float(np.abs(fraction_error[fraction_mask]).mean()),
        "persistence_distance_mae_m": float(np.abs(baseline[:, :7] - target[:, :7])[distance_mask].mean()),
        "persistence_return_fraction_mae": float(np.abs(baseline[:, 7:] - target[:, 7:])[fraction_mask].mean()),
        "returning_sector_count": int(returning.sum()),
        "target_no_return_only": bool(not returning.any()),
        "distinct_target_rows": int(len(np.unique(target, axis=0))),
        "obstacle_distance_overestimation_m": (
            float(np.maximum(distance_error[returning], 0).mean()) if returning.any() else None),
        "near_obstacle_overestimation_m": (
            float(np.maximum(distance_error[near], 0).mean()) if near.any() else None),
        "prediction_outside_physical_bounds_count": int(
            np.count_nonzero((predicted[:, :7] < 0) | (predicted[:, :7] > 7))
            + np.count_nonzero((predicted[:, 7:] < 0) | (predicted[:, 7:] > 1))),
        "metrics_use_unclipped_predictions": True,
    }
    return result, predicted, target, baseline


def _write_evaluation(folder, split_name, metrics, prediction, target, baseline, windows, run_id):
    metrics_path = folder / f"{split_name}_metrics.json"
    _atomic_json(metrics_path, metrics)
    temporary = folder / f".{split_name}_predictions.tmp.npz"
    np.savez_compressed(temporary, prediction=prediction, target=target,
                        persistence=baseline, target_mask=windows["target_mask"],
                        target_times_s=windows["target_times_s"],
                        source_row_positions=windows["source_row_positions"],
                        target_columns=np.asarray(windows["target_columns"], dtype=str),
                        run_id=np.asarray(run_id), split_name=np.asarray(split_name))
    os.replace(temporary, folder / f"{split_name}_predictions.npz")
    table = {"run_id": [run_id] * len(target), "time_s": windows["target_times_s"],
             "source_row_position": windows["source_row_positions"]}
    for index, name in enumerate(windows["target_columns"]):
        table[f"predicted_{name}"] = prediction[:, index]
        table[f"target_{name}"] = target[:, index]
        table[f"persistence_{name}"] = baseline[:, index]
    pd.DataFrame(table).to_csv(folder / f"{split_name}_predictions.csv", index=False)


def _hardware(device, amp):
    value = {"torch": str(torch.__version__), "numpy": np.__version__,
             "device_type": device.type, "effective_amp": amp,
             "mixed_precision_dtype": "float16" if amp else "float32",
             "cuda_runtime": torch.version.cuda,
             "note": "Seeded execution; bitwise equality across GPU/library versions is not guaranteed."}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        value.update({"gpu": props.name, "compute_capability": [props.major, props.minor],
                      "gpu_memory_bytes": props.total_memory})
    return value


def train_saits(train_prepared, val_prepared, *, output_dir,
                config: TrainConfig | None = None, resume=False) -> TrainingResult:
    """Fit on training only; choose the checkpoint using validation only.

    Entire tiny window tensors stay on the selected device. No test data enters
    this function. Resume verifies data/config, restores optimizer/scaler/RNG,
    and permits increasing epochs, log_every, or checkpoint_every only.
    """
    config = config or TrainConfig()
    config.validate()
    _require(tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5),
             "PyTorch >=2.5 is required.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab, or explicitly use device='cpu' for a development check.")
    device = torch.device(config.device)
    use_amp = config.amp and device.type == "cuda"
    train_times = _partition_times(train_prepared)
    val_times = _partition_times(val_prepared)
    _require(_run_id(train_prepared) == _run_id(val_prepared)
             and train_times[-1] < val_times[0], "Expected chronological train/validation partitions of one episode.")
    folder = Path(output_dir).expanduser().resolve()
    if not resume and folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {folder}. Choose a new directory or resume=True.")
    if resume and not folder.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    transform = FeatureTransform.fit(train_prepared)
    train = _bundle(train_prepared, transform, device)
    validation = _bundle(val_prepared, transform, device)
    _require(train["tensors"]["X"].shape[1:] == validation["tensors"]["X"].shape[1:],
             "Train/validation window shapes differ.")
    shape = train["tensors"]["X"].shape
    model_kwargs = {"n_steps": shape[1], "n_features": shape[2], "n_targets": 14,
                    "d_model": config.d_model, "n_heads": config.n_heads,
                    "d_ff": config.d_ff, "n_layers": config.n_layers, "dropout": config.dropout}
    model = SAITSEndpointModel(**model_kwargs).to(device)
    optimizer_kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (TypeError, NotImplementedError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.lr_patience,
        threshold=config.min_delta, threshold_mode="abs", min_lr=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    signatures = {"train": _signature(train_prepared), "validation": _signature(val_prepared)}
    source_signature = _source_signature()
    history, best_state = [], None
    best_metric, best_epoch, bad_epochs, start_epoch = math.inf, 0, 0, 1
    if resume:
        candidates = [torch.load(path, map_location="cpu", weights_only=True)
                      for path in (folder / "last.pt", folder / "best.pt") if path.is_file()]
        _require(bool(candidates), "No complete checkpoint found for resume.")
        checkpoint = max(candidates, key=lambda item: item["epoch"])
        _require(checkpoint["format_version"] == FORMAT
                 and checkpoint["data_signature"] == signatures,
                 "Resume data/outages changed; start a new experiment directory.")
        _require(checkpoint.get("source_signature") == source_signature,
                 "Resume model/trainer/preparation source changed; use the original source or start a new experiment.")
        old_config, new_config = dict(checkpoint["config"]), asdict(config)
        old_epochs = old_config["epochs"]
        for name in ("epochs", "log_every", "checkpoint_every"):
            old_config.pop(name); new_config.pop(name)
        _require(old_config == new_config and config.epochs >= old_epochs
                 and checkpoint["device_type"] == device.type,
                 "Resume hyperparameters/device changed, or epochs decreased; start a new experiment.")
        _require(checkpoint["model_kwargs"] == model_kwargs
                 and checkpoint["transform"] == transform.to_dict(), "Resume feature transform/architecture changed.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        best_state = checkpoint["best_state_dict"]
        best_metric, best_epoch = checkpoint["best_metric"], checkpoint["best_epoch"]
        bad_epochs, start_epoch = checkpoint["bad_epochs"], checkpoint["epoch"] + 1
        _restore_rng(checkpoint["rng"], generator)
    hardware = _hardware(device, use_amp)
    old_run_config = (json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
                      if resume and (folder / "run_config.json").is_file() else {})
    sessions = old_run_config.get("sessions", []) + [{
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_epoch": start_epoch, "requested_epochs": config.epochs, "hardware": hardware}]
    run_config = {"format_version": FORMAT, "config": asdict(config), "model_kwargs": model_kwargs,
                  "initial_config": old_run_config.get("initial_config", asdict(config)),
                  "sessions": sessions, "source_signature": source_signature,
                  "hardware": hardware, "data_signature": signatures,
                  "run_id": _run_id(train_prepared),
                  "train_window_count": train["count"], "validation_window_count": validation["count"],
                  "train_diagnostics": train_prepared["metadata"],
                  "validation_diagnostics": val_prepared["metadata"],
                  "model_source_sha256": hashlib.sha256(Path(__file__).with_name("saits_endpoint_model.py").read_bytes()).hexdigest(),
                  "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _atomic_json(folder / "run_config.json", run_config)
    _atomic_json(folder / "preprocessing.json", transform.to_dict())
    if val_prepared["metadata"].get("scored_target_content") == "no-return-only":
        print("PIPELINE CHECK: validation targets contain no forward returns; persistence is already perfect. This is not an obstacle-reconstruction benchmark.", flush=True)
    print(f"Device={device}; AMP FP16={use_amp}; windows={train['count']}/{validation['count']}; parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    elapsed_start = time.perf_counter()
    last_epoch = start_epoch - 1

    def save_checkpoint(epoch, destination):
        payload = {"format_version": FORMAT, "epoch": epoch, "best_epoch": best_epoch,
                   "best_metric": best_metric, "bad_epochs": bad_epochs,
                   "model_state_dict": _cpu_tree(model.state_dict()),
                   "best_state_dict": best_state,
                   "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                   "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "rng": _rng_state(generator), "history": history, "config": asdict(config),
                   "transform": transform.to_dict(), "model_kwargs": model_kwargs,
                   "device_type": device.type, "data_signature": signatures,
                   "source_signature": source_signature}
        _atomic_checkpoint(destination, payload)

    for epoch in range(start_epoch, config.epochs + 1):
        if bad_epochs >= config.patience:
            break
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(train["count"], device=device, generator=generator)
        total_loss, count, skipped = 0.0, 0, 0
        data = train["tensors"]
        for start in range(0, train["count"], config.batch_size):
            indices = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                output = model(data["X"][indices], data["observed_mask"][indices])
            # Loss reductions stay FP32 outside autocast.
            loss = _endpoint_loss(output, data["y"][indices], data["target_mask"][indices], config)
            _require(bool(torch.isfinite(loss).item()), "Nonfinite training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm,
                                           error_if_nonfinite=not use_amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(use_amp and scaler.get_scale() < previous_scale)
            total_loss += float(loss.detach().item()) * len(indices)
            count += len(indices)
        predicted = _predict(model, validation, config, device)
        val_data = validation["tensors"]
        val_mask = val_data["target_mask"].float()
        val_metric = float(((predicted - val_data["y"]).abs() * val_mask).sum().div(val_mask.sum()).item())
        _require(math.isfinite(val_metric), "Nonfinite validation metric.")
        scheduler.step(val_metric)
        improved = val_metric < best_metric - config.min_delta
        if improved:
            best_metric, best_epoch, bad_epochs = val_metric, epoch, 0
            best_state = _cpu_tree(model.state_dict())
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_loss": total_loss / count,
                        "val_mae_normalized": val_metric,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "epoch_seconds": time.perf_counter() - epoch_start,
                        "amp_scale": float(scaler.get_scale()), "skipped_optimizer_steps": skipped})
        last_epoch = epoch
        if improved:
            save_checkpoint(epoch, folder / "best.pt")
        if epoch % config.checkpoint_every == 0:
            save_checkpoint(epoch, folder / "last.pt")
            _save_history(folder, history)
        if epoch == 1 or epoch % config.log_every == 0 or bad_epochs >= config.patience:
            print(f"epoch {epoch:03d} train={total_loss/count:.6f} val={val_metric:.6f} best={best_metric:.6f} lr={optimizer.param_groups[0]['lr']:.2g}", flush=True)
    _require(best_state is not None, "No valid checkpoint was produced.")
    save_checkpoint(last_epoch, folder / "last.pt")
    _save_history(folder, history)
    model.load_state_dict(best_state)
    model.eval()
    validation_prediction = _predict(model, validation, config, device).cpu().numpy()
    val_metrics, values, truth, baseline = _metrics(validation_prediction, validation["windows"], transform)
    val_metrics.update({"run_id": _run_id(val_prepared), "split_name": "validation",
                        "best_epoch": best_epoch})
    _write_evaluation(folder, "validation", val_metrics, values, truth, baseline, validation["windows"],
                      _run_id(val_prepared))
    summary = {"format_version": FORMAT, "best_epoch": best_epoch,
               "best_validation_mae_normalized": best_metric, "completed_epochs": last_epoch,
               "early_stopped": bad_epochs >= config.patience,
               "session_elapsed_seconds": time.perf_counter() - elapsed_start,
               "hardware": hardware, "validation": val_metrics,
               "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
               "test_used_for_model_selection": False,
               "interpretation": "pipeline-check-only" if val_metrics["target_no_return_only"] else "single-flight-feasibility",
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(folder / "training_summary.json", summary)
    return TrainingResult(model, transform, config, pd.DataFrame(history), best_epoch, best_metric,
                          folder, device, _run_id(train_prepared), float(train_times[-1]), float(val_times[-1]))


def evaluate_saits(result: TrainingResult, prepared, *, split_name="test"):
    """Evaluate selected weights after training, with persistence on identical targets."""
    _require(split_name in ("validation", "test"), "split_name must be validation or test.")
    _partition_times(prepared)
    if _run_id(prepared) == result.run_id:
        boundary = result.validation_end_time_s if split_name == "test" else result.training_end_time_s
        _require(float(prepared["observed_df"]["time_s"].iloc[0]) > boundary,
                 "Evaluation partition overlaps earlier fitting/model-selection data.")
    bundle = _bundle(prepared, result.transform, result.device)
    prediction = _predict(result.model, bundle, result.config, result.device).cpu().numpy()
    metrics, values, target, baseline = _metrics(prediction, bundle["windows"], result.transform)
    metrics.update({"run_id": _run_id(prepared), "split_name": split_name, "best_epoch": result.best_epoch})
    _write_evaluation(result.output_dir, split_name, metrics, values, target, baseline, bundle["windows"],
                      _run_id(prepared))
    if metrics["target_no_return_only"]:
        print(f"{split_name}: no-return-only targets; persistence distance MAE={metrics['persistence_distance_mae_m']:.6f} m. Treat this as a pipeline check.", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def plot_training(result: TrainingResult):
    """Optional notebook plot; requires matplotlib. Returns a figure, not a file."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(result.history["epoch"], result.history["train_loss"], label="Training Huber + auxiliary")
    axes[0].set(xlabel="Epoch", ylabel="Normalized loss")
    axes[0].legend()
    axes[1].plot(result.history["epoch"], result.history["val_mae_normalized"], label="Validation endpoint MAE")
    axes[1].axvline(result.best_epoch, linestyle="--", color="grey", label="Selected epoch")
    axes[1].set(xlabel="Epoch", ylabel="Normalized MAE")
    axes[1].legend()
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-id", default="c3eaab07-d882-4825-98a5-84c278cf8935")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from colab_episode_starter import EpisodeDataset
    from lidar_reconstruction_data import make_table, split_train_val_test, introduce_lidar_perturbations
    dataset = EpisodeDataset(args.repo_root, args.batch_id)
    run_id = args.run_id or dataset.runs.iloc[0]["RunId"]
    parts = split_train_val_test(make_table(dataset.load_episode(run_id)))
    prepared = [introduce_lidar_perturbations(frame, seed=seed, n_outages=outages)
                for frame, seed, outages in zip(parts, (42, 43, 44), (3, 1, 1))]
    output = args.output_dir or args.repo_root / "results" / f"saits_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    result = train_saits(prepared[0], prepared[1], output_dir=output, config=config, resume=args.resume)
    evaluate_saits(result, prepared[2])


if __name__ == "__main__":
    main()
