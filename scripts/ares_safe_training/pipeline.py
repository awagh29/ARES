"""Colab orchestration: prepare -> pretrain -> closed loop -> sealed test -> export."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .barrier import BarrierSafetyFilter
from .config import PipelineConfig
from .data import FeatureNormalizer, build_dataset
from .dynamics import GuidanceDynamics, load_calibration
from .geometry import TorchLidarScene
from .models import AugmentedSAITS
from .qp import DifferentiableQP
from .training import (SCHEMA, amp_context, device_for, evaluate_closed_loop, evaluate_reconstruction,
    experiment_metadata, immutable_signature, model_state_sha256, prepare_pack, reconstruction_loss,
    restore_rng, rng_state, seed_everything, train_stage_one, train_stage_two, write_json)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize_deployment(folder, result):
    """Recover publication after a sealed test without ever rerunning test."""
    from export_augmented_saits_barrier import export_checkpoint
    folder = Path(folder)
    checkpoint = folder / "stage2" / "best.pt"
    checkpoint_hash = _sha256(checkpoint)
    if result["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Sealed test does not match the deployment checkpoint.")
    destination = folder / "deployment"
    if not destination.exists():
        export_checkpoint(checkpoint, destination)
    manifest = json.loads((destination / "deployment.json").read_text(encoding="utf-8"))
    if manifest["provenance"]["source_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Existing deployment belongs to another checkpoint.")
    for name, artifact in manifest["artifacts"].items():
        if Path(name).name != name or _sha256(destination / name) != artifact["sha256"]:
            raise ValueError("Existing deployment artifact failed its integrity check.")
    result = dict(result, deployment_dir=str(destination))
    write_json(folder / "summary.json", result)
    return result


def run_device_preflight(model, pack, calibration, config):
    """Exercise actual device forward/backward before committing GPU hours."""
    device = next(model.parameters()).device
    rng = rng_state()
    try:
        model.train()
        count = min(config.rollout_batch_size, pack["count"])
        tensors = pack["tensors"]
        index = slice(0, count)
        with amp_context(device, config):
            output = model(tensors["history"][index], tensors["history_mask"][index],
                           tensors["state"][index], tensors["state_mask"][index])
        loss = reconstruction_loss(output, tensors["target"][index], tensors["target_mask"][index], config)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all().item()) for value in gradients):
            raise FloatingPointError("Device reconstructor gradient preflight failed.")
        # A solver verification problem, not training data or a claimed flight.
        linear = torch.tensor([[-.2, .3]], device=device, dtype=torch.float64, requires_grad=True)
        Q = torch.eye(2, device=device, dtype=torch.float64)[None]
        G = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]], device=device, dtype=torch.float64)
        h = torch.ones(1, 4, device=device, dtype=torch.float64)
        result = DifferentiableQP()(Q, linear, G, h)
        result["solution"].square().sum().backward()
        if not bool(result["converged"].all().item()) or not bool(torch.isfinite(linear.grad).all().item()):
            raise FloatingPointError("Device FP64 QP/KKT backward preflight failed.")
        dynamics = GuidanceDynamics(calibration["parameters"], config.plant_dt).to(device)
        state = torch.zeros(count, 13, device=device)
        state[:, 2] = -7
        commands = torch.zeros(count, 4, device=device, requires_grad=True)
        with torch.no_grad():
            commands[:, 3] = calibration["parameters"]["mass_kg"] * calibration["parameters"]["gravity_mps2"]
        next_state = dynamics(state, commands, config.control_dt)
        next_state[:, :6].square().sum().backward()
        if not bool(torch.isfinite(commands.grad).all().item()):
            raise FloatingPointError("Device guidance-dynamics gradient preflight failed.")
        return {"passed": True, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                "reconstruction_autocast": "float16" if device.type == "cuda" and config.amp else "float32",
                "qp_dtype": "float64", "physics_dtype": "float32",
                "checks": ["real-trajectory model forward/backward", "FP64 constrained QP and KKT backward",
                           "13-state guidance forward/backward"],
                "is_gpu_benchmark": False}
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng(rng)


def _load_selected(folder, model, device):
    checkpoint = torch.load(folder / "stage2" / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "closed_loop" or not checkpoint.get("selected_weights"):
        raise ValueError("A selected closed-loop checkpoint is required.")
    model.load_state_dict(checkpoint["model"])
    if checkpoint["uncertainty"]["model_state_sha256"] != model_state_sha256(model.state_dict()):
        raise ValueError("Reconstruction uncertainty is not bound to the selected model weights.")
    barrier = BarrierSafetyFilter(checkpoint["barrier_config"]).to(device)
    barrier.load_state_dict(checkpoint["barrier"])
    dynamics = GuidanceDynamics(checkpoint["dynamics_calibration"]["parameters"], checkpoint["config"]["plant_dt"]).to(device)
    return checkpoint, barrier, dynamics


def run_pipeline(repo_root, output_dir, calibration_path, config=None, *, stage="all", resume=False):
    config = config or PipelineConfig()
    device = device_for(config)
    if stage not in ("all", "pretrain", "finetune", "evaluate"):
        raise ValueError("Unknown pipeline stage.")
    root, folder = Path(repo_root).resolve(), Path(output_dir).expanduser().resolve()
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError("Output directory is not empty. Resume explicitly or choose a new experiment directory.")
    folder.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    corpus = build_dataset(root, train_fraction=config.train_fraction, val_fraction=config.val_fraction, seed=config.seed)
    metadata = experiment_metadata(corpus, config, calibration_path)
    signature = immutable_signature(metadata)
    manifest = folder / "experiment.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Existing experiment data, source, split, calibration or hyperparameters changed.")
        sessions = old.get("sessions", [])
    elif resume and any(folder.iterdir()):
        raise ValueError("Cannot resume an output directory without its experiment manifest.")
    else:
        sessions = []
    sessions.append({"started_at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
                     "configuration": config.to_dict(), "device": str(device)})
    metadata.update(signature=signature, sessions=sessions)
    write_json(manifest, metadata)
    test_seal = folder / "test_evaluation.json"
    test_intent = folder / "test_evaluation_intent.json"
    # Reserve the selected weights BEFORE any test access. An interrupted test
    # is recovery of that same evaluation, never permission to fit new weights.
    if test_intent.exists() or test_seal.exists():
        reservation = json.loads((test_intent if test_intent.exists() else test_seal).read_text(encoding="utf-8"))
        if stage in ("pretrain", "finetune"):
            raise ValueError("Test access already reserved this checkpoint; fitting is closed for this experiment.")
        if _sha256(folder / "stage2" / "best.pt") != reservation["checkpoint_sha256"]:
            raise ValueError("Test access was reserved for different weights; this experiment is closed to fitting.")
        if test_seal.exists():
            return finalize_deployment(folder, json.loads(test_seal.read_text(encoding="utf-8")))
        stage = "evaluate"
    normalizer = FeatureNormalizer.fit(corpus, seed=config.seed, run_indices=metadata["training_run_indices"])
    write_json(folder / "normalizer.json", normalizer.to_dict())
    model = AugmentedSAITS(config.model).to(device)
    calibration = load_calibration(calibration_path)
    validation_pack = prepare_pack(corpus, "validation", normalizer, config, device)
    train_pack = prepare_pack(corpus, "train", normalizer, config, device, metadata["training_run_indices"])
    calibration_pack = prepare_pack(corpus, "train", normalizer, config, device,
                                    metadata["calibration_run_indices_within_train"])
    preflight = run_device_preflight(model, train_pack, calibration, config)
    write_json(folder / "device_preflight.json", preflight)
    print(f"Outer routes: train={len(corpus.splits['train'])}, validation={len(corpus.splits['validation'])}, "
          f"test={len(corpus.splits['test'])}. Fit/calibration inside train="
          f"{len(metadata['training_run_indices'])}/{len(metadata['calibration_run_indices_within_train'])}.", flush=True)
    print(f"Device={device}, AMP FP16={config.amp and device.type == 'cuda'}; "
          f"real pretraining windows={train_pack['count']}; no online map input.", flush=True)
    if stage in ("all", "pretrain"):
        train_stage_one(model, train_pack, validation_pack, normalizer, folder, config, signature, resume=resume)
    elif not (folder / "stage1" / "best.pt").exists():
        raise FileNotFoundError("Stage 1 must complete before closed-loop fine-tuning or final evaluation.")
    else:
        selected = torch.load(folder / "stage1" / "best.pt", map_location="cpu", weights_only=True)
        if selected["signature"] != signature:
            raise ValueError("Stage-1 checkpoint does not match this experiment.")
        model.load_state_dict(selected["model"])
    if stage == "pretrain":
        return {"output_dir": str(folder), "stage": "reconstruction", "device_preflight": preflight}
    # Dense pretraining windows are no longer needed by the rollout optimizer.
    del train_pack
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if stage in ("all", "finetune"):
        selected, barrier, dynamics, scene = train_stage_two(model, corpus, metadata["training_run_indices"],
            calibration_pack, validation_pack, normalizer, calibration_path, folder, config, signature, resume=resume)
    else:
        selected, barrier, dynamics = _load_selected(folder, model, device)
        if selected["signature"] != signature:
            raise ValueError("Selected checkpoint does not match this experiment.")
        scene = TorchLidarScene(corpus.scene, device)
    if stage == "finetune":
        return {"output_dir": str(folder), "stage": "closed_loop", "selected_epoch": selected["epoch"]}
    checkpoint_hash = _sha256(folder / "stage2" / "best.pt")
    if test_seal.exists():
        seal = json.loads(test_seal.read_text(encoding="utf-8"))
        if seal["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Test was already evaluated for another checkpoint. Do not tune against a repeatedly inspected test set.")
        return finalize_deployment(folder, seal)
    write_json(test_intent, {"schema_version": SCHEMA, "checkpoint_sha256": checkpoint_hash,
                            "signature": signature, "selected_epoch": selected["epoch"],
                            "reserved_at_utc": datetime.now(timezone.utc).isoformat()})
    # This is the first access to test windows inside any fit/select workflow.
    test_pack = prepare_pack(corpus, "test", normalizer, config, device)
    metrics, _ = evaluate_reconstruction(model, test_pack, config, device,
                                         folder / "test_reconstruction_predictions.npz")
    closed = evaluate_closed_loop(model, barrier, dynamics, scene, corpus.by_split("test"), normalizer, config,
                                  output_dir=folder / "evaluation", split_name="test", steps=config.evaluation_steps)
    result = {"schema_version": SCHEMA, "checkpoint_sha256": checkpoint_hash, "selected_epoch": selected["epoch"],
              "test_reconstruction": metrics, "test_closed_loop": closed, "safety_certified": False,
              "online_map_input": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(test_seal, result)
    # A versioned bundle retains the online QP, nominal controller, allocation
    # and calibration. It cannot be loaded by the earlier SAITS-only MEX.
    return finalize_deployment(folder, result)
