"""Package selected Stage 2 weights for a future map-free ARES runtime.

    python scripts/export_augmented_saits_barrier.py --checkpoint RUN/stage2/best.pt \
        --output-dir NEW_DEPLOYMENT_DIRECTORY

This is a new contract, not an input to export_saits_matlab.py or its existing
Simulink/MEX consumer. MATLAB-v5 contains ordinary numeric weight records and
reference cases; it does not implement the new runtime. No MATLAB is launched.
Reference inputs are deterministic numerical test fixtures, not training data
or a reconstruction/safety benchmark. CPU FP32 reconstruction and FP64 online
QP coefficients/solutions are computed from the selected checkpoint weights.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from scipy.io import loadmat, savemat
import torch
from torch import nn

from ares_safe_training.barrier import BarrierConfig, BarrierSafetyFilter
from ares_safe_training.config import PipelineConfig
from ares_safe_training.control import desired_yaw_rate, nominal_acceleration
from ares_safe_training.data import FeatureNormalizer
from ares_safe_training.dynamics import (
    CALIBRATION_SCHEMA, COMMAND_NAMES, PARAMETER_NAMES, STATE_NAMES,
    _parameters, acceleration_to_commands,
)
from ares_safe_training.models import AugmentedSAITS, ModelConfig


SCHEMA = "ares-augmented-saits-barrier-deployment-v1"
TRAINING_SCHEMA = "ares-augmented-saits-barrier-training-v1"
FIXTURE_SCHEMA = "ares-augmented-saits-barrier-reference-v1"
CONTROLLER_KEYS = (
    "control_dt", "plant_dt", "reference_speed_mps", "tracking_tube_radius_m",
    "maximum_yaw_rate_radps", "nominal_position_gain", "nominal_velocity_gain",
    "nominal_repulsion_gain",
)
RUNTIME_SOURCES = ("models.py", "barrier.py", "qp.py", "control.py", "dynamics.py", "data.py", "config.py",
                   "saits_endpoint_model.py")
INTEGRATION_REQUIREMENT = (
    "Implement the 34-variable state encoder and two-branch augmented SAITS; "
    "retain the frozen nominal acceleration controller, solve the state-dependent "
    "online BarrierNet QP at every control step, and allocate its acceleration "
    "with the calibrated guidance-plant formula. The previous SAITS Simulink "
    "System/MEX cannot consume this architecture."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    """Canonical hash shared with uncertainty calibration in the trainer."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def runtime_source_sha256():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name if name == "saits_endpoint_model.py" else
            root / "ares_safe_training" / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for name in RUNTIME_SOURCES}


def _sha(value, name):
    _require(isinstance(value, str) and len(value) == 64
             and all(char in "0123456789abcdef" for char in value), f"Invalid {name} SHA-256.")
    return value


def _json(value):
    """Round trip through strict JSON: no tensors, custom objects or NaNs."""
    return json.loads(json.dumps(value, allow_nan=False))


def _scalar(value, name, *, nonnegative=False):
    _require(not isinstance(value, bool) and isinstance(value, (float, int)), f"Invalid {name}.")
    result = float(value)
    _require(math.isfinite(result) and (result >= 0 if nonnegative else result > 0), f"Invalid {name}.")
    return result


def _checked_state(state, module, name):
    _require(isinstance(state, dict), f"Missing {name} numeric state dictionary.")
    expected = module.state_dict()
    parameter_names = set(dict(module.named_parameters()))
    _require(set(state) == set(expected), f"{name} parameter names differ from this architecture.")
    result = {}
    for key, reference in expected.items():
        value = state[key]
        _require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                 and value.shape == reference.shape and value.dtype == reference.dtype,
                 f"{name} parameter shape/dtype differs: {key}.")
        tensor = value.detach().cpu().contiguous().clone()
        _require(bool(torch.isfinite(tensor).all()), f"Nonfinite {name} parameter: {key}.")
        _require(key in parameter_names or torch.equal(tensor, reference),
                 f"{name} structural/calibration buffer differs from its declared architecture: {key}.")
        result[key] = tensor
    module.load_state_dict(result, strict=True)
    return result


def _canonical_normalizer(value):
    result = FeatureNormalizer.from_dict(value)
    canonical = _json(result.to_dict())
    # Do not silently ignore contradictory redundant affine coefficients.
    for name in ("history_mean", "history_std", "state_mean", "state_std"):
        _require(name in value and np.array_equal(np.asarray(value[name], np.float32),
                 np.asarray(canonical[name], np.float32)), f"Normalizer {name} is inconsistent.")
    _require(list(value["target_names"]) == canonical["target_names"], "Normalizer target order differs.")
    _require(canonical["fit_row_count"] > 0 and bool(canonical["training_run_ids"]),
             "Normalizer must have training-only fit provenance.")
    return result, canonical


def _physical_calibration(value, barrier_config):
    """Explicit allowlist: full experiment/calibration dictionaries never ship."""
    _require(value.get("schemaVersion") == CALIBRATION_SCHEMA and value.get("calibrationComplete") is True,
             "A completed authoritative dynamics calibration is required.")
    parameters = _parameters(value)
    tracking = value.get("tracking_error", {})
    _require(tracking.get("bound_dimension") == "horizontal-XY",
             "Dynamics disturbance calibration must explicitly bound horizontal XY tracking.")
    disturbance = _scalar(tracking.get("xy_bound_mps2"), "calibrated XY disturbance", nonnegative=True)
    _require(math.isclose(disturbance, barrier_config.disturbance_bound_mps2, rel_tol=1e-10, abs_tol=1e-12),
             "Barrier disturbance bound differs from the dynamics calibration.")
    envelope = value.get("software_design_envelope", {})
    policy_dt = _scalar(envelope.get("policy_step_s"), "calibrated policy step")
    _require(math.isclose(policy_dt, barrier_config.dt, abs_tol=1e-12), "Calibrated policy/QP steps differ.")
    result = {
        "schema_version": CALIBRATION_SCHEMA, "calibration_complete": True,
        "parameters": parameters, "state_names": list(STATE_NAMES), "command_names": list(COMMAND_NAMES),
        "disturbance_bound_xy_mps2": disturbance, "policy_step_s": policy_dt,
        "source_calibration_sha256": _sha(value.get("calibration_sha256"), "calibration file"),
        "complete_calibration_payload_sha256": hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest(),
        "software_design_bounds_only": True, "hardware_safety_certificate": False,
    }
    # Preserve declared numeric actuation bounds without copying coverage poses,
    # profiles, scene geometry, local file paths or arbitrary calibration fields.
    vector_keys = ("acceleration_lower_mps2", "acceleration_upper_mps2", "acceleration_slew_mps3")
    scalar_keys = ("yaw_rate_limit_radps", "roll_limit_rad", "pitch_limit_rad",
                   "truth_sample_step_s", "thrust_lower_n", "thrust_upper_n")
    design = {}
    for name in vector_keys:
        array = np.asarray(envelope.get(name), dtype=np.float64)
        _require(array.shape == (3,) and np.isfinite(array).all(), f"Missing calibrated {name}.")
        design[name] = array.tolist()
    for name in scalar_keys:
        design[name] = _scalar(envelope.get(name), f"calibrated {name}")
    _require(np.all(np.asarray(design["acceleration_lower_mps2"]) < design["acceleration_upper_mps2"])
             and np.all(np.asarray(design["acceleration_slew_mps3"]) > 0), "Invalid calibrated software bounds.")
    _require(np.allclose(design["acceleration_lower_mps2"][:2], -barrier_config.acceleration_limit_mps2)
             and np.allclose(design["acceleration_upper_mps2"][:2], barrier_config.acceleration_limit_mps2)
             and np.allclose(design["acceleration_slew_mps3"][:2], barrier_config.jerk_limit_mps3),
             "QP acceleration/jerk limits differ from the calibrated software envelope.")
    _require(design["acceleration_lower_mps2"][2] == -4 and design["acceleration_upper_mps2"][2] == 4
             and design["acceleration_slew_mps3"][2] == 1, "Frozen vertical controller differs from its calibrated envelope.")
    result["software_design_envelope"] = design
    for name in ("model_sha256", "derivative_source_sha256", "calibration_source_sha256"):
        source = value.get("reference", {}).get(name)
        if source is not None:
            result[name] = _sha(source, name)
    return result


def _controller_defaults(config, barrier_config, physical):
    _require(isinstance(config, dict) and all(key in config for key in CONTROLLER_KEYS),
             "Checkpoint lacks the frozen online-controller defaults.")
    values = {name: _scalar(config[name], name, nonnegative=name == "nominal_repulsion_gain")
              for name in CONTROLLER_KEYS}
    _require(math.isclose(values["control_dt"], barrier_config.dt, abs_tol=1e-12)
             and values["plant_dt"] <= .001, "Controller/QP interval or integration step differs.")
    _require(values["maximum_yaw_rate_radps"] <= physical["parameters"]["yaw_rate_limit_radps"] + 1e-12,
             "Controller yaw limit exceeds the calibrated guidance limit.")
    values.update({
        "reference": "C2 minimum-jerk; duration=max(10,1.875*distance/reference_speed)",
        "nominal_horizontal": "tanh(reference_a+Kp*(reference_p-p)+Kv*(reference_v-v)+sector_repulsion)",
        "nominal_vertical": "clip(reference_a_D+Kp*(reference_p_D-p_D)+Kv*(reference_v_D-v_D),-4,4)",
        "vertical_acceleration_slew_mps3": 1.,
        "vertical_slew": "max(previous_vertical-control_dt,min(previous_vertical+control_dt,nominal_vertical))",
        "desired_yaw_rate": "maximum_yaw_rate*tanh(2*wrapped(goal_bearing-yaw)/maximum_yaw_rate); zero at goal",
        "allocation": "ares_safe_training.dynamics.acceleration_to_commands; no post-QP clipping",
        "online_qp_required": True, "online_obstacle_map_input": False,
        "measured_reconstruction_fusion": "where(current_14_observed,measured_14,predicted_14)",
        "uncertainty_per_sector": "zero only when BOTH distance and return fraction are measured; else calibrated bound",
        "no_return_is_measured": True, "return_fraction_is_confidence": False,
    })
    return values


def _selected_checkpoint(checkpoint):
    _require(isinstance(checkpoint, dict), "Checkpoint must be a weights-only dictionary.")
    _require(checkpoint.get("schema_version") == TRAINING_SCHEMA
             and checkpoint.get("stage") == "closed_loop", "Only Stage 2 closed_loop checkpoints can be exported.")
    _require(checkpoint.get("checkpoint_role") == "best" and checkpoint.get("selected_weights") is True,
             "Export requires the selected Stage 2 best checkpoint; last/pretraining weights are not deployable.")
    _sha(checkpoint.get("signature"), "training signature")
    source_hashes = checkpoint.get("source_signature")
    _require(isinstance(source_hashes, dict), "Selected checkpoint lacks its runtime source_signature hashes.")
    training_sources = {}
    for name, digest in source_hashes.items():
        _require(isinstance(name, str) and Path(name).name == name and name.endswith(".py"),
                 "Invalid training source-signature filename.")
        training_sources[name] = _sha(digest, "training source " + name)
    for name, digest in runtime_source_sha256().items():
        _require(training_sources.get(name) == digest, "Runtime source differs from selected checkpoint: " + name)
    _require(isinstance(checkpoint.get("epoch"), int) and checkpoint["epoch"] >= 1,
             "Selected checkpoint epoch must be positive.")
    _require(math.isfinite(float(checkpoint.get("best_score", float("nan")))), "Selected validation score is missing.")
    history = checkpoint.get("history", [])
    _require(isinstance(history, list) and bool(history) and isinstance(history[-1], dict)
             and history[-1].get("epoch") == checkpoint["epoch"]
             and math.isclose(float(history[-1].get("selection_score", float("nan"))),
                              float(checkpoint["best_score"]), rel_tol=1e-12, abs_tol=1e-12),
             "Selected epoch/validation score differs from the checkpoint history.")
    cfg = ModelConfig(**checkpoint["model_config"])
    cfg.validate()
    _require(cfg.window_size == 50, "This runtime contract requires 50 past/current rows.")
    barrier_cfg = BarrierConfig(**checkpoint["barrier_config"])
    barrier_cfg.validate()
    model = AugmentedSAITS(cfg).cpu().eval()
    barrier = BarrierSafetyFilter(barrier_cfg).cpu().eval()
    model_state = _checked_state(checkpoint["model"], model, "model")
    barrier_state = _checked_state(checkpoint["barrier"], barrier, "barrier")
    model_hash = state_sha256(model_state)
    uncertainty = checkpoint.get("uncertainty", {})
    _require(uncertainty.get("schema_version") == "ares-empirical-reconstruction-bound-v1"
             and uncertainty.get("certified") is False,
             "Selected checkpoint requires empirical reconstruction-bound provenance.")
    _require(_sha(uncertainty.get("model_state_sha256"), "uncertainty source model") == model_hash,
             "Reconstruction error bounds were calibrated for different model weights.")
    bounds = np.asarray(uncertainty.get("bounds_m"), dtype=np.float64)
    _require(bounds.shape == (7,) and np.isfinite(bounds).all() and (bounds >= 0).all()
             and np.array_equal(bounds, np.asarray(barrier_cfg.range_error_bound_m, np.float64)),
             "Uncertainty bounds differ from the selected barrier configuration.")
    _require(np.array_equal(barrier_state["calibration_bounds"].numpy(), bounds),
             "Barrier state/calibration bounds differ.")
    normalizer, normalizer_dict = _canonical_normalizer(checkpoint["normalizer"])
    physical = _physical_calibration(checkpoint["dynamics_calibration"], barrier_cfg)
    defaults = _controller_defaults(checkpoint["config"], barrier_cfg, physical)
    clean_uncertainty = {"schema_version": "ares-empirical-reconstruction-bound-v1",
        "bounds_m": bounds.tolist(), "model_state_sha256": model_hash,
        "one_sided_error": "max(predicted_distance-clean_distance,0)",
        "certified": False, "provenance": "finite reserved-TRAIN-route calibration; not a uniform certificate"}
    for name in ("quantile", "engineering_margin_m"):
        clean_uncertainty[name] = _scalar(uncertainty.get(name), name, nonnegative=True)
    _require(0 < clean_uncertainty["quantile"] <= 1, "Invalid empirical bound quantile.")
    counts = uncertainty.get("sample_counts", [])
    _require(len(counts) == 7 and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in counts),
             "Invalid bound sample counts.")
    clean_uncertainty["sample_counts"] = list(counts)
    return model, barrier, normalizer, {
        "schema_version": SCHEMA, "model": model_state, "barrier": barrier_state,
        "model_config": asdict(cfg), "barrier_config": asdict(barrier_cfg),
        "architecture": model.architecture(), "normalizer": normalizer_dict,
        "uncertainty": clean_uncertainty, "physical_calibration": physical,
        "controller_defaults": defaults,
        "training_selection": {"signature": checkpoint["signature"], "epoch": checkpoint["epoch"],
            "validation_score": float(checkpoint["best_score"]), "checkpoint_role": "best", "selected_weights": True},
        "training_source_sha256": training_sources,
        "integration": {"implemented_in_existing_simulink_runtime": False,
            "previous_saits_mex_compatible": False, "requires_online_qp": True,
            "required_work": INTEGRATION_REQUIREMENT},
    }


def make_reference_inputs(normalizer):
    """Deterministic generic parity cases; no dataset, goal or map is read."""
    names = ("healthy_no_return", "short_2s_outage", "long_full_window_outage",
             "partial_paired_sectors", "cold_one_row", "healthy_returns", "invalid_previous_acceleration")
    count, steps = len(names), 50
    raw = np.zeros((count, steps, 31), dtype=np.float32)
    row = np.arange(steps, dtype=np.float32)
    for index in range(count):
        raw[index, :, :7] = 3 + np.arange(7)[None, :] * .35 + .05 * np.sin(row[:, None] / 9)
        raw[index, :, 7:14] = np.linspace(.1, .7, 7)[None, :]
        raw[index, :, 17:20] = [.04, .01, 0]
        raw[index, :, 20:26] = [0, 1, 0, 1, 0, 1]
        raw[index, :, 29] = 1
    raw[0, :, :7], raw[0, :, 7:14] = 7, 0  # Valid no-return, never an outage.
    mask = np.ones(raw.shape, dtype=bool)
    mask[1, -20:, :14] = False
    mask[2, :, :14] = False
    paired = [0, 2, 5, 7, 9, 12]
    mask[3, -25:, paired] = False
    mask[4, :-1] = False  # Leading cold-history padding masks motion too.
    mask[4, -1, :14] = False
    raw[6] = raw[2]
    mask[6] = mask[2]
    for index in range(count):
        healthy = mask[index, :, :14].all(1)
        last = 0
        for time in range(steps):
            if healthy[time]:
                last = time
            age = .1 * (time - last)
            raw[index, time, 14:17] = [.04 * age, .01 * age, 0]
            raw[index, time, 29:] = [healthy[time], age]
    raw[~mask] = np.nan
    position = np.tile(np.array([0., 0., -7.], np.float32), (count, 1))
    state = np.concatenate((position, raw[:, -1]), axis=1)
    state_mask = np.concatenate((np.ones((count, 3), bool), mask[:, -1]), axis=1)
    return names, {
        "raw_history": raw, "history_mask": mask,
        "history_normalized": normalizer.transform_history_numpy(raw, mask),
        "raw_current_state": state, "current_state_mask": state_mask,
        "current_state_normalized": normalizer.transform_state_numpy(state, state_mask),
    }


@torch.inference_mode()
def make_reference_cases(model, barrier, normalizer, deployment):
    names, arrays = make_reference_inputs(normalizer)
    output = model(torch.from_numpy(arrays["history_normalized"]), torch.from_numpy(arrays["history_mask"]),
                   torch.from_numpy(arrays["current_state_normalized"]), torch.from_numpy(arrays["current_state_mask"]))
    arrays.update({"prediction_" + key: value.cpu().numpy() for key, value in output.items()})
    prediction = output["physical"].double()
    endpoint_mask = torch.from_numpy(arrays["history_mask"][:, -1, :14])
    measured = torch.from_numpy(np.nan_to_num(arrays["raw_history"][:, -1, :14], nan=0)).double()
    fused = torch.where(endpoint_mask, measured, prediction)
    # Independent online context: fixed current state/reference, not replayed
    # recorded future states or oracle geometry. Absolute position is arbitrary.
    raw_state = np.nan_to_num(arrays["raw_current_state"], nan=0)
    position = torch.from_numpy(raw_state[:, :3]).double()
    velocity = torch.from_numpy(raw_state[:, 20:23]).double()
    orientation = torch.from_numpy(raw_state[:, 23:29]).double().reshape(-1, 3, 2)
    rpy = torch.atan2(orientation[..., 0], orientation[..., 1])
    batch = len(names)
    physical_state = torch.cat((position, velocity, rpy, torch.zeros(batch, 4, dtype=torch.float64)), 1)
    previous = torch.zeros(batch, 2, dtype=torch.float64)
    previous[-1, 0] = 2 * barrier.config.acceleration_limit_mps2  # Finite deliberate invalidity.
    reference = {"p": position + position.new_tensor([.2, .05, -.3]),
                 "v": torch.zeros_like(position), "a": torch.zeros_like(position),
                 "rho": position.new_full((batch,), deployment["controller_defaults"]["tracking_tube_radius_m"]),
                 "rhodot": position.new_zeros(batch), "rhoddot": position.new_zeros(batch)}
    defaults = deployment["controller_defaults"]
    control_config = PipelineConfig(**{key: defaults[key] for key in CONTROLLER_KEYS})
    nominal = nominal_acceleration(physical_state, fused, reference, control_config)
    pair_observed = endpoint_mask[:, :7] & endpoint_mask[:, 7:]
    bounds = torch.where(pair_observed, torch.zeros_like(prediction[:, :7]), barrier.calibration_bounds.expand(batch, 7))
    result = barrier(position, velocity, rpy, fused, reference, previous,
                     uncertainty_bounds=bounds, nominal_acceleration=nominal[:, :2])
    arrays.update({"qp_" + key: value.cpu().numpy() for key, value in result.items()})
    selected = torch.cat((result["acceleration"], result["tracking_slack"][:, None]), 1)
    hessian = torch.diag(position.new_tensor([1., 1., 2 * barrier.config.tracking_slack_penalty]))
    previous_vertical = position.new_zeros(batch)
    vertical = torch.maximum(previous_vertical - barrier.config.dt,
                             torch.minimum(previous_vertical + barrier.config.dt, nominal[:, 2]))
    acceleration = torch.cat((result["acceleration"], vertical[:, None]), 1)
    goal = position + position.new_tensor([3., 1., 0.])
    yaw_rate = desired_yaw_rate(physical_state, goal, defaults["maximum_yaw_rate_radps"])
    commands = acceleration_to_commands(acceleration, rpy[:, 2], yaw_rate,
                                        parameters=deployment["physical_calibration"]["parameters"])
    arrays.update({"fused_physical": fused.numpy(), "qp_physical_state": physical_state.numpy(),
        "qp_previous_acceleration": previous.numpy(), "qp_uncertainty_bounds_m": bounds.numpy(),
        "qp_nominal_acceleration": nominal.numpy(), "qp_hessian": hessian.expand(batch, 3, 3).numpy(),
        "qp_linear": torch.cat((-nominal[:, :2], position.new_zeros(batch, 1)), 1).numpy(),
        "qp_selected_solution": selected.numpy(), "qp_full_acceleration": acceleration.numpy(),
        "qp_previous_vertical_acceleration": previous_vertical.numpy(), "qp_goal_position": goal.numpy(),
        "qp_yaw_rate_command": yaw_rate.numpy(), "allocated_commands": commands.numpy()})
    for key, value in reference.items():
        arrays["qp_reference_" + key] = value.numpy()
    metadata = {"schema_version": FIXTURE_SCHEMA, "case_names": list(names),
        "purpose": "deterministic generic numerical parity and invalidity fixtures; not dataset predictions or performance metrics",
        "reconstruction_execution": "CPU FP32 evaluation, no autocast/dropout",
        "gain_head_execution": "selected FP32 gain-head weights; output cast to FP64",
        "qp_execution": "CPU FP64 coefficients and differentiable online inequality solver",
        "nominal_and_allocation_execution": "CPU FP64 fixed online context; vertical slew limit is upstream of allocation",
        "qp_objective": "0.5*z.T@Q@z+linear.T@z; z=[a_N,a_E,tracking_slack]",
        "qp_constraint": "constraint_matrix@z<=constraint_rhs",
        "qp_constraint_rows": {"obstacle_sector_vertices": [0, 21], "visibility_cone": [21, 24],
            "acceleration_and_jerk": [24, 28], "soft_tracking_tube": [28, 29], "nonnegative_tracking_slack": [29, 30]},
        "fallback": "emergency_used is NOT a feasible/certified safety action; compare raw QP and selected solutions separately",
        "reference_units": {"p": "m NED", "v": "m/s NED", "a": "m/s^2 NED", "rho": "m", "rhodot": "m/s", "rhoddot": "m/s^2"},
        "array_layout": "Python batch-major; MATLAB fixture arrays keep the same logical dimensions",
        "oracle_map_used": False, "online_qp_mandatory": True}
    arrays["case_names"] = np.asarray(names)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    _require(not bool(arrays["qp_input_valid"][-1]) and bool(arrays["qp_emergency_used"][-1]),
             "Invalid-input reference did not select the documented fallback.")
    _require(np.isfinite(arrays["prediction_physical"]).all()
             and np.isfinite(arrays["allocated_commands"]).all(), "Reference outputs are nonfinite.")
    return arrays, metadata


def _weight_records(state):
    records = np.empty((1, len(state)), dtype=object)
    for index, name in enumerate(sorted(state)):
        value = state[name].detach().cpu().numpy()
        records[0, index] = {"name": name, "value": value, "shape": np.asarray(value.shape, dtype=np.int64),
                             "dtype": str(value.dtype), "layout": "PyTorch native; linear weights [output,input]"}
    return records


def _mat_pack(deployment):
    # JSON companions preserve exact feature order and nested field names;
    # numeric affine coefficients and every parameter are directly accessible.
    normalizer = deployment["normalizer"]
    result = {"schemaVersion": SCHEMA, "modelWeights": _weight_records(deployment["model"]),
        "barrierWeights": _weight_records(deployment["barrier"]),
        "modelConfigJson": json.dumps(deployment["model_config"], sort_keys=True),
        "architectureJson": json.dumps(deployment["architecture"], sort_keys=True),
        "barrierConfigJson": json.dumps(deployment["barrier_config"], sort_keys=True),
        "normalizerJson": json.dumps(normalizer, sort_keys=True),
        "physicalCalibrationJson": json.dumps(deployment["physical_calibration"], sort_keys=True),
        "controllerDefaultsJson": json.dumps(deployment["controller_defaults"], sort_keys=True),
        "uncertaintyJson": json.dumps(deployment["uncertainty"], sort_keys=True),
        "historyMean": np.asarray(normalizer["history_mean"], np.float32),
        "historyStd": np.asarray(normalizer["history_std"], np.float32),
        "stateMean": np.asarray(normalizer["state_mean"], np.float32),
        "stateStd": np.asarray(normalizer["state_std"], np.float32),
        "targetScale": np.asarray([7.] * 7 + [1.] * 7, np.float32),
        "rangeErrorBoundM": np.asarray(deployment["uncertainty"]["bounds_m"], np.float64),
        "physicalParameterNames": np.asarray(PARAMETER_NAMES, dtype=object),
        "physicalParameterValues": np.asarray([deployment["physical_calibration"]["parameters"][key]
                                               for key in PARAMETER_NAMES], np.float64),
        "requiresOnlineQP": True, "previousSaitsMexCompatible": False,
        "integrationRequirement": INTEGRATION_REQUIREMENT,
        "provenanceJson": json.dumps(deployment["provenance"], sort_keys=True)}
    return {"deployment": result}


def _export_onnx(model, arrays, destination):
    class PredictionOnly(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped

        def forward(self, history, history_mask, state, state_mask):
            result = self.model(history, history_mask, state, state_mask)
            return result["normalized"], result["physical"]

    import onnx
    inputs = tuple(torch.from_numpy(arrays[key][:1]) for key in
                   ("history_normalized", "history_mask", "current_state_normalized", "current_state_mask"))
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(PredictionOnly(model).eval(), inputs, str(destination),
            input_names=["history_normalized", "history_mask", "current_state_normalized", "current_state_mask"],
            output_names=["prediction_normalized", "prediction_physical"],
            opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(destination)))
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def export_checkpoint(checkpoint_path, output_dir=None, *, export_onnx=False):
    """Read a trusted weights-only Stage 2 best checkpoint; publish a new folder.

    Existing destinations are always rejected. The original checkpoint, raw
    data and previous deployment artifacts are never edited. The deployment
    intentionally excludes optimizer/RNG/history and simulator scene geometry.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint_hash = file_sha256(checkpoint_path)
    destination = (Path(output_dir).expanduser() if output_dir is not None else
                   checkpoint_path.parent / ("deployment-" + checkpoint_hash[:12])).resolve()
    if destination.exists():
        raise FileExistsError(f"Deployment destination already exists: {destination}")
    if export_onnx and importlib.util.find_spec("onnx") is None:
        raise ImportError("--onnx requires the optional onnx package. Omit the flag for portable MAT-v5/PT export.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, barrier, normalizer, deployment = _selected_checkpoint(checkpoint)
    package_root = Path(__file__).resolve().parent
    source_paths = [Path(__file__), package_root / "saits_endpoint_model.py"]
    source_paths.extend(package_root / "ares_safe_training" / name for name in
                        ("models.py", "barrier.py", "qp.py", "control.py", "dynamics.py", "data.py", "config.py"))
    deployment["provenance"] = {
        "source_checkpoint_filename": checkpoint_path.name, "source_checkpoint_sha256": checkpoint_hash,
        "source_training_schema": TRAINING_SCHEMA,
        "source_model_state_sha256": state_sha256(deployment["model"]),
        "source_barrier_state_sha256": state_sha256(deployment["barrier"]),
        "export_source_sha256": {str(path.relative_to(package_root)):
            hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest() for path in source_paths},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "reference_device": "cpu", "benchmark_or_matlab_validation_performed": False,
    }
    arrays, reference_metadata = make_reference_cases(model, barrier, normalizer, deployment)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Inherit the parent ACL, allowing a separate MATLAB process to read MAT-v5.
    staging = destination.parent / ("." + destination.name + ".export-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        torch.save(deployment, staging / "deployment.pt")
        savemat(str(staging / "model.mat"), _mat_pack(deployment), format="5", do_compression=True,
                long_field_names=True, oned_as="row")
        np.savez_compressed(staging / "reference_inputs_outputs.npz", **arrays)
        savemat(str(staging / "reference_inputs_outputs.mat"),
                {"reference": {key: (np.asarray(value, dtype=object) if key == "case_names" else value)
                               for key, value in arrays.items()}},
                format="5", do_compression=True, long_field_names=True, oned_as="row")
        (staging / "preprocessing.json").write_text(json.dumps(deployment["normalizer"], indent=2,
            allow_nan=False) + "\n", encoding="utf-8")
        if export_onnx:
            _export_onnx(model, arrays, staging / "reconstruction.onnx")
        # Verify both portable containers can be reopened before publication.
        reloaded = torch.load(staging / "deployment.pt", map_location="cpu", weights_only=True)
        _require(state_sha256(reloaded["model"]) == deployment["provenance"]["source_model_state_sha256"],
                 "Deployment weight round trip failed.")
        _require("deployment" in loadmat(str(staging / "model.mat")), "MAT-v5 round trip failed.")
        with np.load(staging / "reference_inputs_outputs.npz", allow_pickle=False) as saved:
            _require(saved["history_normalized"].shape == (7, 50, 31), "Reference NPZ round trip failed.")
        _require(file_sha256(checkpoint_path) == checkpoint_hash, "Source checkpoint changed during export.")
        manifest = {key: value for key, value in deployment.items() if key not in ("model", "barrier")}
        manifest["reference"] = reference_metadata
        manifest["artifacts"] = {path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                                 for path in sorted(staging.iterdir())}
        manifest["onnx"] = {"included": bool(export_onnx), "contains_online_qp": False,
                            "fixed_batch_size": 1, "opset": 17 if export_onnx else None,
                            "runtime_numerical_parity_checked": False}
        (staging / "deployment.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Deployment destination appeared during export: {destination}")
        os.rename(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory; an existing directory is never overwritten.")
    parser.add_argument("--onnx", action="store_true", help="Also export batch-one reconstruction ONNX; online QP remains mandatory.")
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output_dir, export_onnx=args.onnx))


if __name__ == "__main__":
    main()
