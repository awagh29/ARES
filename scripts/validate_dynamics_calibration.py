"""Validate the differentiable plant against immutable MATLAB numerical oracles.

Run before Stage 2, for example::

    python scripts/validate_dynamics_calibration.py --calibration data/.../calibration.json

The sibling validation.json seals both calibration/fixture bytes and the Python
plant source. Its numerical budgets apply to this software guidance model and
the exported test cases; this is not a hardware or global safety certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from ares_safe_training import dynamics as dynamics_module
from ares_safe_training.dynamics import GuidanceDynamics, load_calibration


VALIDATION_SCHEMA = "ares-guidance-dynamics-validation-v1"
SOURCE_PATH = Path(dynamics_module.__file__).resolve()
GROUPS = {"position_m": slice(0, 3), "velocity_mps": slice(3, 6),
          "attitude_rad": slice(6, 9), "body_rate_radps": slice(9, 12),
          "actual_thrust_n": slice(12, 13)}
# Position/velocity accuracy matters for 1ms collision truth and the barrier
# disturbance model. A separate 0.1rad/s budget exposes the stiff 1950/s yaw
# rate's first-millisecond RK4 transient instead of hiding it in an aggregate
# all-state relative tolerance. A half-step convergence gate also checks it.
TRAJECTORY_BUDGETS = {"position_m": 3e-5, "velocity_mps": 3e-4,
    "attitude_rad": 1e-4, "body_rate_radps": 0.1, "actual_thrust_n": 1e-7}
ENDPOINT_BUDGETS = {"position_m": 3e-5, "velocity_mps": 3e-4,
    "attitude_rad": 1e-6, "body_rate_radps": 2e-5, "actual_thrust_n": 1e-7}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256(path: Path) -> str:
    # read_text's universal-newline conversion makes Windows/Colab checkouts
    # agree on source identity even when Git materializes CRLF versus LF.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _comparison(actual, expected, *, atol, rtol=0.0) -> dict:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape), "atol": atol, "rtol": rtol}
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    error = np.abs(actual - expected)
    return {"passed": finite and bool(np.all(error <= atol + rtol * np.abs(expected))),
            "shape": list(actual.shape), "all_finite": finite,
            "max_absolute_error": float(error.max()) if finite else None,
            "atol": atol, "rtol": rtol}


def _group_comparisons(actual, expected, budgets) -> dict:
    return {name: _comparison(actual[..., selected], expected[..., selected], atol=budgets[name])
            for name, selected in GROUPS.items()}


def _trajectory(dynamics, initial, commands, times):
    samples = [initial]
    with torch.no_grad():
        for duration in np.diff(times):
            initial = dynamics(initial, commands, float(duration))
            samples.append(initial)
    return torch.stack(samples, dim=1).cpu().numpy()


def _all_passed(value):
    if isinstance(value, dict):
        if "passed" in value and value["passed"] is not True:
            return False
        return all(_all_passed(child) for child in value.values())
    if isinstance(value, list):
        return all(_all_passed(child) for child in value)
    return True


def validate_calibration(calibration_path, output_path=None, *, integration_dt=0.001,
                         write_report=True) -> dict:
    """Measure numerical parity and optionally publish its source-bound report.

    Integrity/shape errors raise ValueError. Numerical failures produce a report
    with passed=false, including the failed gate, and the CLI exits nonzero.
    No authoritative calibration or fixture bytes are changed.
    """
    calibration_path = Path(calibration_path).expanduser().resolve(strict=True)
    calibration = load_calibration(calibration_path)
    fixture_path = (calibration_path.parent / calibration["fixtures"]["file"]).resolve(strict=True)
    if fixture_path.parent != calibration_path.parent:
        raise ValueError("The numerical fixture must be inside the calibration directory.")
    fixture_sha = _sha256(fixture_path)
    if fixture_sha != calibration["fixtures"]["sha256"]:
        raise ValueError("The MATLAB numerical fixture SHA-256 does not match the calibration.")
    fixture = loadmat(fixture_path, squeeze_me=True)
    times = np.asarray(fixture["rollout_time_s"], dtype=np.float64).reshape(-1)
    policy_step = float(calibration["software_design_envelope"]["policy_step_s"])
    if (len(times) < 3 or not np.isfinite(times).all() or times[0] != 0
            or np.any(np.diff(times) <= 0)
            or not math.isclose(times[-1], policy_step, rel_tol=0, abs_tol=1e-12)
            or np.max(np.diff(times)) > 0.001 + 1e-12):
        raise ValueError("MATLAB rollout times must span the policy interval on a <=1ms grid.")
    plant = GuidanceDynamics(calibration, integration_dt=integration_dt)
    to64 = lambda name: torch.tensor(np.asarray(fixture[name]), dtype=torch.float64)
    states, commands = to64("states"), to64("commands")
    with torch.no_grad():
        derivative = plant.derivative(states, commands).numpy()
        derivative32 = plant.derivative(to64("states_single").float(),
                                       to64("commands_single").float()).numpy()
        # MAT-v7 integer-valued DOUBLE arrays may load as int16. Restore their
        # intended arithmetic explicitly; mixed Torch scalar promotion is lossy.
        allocated = plant.acceleration_to_commands(to64("acceleration_inputs"),
            to64("allocation_yaw"), to64("allocation_yaw_rate")).numpy()
    gates = {
        "derivative_float64": _comparison(derivative, fixture["derivatives"], atol=3e-9, rtol=2e-12),
        "derivative_float32": _comparison(derivative32, fixture["derivatives_single"], atol=2e-3, rtol=3e-6),
        "acceleration_allocation_float64": _comparison(allocated, fixture["allocation_commands"],
                                                       atol=1e-12, rtol=1e-12),
    }
    indices = np.asarray(fixture["jacobian_case_indices"]).reshape(-1).astype(int) - 1
    if np.any(indices < 0) or np.any(indices >= len(states)):
        raise ValueError("MATLAB control-Jacobian indices are outside the derivative fixtures.")
    jacobians = []
    for index in indices:
        jacobian = torch.autograd.functional.jacobian(
            lambda command: plant.derivative(states[index:index + 1], command[None])[0],
            commands[index])
        jacobians.append(jacobian.detach().numpy())
    gates["continuous_control_jacobian"] = _comparison(np.asarray(jacobians),
        fixture["control_jacobian"], atol=1e-6, rtol=1e-8)
    initial, rollout_commands = to64("rollout_initial_states"), to64("rollout_commands")
    # MATLAB supplies the continuous Jacobian above. Independently check that
    # autograd also differentiates through the actual discrete training rollout
    # (including the analytic thrust stages), against central differences.
    discrete_jacobians, discrete_differences = [], []
    selected_cases = sorted({0, len(initial) - 1})
    epsilon = 1e-5
    for index in selected_cases:
        state, command = initial[index:index + 1], rollout_commands[index]
        jacobian = torch.autograd.functional.jacobian(
            lambda action: plant(state, action[None], policy_step)[0], command)
        perturbations = torch.eye(4, dtype=torch.float64) * epsilon
        plus_minus = torch.cat((command + perturbations, command - perturbations), dim=0)
        with torch.no_grad():
            endpoints = plant(state.expand(8, -1), plus_minus, policy_step)
        central = ((endpoints[:4] - endpoints[4:]) / (2 * epsilon)).T
        discrete_jacobians.append(jacobian.detach().numpy())
        discrete_differences.append(central.numpy())
    gates["zoh_autograd_vs_central_difference"] = _comparison(
        np.asarray(discrete_jacobians), np.asarray(discrete_differences), atol=1e-6, rtol=1e-5)
    oracle = np.asarray(fixture["rollout_trajectory"], dtype=np.float64)
    trajectory = _trajectory(plant, initial, rollout_commands, times)
    gates["trajectory_float64"] = _group_comparisons(trajectory, oracle, TRAJECTORY_BUDGETS)
    gates["endpoint_float64"] = _group_comparisons(trajectory[:, -1],
        np.asarray(fixture["rollout_next_states"]), ENDPOINT_BUDGETS)
    with torch.no_grad():
        endpoint, trace = plant(initial, rollout_commands, policy_step, return_trace=True)
    expected_steps = int(round(policy_step / plant.integration_dt))
    # The public collision trace must retain initial and every integration
    # endpoint. It must agree with the complete-state sampler on the native grid.
    if (expected_steps + 1 == len(times)
            and math.isclose(policy_step / plant.integration_dt, expected_steps, abs_tol=1e-10)):
        gates["collision_trace_positions"] = _comparison(trace.numpy(), trajectory[..., :3], atol=2e-12)
    else:
        gates["collision_trace_positions"] = {"passed": False,
            "reason": "Production integration step must match the authoritative collision grid."}
    gates["trace_endpoint_state"] = _comparison(endpoint.numpy(), trajectory[:, -1], atol=2e-10)
    fine = _trajectory(GuidanceDynamics(calibration, integration_dt=plant.integration_dt / 2),
                       initial, rollout_commands, times)
    convergence = {}
    for name, selected in GROUPS.items():
        coarse_error = float(np.max(np.abs(trajectory[..., selected] - oracle[..., selected])))
        fine_error = float(np.max(np.abs(fine[..., selected] - oracle[..., selected])))
        # Reference ode15s error dominates tiny thrust differences. Its known
        # numerical floor is excluded; the other twelve states must converge.
        floor = 1e-7 if name == "actual_thrust_n" else 1e-9
        convergence[name] = {"passed": fine_error <= max(floor, 0.2 * coarse_error),
            "coarse_max_absolute_error": coarse_error, "half_step_max_absolute_error": fine_error,
            "maximum_error_ratio": 0.2, "reference_error_floor": floor}
    gates["half_step_convergence"] = convergence
    trajectory32 = _trajectory(plant, initial.float(), rollout_commands.float(), times)
    budgets32 = {**TRAJECTORY_BUDGETS, "position_m": 8e-5, "velocity_mps": 4e-4,
                 "actual_thrust_n": 5e-7}
    gates["trajectory_float32"] = _group_comparisons(trajectory32, oracle, budgets32)
    # Allocation gates establish that the calibrated virtual-acceleration box
    # is feasible without silently clipping a QP result after optimization.
    envelope, parameters = calibration["software_design_envelope"], calibration["parameters"]
    limits = np.array([parameters["roll_limit_rad"], parameters["pitch_limit_rad"],
                       parameters["yaw_rate_limit_radps"]])
    gates["allocation_design_bounds"] = {"passed": bool(
        np.all(np.abs(allocated[:, :3]) <= limits + 1e-12)
        and np.all(allocated[:, 3] >= envelope["thrust_lower_n"] - 1e-12)
        and np.all(allocated[:, 3] <= envelope["thrust_upper_n"] + 1e-12)),
        "maximum_absolute_commands": np.max(np.abs(allocated), axis=0).tolist(),
        "software_bounds_not_motor_hardware": True}
    report = {"schemaVersion": VALIDATION_SCHEMA, "passed": _all_passed(gates),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_sha256": calibration["calibration_sha256"], "fixtures_sha256": fixture_sha,
        "dynamics_source_sha256": _source_sha256(SOURCE_PATH),
        "validator_source_sha256": _source_sha256(Path(__file__).resolve()),
        "calibration_file": calibration_path.name, "fixtures_file": fixture_path.name,
        "integration_dt_s": plant.integration_dt, "policy_step_s": policy_step,
        "torch_version": str(torch.__version__), "validation_device": "cpu",
        "coverage": "Exported MATLAB derivative/Jacobian/grid-allocation and six 0.1s ZOH cases only.",
        "limitations": ["Software guidance plant, not hardware certification.",
            "Stiff first-millisecond yaw/body-rate error has a separate explicit 0.1rad/s budget.",
            "No GPU throughput or long-horizon global-error guarantee is inferred."],
        "gates": gates}
    if write_report:
        destination = Path(output_path) if output_path is not None else calibration_path.with_name("validation.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def verify_validation_report(calibration_path, report_path=None) -> dict:
    """Fail closed on missing/failed/stale reports before production Stage 2."""
    calibration_path = Path(calibration_path).expanduser().resolve(strict=True)
    path = Path(report_path) if report_path is not None else calibration_path.with_name("validation.json")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Run validate_dynamics_calibration.py before Stage 2; its report is missing or invalid.") from error
    if report.get("schemaVersion") != VALIDATION_SCHEMA or report.get("passed") is not True:
        raise ValueError("The dynamics validation report did not pass the required parity gates.")
    calibration = load_calibration(calibration_path)
    fixture_path = calibration_path.parent / calibration["fixtures"]["file"]
    expected = {"calibration_sha256": calibration["calibration_sha256"],
                "fixtures_sha256": _sha256(fixture_path),
                "dynamics_source_sha256": _source_sha256(SOURCE_PATH),
                "validator_source_sha256": _source_sha256(Path(__file__).resolve())}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("The dynamics validation report is stale: calibration, fixture or Python source changed.")
    if expected["fixtures_sha256"] != calibration["fixtures"]["sha256"]:
        raise ValueError("The dynamics calibration fixture hash does not match its manifest.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    report = validate_calibration(arguments.calibration, arguments.output)
    failed = [name for name, gate in report["gates"].items() if not _all_passed(gate)]
    print(json.dumps({"passed": report["passed"], "failed_gates": failed,
                      "calibration_sha256": report["calibration_sha256"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
