# Train state-augmented SAITS with a LiDAR BarrierNet in Colab

Use [`notebooks/train_augmented_saits_barrier.ipynb`](../notebooks/train_augmented_saits_barrier.ipynb) in your existing Colab checkout. This is a new two-stage experiment across all 80 recorded ARES episodes. It pretrains a state-augmented reconstructor, then fine-tunes reconstruction and bounded barrier gains through closed-loop rollouts of the calibrated 13-state guidance plant. It does not deploy the result to Simulink or reuse the earlier single-episode checkpoint.

The implementation is [`scripts/ares_safe_training/`](../scripts/ares_safe_training/); the command-line entry point is [`scripts/train_augmented_saits_barrier.py`](../scripts/train_augmented_saits_barrier.py). Read the [barrier derivation and assumptions](LIDAR_BARRIER_DERIVATION.md) before interpreting navigation results. A training run completing, or a QP returning a solution, does not establish collision-free operation.

## What must be in the checkout

Keep the repository and its data layout together. The default dataset batch is `c3eaab07-d882-4825-98a5-84c278cf8935`. Required inputs include:

- `data/batches/<batch-id>/python_index.csv`, its dataset/export QA JSON files, and the batch manifests.
- The corresponding 80 `data/processed/lidar_episode_v1/<run-id>/episode.npz` files and `data/raw/<run-id>/manifest.json` files.
- `data/calibration/saits_barrier_dynamics/de5533e2-d348-4e43-a06f-4e5bb2e50055/calibration.json`, `fixtures.mat`, and the generated `validation.json`.
- The current scripts, configuration, and their supporting modules; uploading only this notebook is insufficient.

The loader reconstructs episode locations relative to the checkout. Windows absolute paths inside collection metadata remain provenance; Colab does not open them. It verifies all 80 episodes and their recorded hashes before training. Keep the two collision episodes and the clearance-violation episodes: filtering by outcome would change the declared dataset. The original healthy batch contains 19,185 scans, 33 clearance violations, and two sampled collisions; these are collection outcomes, not results of the new controller.

## Runtime and persistent output

Select **Runtime → Change runtime type → T4 GPU**, then run the notebook cells in order. Its hardware cell requires CUDA, prints the actual GPU and library versions, and makes no assumption that a T4 was assigned. This workflow targets a T4; no T4 training duration, throughput, memory result, or completed training score has been measured locally.

Keep the CUDA-enabled PyTorch installation supplied by Colab. The dependency inventory is [`requirements-colab-safe-training.txt`](../requirements-colab-safe-training.txt): PyTorch `>=2.5`, NumPy, pandas, and SciPy; matplotlib is optional for plots, and ONNX is optional for a reconstruction-only export. The notebook installs missing lightweight dependencies additively. It does not install `qpth`, replace PyTorch, or downgrade NumPy. The hard-inequality QP uses native batched PyTorch FP64 operations and an implicit KKT backward pass.

Mount Drive with Colab's normal authorization flow. The notebook checks that `/content/drive` is an actual mounted filesystem and that `MyDrive` exists before placing output there. Merely creating `/content/drive/MyDrive` does not make files persistent. Use a new experiment name for a new split, architecture, fault policy, calibration, or optimization configuration.

Keep input episodes on Colab's local disk when practical, with checkpoints on mounted Drive. A local `/content` output directory is deleted when the runtime is reclaimed. Checkpoints are written atomically after completed epochs; interruption during an epoch can lose that epoch's work. A Drive disconnect still interrupts writes, so retain the most recent valid checkpoint.

## Split and leakage contract

The outer split is frozen by whole run ID before fault injection or normalization: **56 training / 12 validation / 12 test**. Assignment is stratified by North/East starts and their starting-position ranges, without using outcomes or target values. All scans and all overlapping history windows from one run remain in the same outer split.

Inside the 56 training runs, 12 are reserved for empirical reconstruction-error calibration, leaving **44 runs for gradient fitting**. This reservation does not add a fourth outer test fold. The normalizer uses only those 44 fitting runs, with one contribution per native row and declared augmentation condition; overlapping windows do not repeatedly weight a row. The reserved 12 supply no gradient or normalization statistics. They determine one-sided distance-error bounds for the current candidate model. Validation selects checkpoints and drives early stopping; test IDs enter neither selection nor calibration.

This is generalization to held-out starts in the same recorded scenario, not evidence of transfer to an unseen obstacle map. The state branch contains absolute NED position, so route or scene memorization remains possible even though the online model never receives obstacle-map coordinates.

## Predictor and label contract

The temporal branch receives a **50 × 31** past-to-present window with an explicit observation mask. Its 31 channels are:

1. Seven minimum valid-return distances, ordered from body bearing −60° to +60°.
2. Seven valid-return fractions in the same sector order.
3. Three NED displacements since the last completely measured scan, three velocities, six sine/cosine attitude values, three body angular rates, a complete-scan availability flag, and outage age: 17 context channels.

The current-state branch attends over **34 feature tokens**: absolute NED position followed by the current 31 channels. Two SAITS attention blocks produce the temporal representation; a separate feature-token transformer produces the state representation; a joint decoder outputs 14 bounded physical quantities. This is a project extension of [SAITS](https://arxiv.org/abs/2202.08516), not a claim to reproduce the paper's full benchmark architecture or results.

The outputs are seven distances in `[0,7]` m and seven fractions in `[0,1]`. Fractions describe observed return occupancy; they are not confidence probabilities. An intact sector with no return is observed `[7 m, 0]`. An outage changes the input mask and hides both members of the affected sector pair. It never changes the clean target. Masked numerical placeholders remain unobserved after normalization.

No goal, controller command, avoidance reference, status code, map coordinate, clean hidden target, or future state enters either encoder. The mission goal enters only the frozen reference/controller. Leading cold history is padded with missing values and false masks; it is not filled from future scans. Reconstructed readings do not reset the last-healthy position/time anchor or become falsely measured history.

## Stage 1: recorded-data reconstruction

Defaults are 120 epochs, batch size 128, AdamW, and FP16 autocast/gradient scaling on CUDA. Each recorded trajectory supplies a healthy profile and a seeded mixed-fault profile: complete short outages of 1.5–3 s, complete long outages of 5–7 s, and partial sector outages, separated by recovery. Fault schedules for validation/calibration/test remain fixed. Long faults can remove all measured LiDAR from the 50-row history.

Training uses hidden-target Smooth-L1 reconstruction loss, plus an observed-value term and auxiliary losses for the two temporal stages. Targets are recorded clean sensor features, not synthetic geometry labels. Healthy examples remain in the training set. Validation selects reconstruction checkpoints using normalized distance/fraction error. Inspect return-containing examples separately: constant no-return segments can give deceptively small reconstruction errors.

## Validate the actual plant before Stage 2

Run this cell on each synced runtime, before starting the training CLI:

```python
import subprocess, sys
subprocess.run([
    sys.executable, str(REPO_ROOT / "scripts/validate_dynamics_calibration.py"),
    "--calibration", str(CALIBRATION),
], check=True, cwd=REPO_ROOT)
```

This reads the immutable MATLAB calibration and numerical fixtures and publishes a fresh sibling `validation.json`. Stage 2 fails closed if the report is missing, failed, or mismatched to the calibration, fixture, or normalized Python source hashes. Newline normalization makes source hashes portable between Windows and Linux checkouts. Windows MATLAB installation paths in the provenance are not runtime dependencies.

The plant state is `[p_NED(3), v_NED(3), roll/pitch/yaw(3), body_rates(3), actual_thrust]`. It integrates the installed guidance-model equations, including attitude/yaw and thrust lags, rather than replacing the plant with a double integrator. Commands are `[roll, pitch, Euler_yaw_rate, thrust]`. The 3900/s thrust lag is solved analytically inside RK4 stages; the other 12 states use steps no larger than 1 ms. Collision evaluation retains every integration substep's position.

The local MATLAB fixture validation measured FP64 derivative error approximately `2.27e-13` and maximum 1 ms-grid position error `1.38e-5 m` over six 0.1 s held-command cases. The stiff early body-rate transient differs by approximately `0.0763 rad/s`, separately exposed by the validation budget and half-step convergence check; endpoint rate error is much smaller. These are finite software-model checks, not long-horizon or hardware guarantees.

The measured horizontal acceleration-tracking error maximum is `0.169249 m/s²`; adding the declared `0.05 m/s²` engineering margin gives `0.219249 m/s²`. Startup with zero actual thrust is included. The separate 3D bound is `9.86 m/s²`, dominated by vertical startup gravity, and is not substituted into the horizontal barrier. Calibration covers the declared ramp profiles and software envelope; it does not certify every possible command history. Hardware position/velocity-error bounds are unimplemented, not assumed known.

## Stage 2: differentiable closed-loop co-design

Defaults are 30 epochs, batches of eight trajectories, 200 policy steps at 0.1 s (up to 20 s), with gradients truncated every ten policy steps. Each rollout begins at a recorded trajectory's initial position with the plant's other initial states zero, including actual thrust. Interior thrust states are not fabricated from recorded commands. Each subsequent state is integrated from the issued command; recorded future poses and scans never replace a transition.

Initial enrollment is strict. A trajectory must pass the simulator's true footprint-clearance check, the sensed-sector `h≥0` and `ψ1≥0` checks, angular/input validity, and initial hard-QP feasibility/convergence before any plant step counts as flight. Rejected starts stay in the denominator and records with `steps_simulated=0`, `initial_state_safe=false`, and `termination="invalid-initial-state"`; separate `initial_truth_clearance_ok`, `initial_cbf_valid`, and `initial_qp_feasible` flags explain the rejection. They are not removed or counted as successful flight. The truth-clearance check is an offline simulator admission/evaluation check, not a map input to online control.

The simulator renders the current clean sensor view from static-box geometry at the rolled pose. Geometry supplies simulated sensor measurements, detached clean reconstruction targets, collision labels, and training/evaluation losses. It is not an online input to the reconstructor, gain network, nominal controller, or barrier QP. At future deployment the real LiDAR replaces this simulated sensor.

The first 5 s are measured. Training then includes repeated short/long and partial outages; evaluation uses a fixed repeatable failure schedule. Available sectors use measurements; unavailable sectors use the model prediction. The fixed nominal controller tracks a C2 quintic reference to the actual mission goal, with sector repulsion. Its tracking tube radius is 2.5 m. This is a **new frozen differentiable controller**, not an imitation of the original VFH/PID controller.

Only horizontal acceleration goes through the BarrierNet QP. Vertical tracking and goal-facing yaw remain separate bounded controllers. The QP enforces obstacle, visibility, acceleration, and slew inequalities; only tracking has a slack variable. A learned bounded positive `k2` adapts conservatism while `k1`, uncertainty bounds, and physical margins remain fixed during each solve. The reconstruction bounds are recomputed on the reserved calibration runs for each candidate checkpoint, with no gradient through the bounds.

The total objective combines reconstruction, reference tracking, control correction, true constraint violation, tracking slack, and geometry/altitude penalties. Both reconstruction and barrier-gain weights receive gradients through the actual QP and plant. Mixed precision applies to neural-network work; QP arithmetic explicitly disables autocast and uses FP64. Dynamics compilation targets one reusable RK4 substep, avoiding a giant unrolled compilation of the whole rollout.

Validation and final test use up to **600 steps / 60 s** per enrolled trajectory. Invalid/infeasible filters, collisions, or altitude failures can terminate a trajectory earlier; the saved duration and termination reason must accompany aggregate scores. Checkpoint selection includes validation reconstruction, closed-loop loss, collision rate, and filter-failure rate. Test is evaluated only after selection. No test metric changes the selected checkpoint.

Before creating the first test windows, the pipeline writes `test_evaluation_intent.json`, reserving test access for the selected checkpoint hash and experiment signature. That reservation closes fitting for this experiment even if evaluation is interrupted. Recovery uses the same weights with `--stage evaluate --resume`; a changed checkpoint is rejected. Once `test_evaluation.json` exists, recovery reuses the sealed result and completes/verifies deployment and `summary.json` without rerunning the test.

## Run, resume, and inspect

With `REPO_ROOT`, `OUT`, and `CALIBRATION` set in the notebook:

```python
command = [
    sys.executable, str(REPO_ROOT / "scripts/train_augmented_saits_barrier.py"),
    "--repo-root", str(REPO_ROOT), "--output-dir", str(OUT),
    "--calibration", str(CALIBRATION),
    "--config", str(REPO_ROOT / "configs/augmented_saits_barrier.json"),
    "--stage", "all", "--device", "cuda",
]
subprocess.run(command, check=True, cwd=REPO_ROOT)
```

`--stage pretrain`, `--stage finetune`, and `--stage evaluate` run the corresponding parts with their required prior artifacts. Before test reservation, `--resume` resumes the same experiment from completed checkpoints, preserving source/data/calibration/configuration checks and random-generator, optimizer, scheduler, and gradient-scaler state. Epoch limits can be extended before test reservation; changed splits, fault policies, geometry, architecture, or optimizer settings require a new experiment directory. After reservation, use `--stage evaluate --resume` only, with unchanged selected weights; fitting is closed. Exact replay across CUDA/PyTorch versions is not promised. CPU selection is explicit `--device cpu` for verification, not a substitute for a measured T4 run.

Review the experiment metadata, nested fitting/calibration IDs, normalizer fitting IDs, stage histories, selected checkpoints, and held-out summaries. For closed-loop evaluation inspect mission success, collision/near-clearance rates, filter-failure rate, minimum clearance, altitude failures, actual outage exposure, QP violations, and termination reasons. A rollout that fails before its first outage is not evidence of reconstruction-driven control during sensor failure. Preserve the saved traces and compare healthy, persistence, fixed-gain, and learned-gain results wherever reported; read their actual metrics rather than inferring improvement from a decreasing training loss.

## Local validation completed before Colab training

The local CPU checks and a one-epoch, reduced-horizon workflow using all 80 actual episodes passed through recorded-data Stage 1, closed-loop Stage 2, reserved/sealed test evaluation, and deployment export. The CUDA-specific check was skipped on the local CPU host. These checks establish that the workflow executes and its artifacts agree; the short workflow does not assess learned performance, long-horizon navigation, or control during outages. It is not a completed T4 training experiment. Retain the full-horizon Colab metrics as the experiment's eventual evidence.

A separate six-second, two-training-route CPU check exercised the first complete outage and gradient updates. The undertrained one-epoch weights triggered filter failure on the first unavailable scan for both routes; each was recorded as failed and terminated. This verifies the outage/failure path, not successful outage navigation. The independent control-only gradient test confirms a nonzero reconstructor gradient through the QP, allocation, and 13-state plant without a reconstruction-loss contribution.

## Export for the later Simulink work

The full `all` workflow exports the selected Stage-2 checkpoint to `OUT/deployment`. For a separate new destination:

```python
subprocess.run([
    sys.executable, str(REPO_ROOT / "scripts/export_augmented_saits_barrier.py"),
    "--checkpoint", str(OUT / "stage2/best.pt"),
    "--output-dir", str(OUT / "deployment_manual"),
], check=True, cwd=REPO_ROOT)
```

An existing export destination is refused. The package includes model/barrier weights, preprocessing, empirical bounds, physical parameters, reference cases, and provenance, in portable PT/MAT-v5/JSON/NPZ containers. It excludes optimizer state and the obstacle map. Optional `--onnx` additionally exports batch-one reconstruction only; it does not contain the online QP or establish ONNX runtime numerical parity. Download the deployment ZIP and retain the selected checkpoint and evaluation summaries.

The previous single-episode `model.mat`/MEX inference path cannot load this artifact. Future Simulink integration must implement and parity-check the augmented temporal/state encoders, causal feature/mask/anchor state, frozen nominal controller/reference, learned-gain online QP, allocation, and matching plant timing. Export is preparation for that work, not evidence that the new controller has already run in Simulink.

The safety layer follows the differentiable-HOCBF idea introduced by [BarrierNet](https://arxiv.org/abs/2111.11277). The project's restricted sensing, finite empirical bounds, sampled implementation, and infeasibility policy require their own assessment; the paper's guarantees do not transfer automatically.
