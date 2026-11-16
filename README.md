# ARES

ARES is a research project for single-UAV navigation with intermittent or unavailable LiDAR. It combines a MATLAB/Simulink quadrotor simulation, reproducible telemetry and dataset collection, Python reconstruction training, and closed-loop inference experiments.

The baseline uses LiDAR obstacle avoidance with a VFH/PID controller. Reconstruction experiments supply estimated observations when measurements are unavailable. A separate augmented SAITS/BarrierNet experiment learns reconstruction and barrier gains and executes its own controller and constrained optimization layer.

This repository contains experimental implementations and recorded simulation data. A completed capture, low reconstruction error, or successful numerical-parity check does not establish collision-free navigation. Evaluate the saved flight outcomes and failure indicators for each experiment.

## Contents

- [Capabilities](#capabilities)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start: capture a baseline flight](#quick-start-capture-a-baseline-flight)
- [Explore the recorded dataset](#explore-the-recorded-dataset)
- [Collect and export new episodes](#collect-and-export-new-episodes)
- [Train reconstruction models](#train-reconstruction-models)
- [Run reconstruction in Simulink](#run-reconstruction-in-simulink)
- [Augmented SAITS and barrier control](#augmented-saits-and-barrier-control)
- [Configuration and reproducibility](#configuration-and-reproducibility)
- [Validation and tests](#validation-and-tests)
- [Troubleshooting](#troubleshooting)
- [Documentation guide](#documentation-guide)
- [Development and artifact handling](#development-and-artifact-handling)

## Capabilities

- Simulate one quadrotor navigating a static obstacle scene toward a mission goal.
- Capture native plant, controller, LiDAR, and fault-schedule telemetry with UUID run identities and provenance manifests.
- Collect repeatable North/East axis-start batches and export scan-aligned episodes to MATLAB, CSV, and NumPy formats.
- Model intermittent LiDAR availability with deterministic outages and recovery.
- Train compact sector-based SAITS, dense single-episode SAITS, CNN/MLP, and sparse-convolution reconstruction models.
- Export supported trained models to native MATLAB inference with reference cases for PyTorch/MATLAB comparison.
- Train and evaluate a state-augmented SAITS reconstructor with a differentiable barrier QP and calibrated 13-state guidance dynamics.
- Inspect predictions, observation routing, commands, reconstruction errors, clearances, mission outcomes, and controller failures in saved artifacts.

## Repository layout

```text
ARES/
├── ares.prj                       MATLAB project entry point
├── models/                        Saved baseline Simulink model
├── src/+ares/
│   ├── +config/                   Default single-UAV configuration
│   ├── +data/                     Data contracts, manifests, telemetry, batch planning
│   └── +inference/                MATLAB predictors and experiment integrations
├── scripts/
│   ├── ares_safe_training/        Augmented reconstruction, dynamics, QP, training
│   ├── eda/                      MATLAB exploratory analysis
│   └── ...                       Capture, export, training, validation entry points
├── notebooks/                     Exploration and model-training notebooks
├── configs/                       Dense SAITS and augmented training configuration
├── data/
│   ├── raw/                       Native telemetry and per-run manifests
│   ├── processed/lidar_episode_v1/ Scan-aligned MAT/CSV/NPZ episodes
│   ├── batches/                   Plans, indexes, checkpoints, and dataset QA
│   ├── calibration/               Dynamics calibration and numerical fixtures
│   ├── models/                    Exported model weights and deployment metadata
│   └── ...                       Scenario-specific captured experiments
├── tests/data/                    MATLAB unit tests
├── tests/python/                  Python unit and integration tests
├── docs/                          Dataset contracts and experiment walkthroughs
├── resources/project/             MATLAB project metadata
├── saits/                         Saved endpoint-SAITS training artifacts
├── results/                       Local training, validation, and generated models
├── requirements-colab-safe-training.txt
└── .github/workflows/matlab-ci.yml
```

`results/`, virtual environments, compiled MEX binaries, and Simulink caches/build directories are ignored by Git. A fresh checkout may therefore lack training checkpoints or deployment bundles referenced by a local experiment guide.

## Requirements

### MATLAB simulation

The CI workflow targets **MATLAB R2026a on Windows** with:

- Simulink
- UAV Toolbox
- Navigation Toolbox
- Robotics System Toolbox
- Control System Toolbox
- Stateflow

Use this as the reference simulation environment; compatibility with other releases or operating systems is not established here. MATLAB desktop is needed for interactive diagrams and animation. Unattended capture is supported by the relevant experiment runners.

Native MATLAB reconstruction inference does not require Deep Learning Toolbox or an online Python process. Exporting Python checkpoints requires Python and the appropriate training dependencies. The augmented barrier integration additionally requires a MATLAB-compatible Python interpreter running out of process. The optional legacy SAITS MEX backend requires a supported C++ compiler.

### Python analysis and training

- Dataset exploration: NumPy and pandas; matplotlib for plots.
- MAT-to-NPZ batch export: NumPy and h5py.
- Neural training/export: PyTorch, NumPy, pandas, SciPy, and any notebook-specific plotting dependencies.
- Augmented training: the dependency inventory in [requirements-colab-safe-training.txt](requirements-colab-safe-training.txt), including PyTorch `>=2.5`; CUDA is expected for the full GPU workflow.

Create a local environment from the repository root, for example in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy matplotlib h5py
```

For neural workflows, install a PyTorch build appropriate for the machine and its GPU. The augmented workflow's dependency file can then be used with that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-colab-safe-training.txt
```

In Colab, retain its working CUDA-enabled PyTorch installation and install missing dependencies additively. [notebooks/requirements.txt](notebooks/requirements.txt) is a pinned environment inventory containing CUDA-specific wheel versions; it is not a portable installation recipe for every host. Reading existing NPZ episodes requires neither MATLAB nor a GPU.

### Repository data

MAT files are configured for Git LFS in [.gitattributes](.gitattributes). Install Git LFS and retrieve the real binary contents when cloning:

```powershell
git lfs install
git lfs pull
```

Keep the repository's `data/` hierarchy intact. Opening a notebook alone does not place its dataset, calibration, or checkpoints into the runtime.

## Quick start: capture a baseline flight

Start MATLAB in the repository root, or change to that directory first. Open the project and explicitly add the code directories:

```matlab
proj = openProject(pwd);
projectRoot = string(proj.RootFolder);
addpath(fullfile(projectRoot,"src"));
addpath(fullfile(projectRoot,"scripts"));

% Avoid inheriting a previous experiment's configuration.
clear SingleUavConfigOverride SingleUavLidarScheduleOverride

% Simulate and persist a fresh nominal episode.
run(fullfile(projectRoot,"scripts","captureNominalBaselineSingleUav.m"));

% Inspect saved telemetry and generate validation output.
report = validateNominalBaselineTelemetry(telemetryPath);
disp(manifestPath);
disp(telemetryPath);
```

The capture saves a new run under `data/raw/<run-UUID>/`, including `manifest.json` and a telemetry MAT file. It leaves `simOut`, `manifestPath`, and `telemetryPath` available in the calling workspace. The validator creates diagnostic figures and reports alongside the run.

To run the nominal simulation without the capture/export workflow:

```matlab
run(fullfile(projectRoot,"scripts","runSingleUav.m"));
```

The default configuration is [src/+ares/+config/defaultSingleUav.m](src/+ares/+config/defaultSingleUav.m): a 60 s maximum horizon, 1 ms simulation sample time, a 0.1 kg vehicle, and final waypoint `[30,35,-7]` in North-East-Down coordinates. Flights may terminate earlier through waypoint logic. Negative Down corresponds to positive altitude.

`runSingleUav.m` checks finite trajectory values, final waypoint error, and altitude limits. Its obstacle-clearance assertion is currently commented out, and its older clearance calculation is not the axis-start dataset's frame-corrected outcome evaluator. Its printed PASS message should not be read as a collision-safety result.

## Explore the recorded dataset

The documented reference batch is `c3eaab07-d882-4825-98a5-84c278cf8935`: **80 runs, 19,185 scans**, with 40 North-axis starts and 40 East-axis starts in the same obstacle scene. These are healthy-sensor collection episodes; synthetic faults are introduced separately for learning experiments.

From a notebook running at the repository root:

```python
from pathlib import Path
import sys

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "scripts"))

from colab_episode_starter import EpisodeDataset, make_run_splits, lidar_grid

dataset = EpisodeDataset(repo_root)
runs = dataset.runs
scans = dataset.load_scan_table()
episode = dataset.load_episode(runs.iloc[0]["RunId"])

print(f"{len(runs)} runs; {len(scans):,} scans")
print(episode["lidar_ranges_m"].shape)
print(episode["position_ned_m"].shape)
```

Set `repo_root` explicitly if the notebook starts inside `notebooks/` or a Colab checkout. The loader resolves episodes relative to that root; stored collection-machine Windows paths remain provenance.

Important data conventions:

- Dense ranges have shape `[scans,10740]`, representing a **15 × 716** organized grid. Elevation varies fastest in MATLAB column-major order. Use `lidar_grid` for plotting.
- Position and velocity use NED; organized XYZ scans use the LiDAR sensor frame, with x forward, y left, and z up. Scenario meshes require the documented ENU/NED conversion for geometric checks.
- Maximum LiDAR range is 7 m. A healthy no-return beam has range 7 and a false valid-return mask. This differs from an unavailable sensor measurement.
- State is simulator truth. Continuous signals are aligned to scan times; commands/reference use previous-value hold. Extrapolation is prohibited.
- The first previous-action row is unavailable, with NaN values and a false availability flag.
- The final mission goal and the avoidance-adjusted controller reference are separate fields.
- Collection completion describes instrumentation, not flight success. The documented batch retains 33 clearance violations and two sampled collisions.

Split complete runs before constructing windows, injecting faults, or fitting normalization. The generic `make_run_splits` helper produces **56/8/8/8 training/validation/calibration/test runs**. The augmented training pipeline uses its own distinct split contract described below. Stream dense episodes as needed to control memory use.

See [the dataset contract](docs/LIDAR_AXIS_DATASET.md) and [the Colab exploration guide](docs/COLAB_EPISODES.md).

## Collect and export new episodes

With the MATLAB project open and `projectRoot` defined:

```matlab
[batchPath,index] = captureLidarAxisBatch(projectRoot);
index = finalizeLidarAxisBatch(batchPath);
```

Resume an interrupted serial collection with:

```matlab
[batchPath,index] = captureLidarAxisBatch( ...
    projectRoot,ResumeBatchPath=batchPath);
```

Collection records the plan, seeds, configuration, source identities, software, outcomes, and immutable attempt records. Resume verifies completed artifacts and rejects incompatible source/software or output-root changes. Concurrent collection uses prepared batches and disjoint worker assignments; follow the [dataset guide](docs/LIDAR_AXIS_DATASET.md) before launching workers.

After finalization, export NPZ episodes from a terminal using your Python environment:

```powershell
.\.venv\Scripts\python.exe scripts/exportLidarBatch.py --batch "data/batches/YOUR_BATCH_ID"
```

The exporter verifies the dataset, checks arrays against their source, and writes `python_index.csv` and `python_export_qa.json`. Existing exports are verified and retained. Batch plans/indexes live in `data/batches/<batch-id>/`; raw runs live in `data/raw/<run-id>/`; processed episodes live in `data/processed/lidar_episode_v1/<run-id>/`.

## Train reconstruction models

Choose the notebook for the intended architecture and dataset scope:

- [ARES_Reconstruction.ipynb](notebooks/ARES_Reconstruction.ipynb): existing data exploration and reconstruction preparation.
- [train_cnn_mlp_baseline.ipynb](notebooks/train_cnn_mlp_baseline.ipynb): CNN/MLP baseline workflow.
- [train_cnn_mlp_baseline_single_episode.ipynb](notebooks/train_cnn_mlp_baseline_single_episode.ipynb): single-episode CNN/MLP training and checkpoint production.
- [train_saits_baseline_single_episode.ipynb](notebooks/train_saits_baseline_single_episode.ipynb): dense single-episode SAITS reconstruction, with [configuration](configs/dense_saits_single_episode.json).
- [train_sparse_reconstruction_single_episode_baseline.ipynb](notebooks/train_sparse_reconstruction_single_episode_baseline.ipynb): sparse-convolution range-grid reconstruction with explicit support masks and state conditioning.
- [train_augmented_saits_barrier.ipynb](notebooks/train_augmented_saits_barrier.ipynb): the separate two-stage augmented reconstruction/control experiment.

The compact endpoint-SAITS workflow uses [SINGLE_EPISODE_SAITS.md](docs/SINGLE_EPISODE_SAITS.md), [TRAIN_SAITS_T4.md](docs/TRAIN_SAITS_T4.md), and `scripts/train_saits_t4.py`. Its input is a **50 × 31** causal window; its 14 outputs are seven forward-sector minimum distances and seven return fractions. It does not reconstruct the complete dense grid.

The early 302-scan single-episode endpoint experiment has constant no-return targets throughout validation and test; persistence already achieves zero error there. Treat it as a pipeline check. Single-episode held-out windows and partial-spatial-gap validation do not establish generalization to new flights or whole-scan outages.

Preserve the selected `best.pt`, preprocessing, configuration, source metadata, split assignment, and evaluation reports. Renaming a checkpoint to `model.mat` does not export it. Exporters require the matching architecture and metadata, and generally refuse an existing destination.

## Run reconstruction in Simulink

Run one scenario at a time with the project open. Use capture scripts to save a new experiment; the diagram's Run button can replay a model without executing the telemetry-export step. Training checkpoints and their metadata must already be present for runners that automatically export a trained model.

### Fault baselines

- `scripts/captureLidarDisabledSingleUavRun.m`: disabled-LiDAR capture.
- `scripts/captureIntermittentLidarDisabledSingleUav.m`: intermittent-availability experiment without a reconstruction model.

The intermittent schedule is generated by `ares.inference.createIntermittentSchedule`, using private random-stream seed 513. Nominal simulated scans may remain in saved telemetry for scoring; they must not enter a predictor while declared unavailable.

### Compact endpoint SAITS

Explicitly select the legacy backend so local augmented deployment files do not change the experiment:

```matlab
SaitsBackendOverride = "legacy";
SaitsShowSimulationOverride = true;
run(fullfile(projectRoot,"scripts","captureSaitsIntermittentLidarSingleUav.m"));
```

The default artifact is `data/models/saits_endpoint/model.mat`. Validate it first:

```matlab
artifactPath = fullfile(projectRoot,"data","models","saits_endpoint","model.mat");
report = validateSaitsMatlab(artifactPath,BenchmarkCount=100,AbsoluteTolerance=1e-4);
assert(report.allParityChecksPassed);
```

This path supplies a coarse sector proxy to obstacle avoidance, with history-readiness fallback and unknown-space guards. It is distinct from dense-grid reconstruction. The optional validated MEX backend accelerates this legacy predictor. See [SAITS_MATLAB_PIPELINE.md](docs/SAITS_MATLAB_PIPELINE.md) for export, parity, routing, and telemetry details.

### Dense single-episode models

The following runners select the source episode and corresponding completed training artifacts, export weights, validate numerical agreement, and capture a flight:

```matlab
% Choose one command for the experiment you trained.
run(fullfile(projectRoot,"scripts","captureCnnMlpSingleEpisodeSingleUav.m"));
run(fullfile(projectRoot,"scripts","captureSaitsSingleEpisodeSingleUav.m"));
run(fullfile(projectRoot,"scripts","captureSparseReconstructionSingleEpisodeSingleUav.m"));
```

The default source episode is `f9ab72ef-e1d7-4479-8faa-03bb5b36776c`. Runner-specific overrides can pin another source episode, a finished training directory, Python executable, or viewer preference. See the individual guides before changing them:

- [CNN/MLP deployment and verification](docs/CNN_MLP_SINGLE_EPISODE_MATLAB_RUN.md)
- [Dense SAITS single-episode deployment](docs/SAITS_SINGLE_EPISODE_MATLAB_RUN.md)
- [Sparse reconstruction deployment](docs/sparse_reconstruction_single_episode_matlab.md)

Dense SAITS and sparse reconstruction also have fully disabled scenario runners: `captureSaitsLidarDisabledSingleUav.m` and `captureSparseReconstructionLidarDisabledSingleUav.m`. Sparse reconstruction processes one scan at a time without a history buffer; its fully disabled scenario supplies no LiDAR measurements from time zero.

Exported versions are stored under `data/models/<model-family>/<source-episode>/<version>/`, with `deployment.json` pointers where supported. Captures use scenario-specific directories under `data/`, normally containing a manifest, telemetry MAT file, `scans.csv`, and `summary.json`. Read the manifest to identify the exact artifact and behavior actually used.

## Augmented SAITS and barrier control

This experiment combines a temporal encoder, current-state encoder, learned bounded barrier gain, hard-inequality QP, frozen nominal controller, and calibrated guidance dynamics. It replaces the old VFH/PID command output in its Simulink experiment copy. Its complete deployment bundle is incompatible with the legacy endpoint-SAITS MAT/MEX path.

### Train and export

Follow [AUGMENTED_SAITS_BARRIER_COLAB.md](docs/AUGMENTED_SAITS_BARRIER_COLAB.md) and the [barrier derivation](docs/LIDAR_BARRIER_DERIVATION.md). Required inputs include all 80 reference episodes, dataset/export QA and raw manifests, and the dynamics calibration with fixtures and a matching validation report.

The outer split is **56 training / 12 validation / 12 test runs**. Twelve training runs are reserved for empirical reconstruction-error calibration, leaving 44 for gradient fitting and normalization. Stage 1 pretrains reconstruction; Stage 2 differentiates through closed-loop QP and plant rollouts. These splits are separate from the generic exploration helper's split.

From the repository root, with the selected Python environment active:

```powershell
python scripts/validate_dynamics_calibration.py --calibration data/calibration/saits_barrier_dynamics/de5533e2-d348-4e43-a06f-4e5bb2e50055/calibration.json
python scripts/train_augmented_saits_barrier.py --repo-root . --output-dir results/augmented_saits_barrier/YOUR_NEW_RUN --calibration data/calibration/saits_barrier_dynamics/de5533e2-d348-4e43-a06f-4e5bb2e50055/calibration.json --config configs/augmented_saits_barrier.json --stage all --device cuda
```

Use a new experiment directory for changed splits, architecture, fault policy, or optimization configuration. Before test reservation, `--resume` restores compatible completed checkpoints. Once `test_evaluation_intent.json` reserves evaluation for selected weights, fitting is closed; recovery uses `--stage evaluate --resume` with unchanged weights. The full pipeline exports a deployment bundle under the output directory. Colab output must be saved to mounted persistent storage to survive runtime deletion.

### Deploy in MATLAB

Supply the **full exported deployment directory** and a compatible Python environment:

```matlab
SaitsBackendOverride = "augmented-barrier";
SaitsDeploymentPathOverride = fullfile(projectRoot,"results", ...
    "augmented_saits_barrier","YOUR_RUN","deployment");
SaitsPythonExecutableOverride = fullfile(projectRoot,".venv","Scripts","python.exe");
SaitsShowSimulationOverride = true;
SaitsContinueAfterBarrierFailureOverride = false;

% Validate the bundle and compile the experiment without flying.
run(fullfile(projectRoot,"scripts","prepareSaitsBarrierSingleUav.m"));

% Capture a fresh flight.
run(fullfile(projectRoot,"scripts","captureSaitsIntermittentLidarSingleUav.m"));
```

The script default is diagnostic continuation after an invalid/infeasible filter. The example explicitly disables continuation so the experiment stops on such a failure. In diagnostic mode the nominal fallback is uncertified; reaching the goal after a filter failure still does not make the run successful under the experiment's safety checks.

Outputs are saved under `data/saits_barrier_intermittent/<UUID>/`. The [MATLAB run guide](docs/SAITS_BARRIER_MATLAB_RUN.md) reports 100% filter failure and 0% mission success for its documented local deployment's held-out evaluation. Preserve those limitations when interpreting that artifact; use the reports belonging to a newly trained bundle for its results.

Without an explicit backend override, the shared runner may choose the local augmented deployment when present, or use the legacy path otherwise. Set the backend explicitly for reproducible comparisons.

## Configuration and reproducibility

- **Vehicle and mission:** `src/+ares/+config/defaultSingleUav.m` defines defaults. `SingleUavConfigOverride` and `SingleUavLidarScheduleOverride` are consumed by initialization; experiment runners may supply their own values.
- **Simulation initialization:** `scripts/initializeSingleUav.m` creates scenario objects, sensor settings, obstacles, and controller variables; `setupSingleUavBusObjects.m` defines required buses.
- **Effective sensor parameters:** inspect the saved model and run manifest. Initializer variables alone do not establish the effective block settings. The documented axis-start batch uses 10 Hz LiDAR, 7 m range, noise off, and 0.02 m range accuracy.
- **Data schema:** `ares.data.defaultDataContract()` defines schema `2.0.0` for single-UAV telemetry.
- **Training:** JSON configurations live in `configs/`; notebook and checkpoint metadata define the actual selected architecture and feature contract.
- **Provenance:** manifests and deployment metadata retain UUIDs, seeds, configuration, source/data/checkpoint hashes, validation, and outcomes. Preserve them with their associated binary artifacts.

Workspace overrides can persist between experiments. Clear unneeded overrides or use a fresh MATLAB session when changing workflows. Do not combine different models' preprocessing or feed scoring-only nominal LiDAR to a missing-data branch.

## Validation and tests

### MATLAB tests

With the project open:

```matlab
testResults = runtests(fullfile(projectRoot,"tests","data"));
assertSuccess(testResults);
```

The suite covers data contracts, telemetry/episode behavior, axis-start planning, inference, and selected outage scenarios. Some checks require local model artifacts or simulation dependencies; inspect the test diagnostics for prerequisites.

Run artifact-specific parity and integration validators before interpreting a new model's flight. Relevant entry points include `validateSaitsMatlab`, `validateCnnMlpSingleEpisode`, `validateSparseReconstructionSingleEpisode`, `validateSaitsBarrierMatlab`, and `verifyCnnMlpSingleEpisode`. Their arguments and scope are documented in the corresponding deployment guides and function headers.

### Python tests

From the repository root with the intended environment active:

```powershell
python -m unittest discover -s tests/python -p "test_*.py"
```

Tests cover loading, split/mask contracts, training/export behavior, inference parity fixtures, calibrated dynamics, QP constraints/gradients, and runtime integration. Some dependency- or hardware-specific checks skip when unavailable; inspect the test summary before treating a run as full coverage.

### Continuous integration

[matlab-ci.yml](.github/workflows/matlab-ci.yml) runs on pushes and pull requests to `main`, and supports manual dispatch. It provisions MATLAB R2026a and the listed products, validates bus definitions, compiles the baseline model, captures a fresh nominal episode, checks its LiDAR telemetry, and uploads diagnostic artifacts. It does not run every Python test or validate every learned controller.

## Troubleshooting

- **`currentProject` or `ares.*` cannot be resolved:** open `ares.prj` and add the root `src` and `scripts` directories to the MATLAB path. Run from the repository checkout.
- **Missing episodes or LFS pointer errors:** retrieve actual data files and preserve batch/processed directories. Use `EpisodeDataset` in Colab to resolve checkout-relative paths instead of collection-machine paths.
- **No completed training checkpoint found:** train or restore the required run, including metadata and `best.pt`. `results/` is ignored by Git; a fresh clone does not restore it. Pin the training directory through the runner's documented override if needed.
- **Artifact/source/hash mismatch:** use the matching source, checkpoint, preprocessing, and deployment bundle. Re-export to a new directory and rerun validation rather than editing recorded hashes.
- **MATLAB has another Python interpreter loaded:** restart MATLAB, select the intended interpreter override, and rerun augmented preparation. The barrier runtime requires out-of-process execution.
- **Colab reports no CUDA:** select a GPU runtime and verify `torch.cuda.is_available()`. Retain the runtime's working CUDA-enabled PyTorch. Use explicit CPU mode only where supported for verification.
- **Simulation is slow:** neural inference and visualization can dominate runtime. Disable the viewer through the runner's override for unattended capture. Legacy MEX acceleration does not accelerate dense or augmented models.
- **A generated experiment model cannot be replaced:** preserve or move its edited copy before regenerating. Runners protect user edits; the saved baseline remains the reference model.
- **The goal was reached but the run failed:** inspect clearance, collisions, altitude, QP/filter failures, fallback use, actual outage exposure, and termination reason. Goal attainment and safety-qualified mission success are separate metrics.

## Documentation guide

- [LiDAR axis-start dataset and collection](docs/LIDAR_AXIS_DATASET.md)
- [Load and explore episodes in Colab](docs/COLAB_EPISODES.md)
- [Compact single-episode SAITS preparation](docs/SINGLE_EPISODE_SAITS.md)
- [Endpoint-SAITS training on a Colab T4](docs/TRAIN_SAITS_T4.md)
- [Legacy endpoint-SAITS export and online integration](docs/SAITS_MATLAB_PIPELINE.md)
- [Dense single-episode SAITS MATLAB runner](docs/SAITS_SINGLE_EPISODE_MATLAB_RUN.md)
- [CNN/MLP single-episode MATLAB runner](docs/CNN_MLP_SINGLE_EPISODE_MATLAB_RUN.md)
- [Sparse reconstruction MATLAB runners](docs/sparse_reconstruction_single_episode_matlab.md)
- [Augmented SAITS/BarrierNet training](docs/AUGMENTED_SAITS_BARRIER_COLAB.md)
- [Barrier formulation, assumptions, and limitations](docs/LIDAR_BARRIER_DERIVATION.md)
- [Augmented SAITS/BarrierNet MATLAB deployment](docs/SAITS_BARRIER_MATLAB_RUN.md)

The training and deployment guides describe different stages and artifacts. For the existing augmented MATLAB integration, use the deployment guide alongside the earlier Colab training guide.

## Development and artifact handling

Keep changes to the baseline model deliberate, and use experiment copies for new controller/inference wiring. Retain failed flight outcomes and immutable raw runs; publish corrections or new exports with their provenance instead of silently replacing evidence. Numerical parity, reconstruction quality, and navigation outcomes need separate evaluation.

Back up ignored training outputs and deployment bundles separately. Do not commit virtual environments, generated caches, or machine-specific MEX binaries. Keep MATLAB project metadata under `resources/project/` with the project.

No repository-level license file is currently provided. Confirm usage and redistribution terms with the project owner before distributing the code, datasets, or derived model artifacts.
