# Single-episode sparse reconstruction in MATLAB

The runners deploy `notebooks/train_sparse_reconstruction_single_episode_baseline.ipynb`'s
**SparseRangeReconstructor**, including its sparse convolutions, circular azimuth padding,
support propagation, FiLM state conditioning, U-Net decoder and `7*sigmoid` range head.
They reconstruct the complete **15 x 716** range grid for the original VFH/PID controller.
There is no BarrierNet, certificate gate, training stage two or online Python process.

## Run commands

Open the ARES MATLAB project, then run **one** scenario at a time. Each command exports
the latest completed sparse run for the selected source episode, checks MATLAB/PyTorch
numerical agreement, opens the diagram/animation on the MATLAB desktop and starts capture.

```matlab
openProject('A:\Projects\ARES\ares.prj');
SparseSingleEpisodeShowSimulationOverride = true;

% Original intermittent failures: seed 513, three outages and recoveries.
run('A:\Projects\ARES\scripts\captureSparseReconstructionSingleEpisodeSingleUav.m');
```

After the first run finishes, run the fully disabled scenario separately:

```matlab
run('A:\Projects\ARES\scripts\captureSparseReconstructionLidarDisabledSingleUav.m');
```

The default source episode is `f9ab72ef-e1d7-4479-8faa-03bb5b36776c`. Its recorded
initial position and first waypoint are restored. Both scenarios use the existing
plant, static scene, goal and controller. Only experiment copies of the model are wired;
`models/ares_single_uav_top.slx` is protected.

Optional overrides, set before either runner:

```matlab
SparseSingleEpisodeIdOverride = "f9ab72ef-e1d7-4479-8faa-03bb5b36776c";
SparseSingleEpisodePythonExecutableOverride = "A:\Projects\ARES\.venv\Scripts\python.exe";
% Pin a completed run, rather than automatically selecting the latest:
SparseSingleEpisodeTrainingRunDirectoryOverride = "A:\Projects\ARES\results\sparse_reconstruction_single_episode_baseline\YOUR_RUN_FOLDER";
% Remove a previous explicit selection to return to automatic discovery:
clear SparseSingleEpisodeTrainingRunDirectoryOverride;
```

Use the capture scripts again for a new saved run. Toolbar Run can replay the
scenario but does not execute the script's telemetry export.

## Outage semantics and interpretation

This model processes one scan at a time. It has **no history buffer**. Inputs are ranges,
valid-return flags, availability support and current state. With the completed run's
`use_position=false`, state is velocity plus sine/cosine of roll, pitch and yaw (9 values).
Available no-return beams remain 7 m, with support true and valid-return false.

When LiDAR is disabled, ranges are zero and **both LiDAR masks are false**. The disabled
branch never reads the nominal cloud. No recorded scan, previous prediction or warm-up
scan is supplied. Fully disabled capture applies this from **t=0 through termination**;
intermittent capture switches back to measured clouds immediately on recovery.
The nominal simulated cloud remains logged only for scoring.

The notebook trained on partial spatial azimuth gaps. Whole-scan outages remove more
information, and with position excluded the model must rely on velocity and attitude.
Its partial-gap validation score does not establish performance on these scenarios.
Mission success, collision/clearance results and reconstruction error are measured after
the run. Safety metric failures do not add a certificate-based stop to the controller.

Native inference uses MATLAB FP32 matrix operations on the CPU and needs no Deep Learning
Toolbox importer. GPU training does not make this deployment use the GPU. The numerical
validation report includes measured inference time; the simulation may run slower than
real time at the 10 Hz scan rate.

## Saved output

Every flight receives the existing UUID run ID, with matching manifest and telemetry IDs:

- Intermittent: `A:\Projects\ARES\data\sparse_reconstruction_single_episode_intermittent\<run-id>\`
- Fully disabled: `A:\Projects\ARES\data\sparse_reconstruction_single_episode_disabled\<run-id>\`

Each folder contains `manifest.json`, the scenario telemetry MAT file, `scans.csv` and
`summary.json`. The MAT file includes full predicted ranges, masked model inputs,
availability and valid-return masks, measured state, executed commands and scoring-only
nominal LiDAR. CSV includes scan availability, reconstruction status, timing, current
state, sector summaries, selected/waypoint references and actions. The summary includes
goal/collision metrics, outage/recovery counts, range errors and input-isolation checks.

Exported weights and comparison fixtures are versioned under
`data/models/sparse_reconstruction_single_episode/<source-episode>/<version>/`.
`deployment.json` points to the current export; manifests record checkpoint and source
data hashes. Export compares the notebook's model definitions with the inference module
and strictly loads all checkpoint tensors. The older notebook checkpoints do not contain
a training-time source fingerprint, so this limitation is recorded in export provenance.

## Validate without flying

After the model is exported, this checks numerical parity, outage/cold-start behavior and
both Simulink model compilations without calling `sim`:

```matlab
addpath('A:\Projects\ARES\scripts');
report = validateSparseReconstructionSingleEpisode( ...
    "A:\Projects\ARES", "f9ab72ef-e1d7-4479-8faa-03bb5b36776c");
```

The report is written under
`results/sparse_reconstruction_validation/<source-episode>/integration_validation.json`.
The existing training notebook and saved training results are not modified by export.
