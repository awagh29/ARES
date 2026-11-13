# Single-episode CNN inference in the existing UAV simulation

Open `ARES.prj`, then run `scripts/captureCnnMlpSingleEpisodeSingleUav.m`.
The selected source episode is initially:

```matlab
CnnEpisodeIdOverride = "f9ab72ef-e1d7-4479-8faa-03bb5b36776c";
project = currentProject;
run(fullfile(project.RootFolder,"scripts","captureCnnMlpSingleEpisodeSingleUav.m"));
```

The script finds the latest **finished** single-episode training run under
`results` whose `source.json` matches that ID, selects that run's `best.pt`,
exports native MATLAB weights, checks PyTorch/MATLAB prediction parity, then
runs the existing Simulink plant with the original seed-513 outage schedule.
An unrelated or still-running training job cannot be selected.

Only the recorded initial position and first waypoint change with the source
episode ID. The final goal, obstacle scene, plant, VFH/PID controller, timing,
and intermittent fault schedule retain the existing configuration. For the
initial episode, the start is `[0.56742387771851766, 0, -7]` NED meters.
Each capture receives a new simulation UUID; its manifest records the source
episode ID and checkpoint hash separately.

## Optional explicit selections

```matlab
% Pin one finished training directory instead of automatic latest-run selection:
CnnTrainingRunDirectoryOverride = "A:\Projects\ARES\results\output_directoryZ\single_episode\<training-run>";
CnnShowSimulationOverride = true;  % false runs without the live diagram/viewer.
% CnnPythonExecutableOverride can select the export interpreter if needed.
```

Clear an override before returning to automatic selection:

```matlab
clear CnnTrainingRunDirectoryOverride
```

## Artifacts and model behavior

The run-ID-based deployment pointer is:

`data/models/cnn_mlp_single_episode/<source-episode-id>/deployment.json`

Its `model_path` points to a versioned `model.mat`. The accompanying export
manifest records the exact training directory, checkpoint, source episode
hashes and validation score. `matlab_validation.json` records actual native
MATLAB/PyTorch numerical parity; `integration_validation.json` is produced by
`validateCnnMlpSingleEpisode` when its additional isolation checks are run.
The export does not change the training checkpoint. No Python host, ONNX
importer, or Deep Learning Toolbox is needed during simulation inference;
Python/PyTorch/SciPy are used only for export.

The model receives `[position(3), velocity(3), roll/pitch/yaw(3)]`, encodes
each angle as sine/cosine, and predicts the complete `15 x 716` range grid.
During healthy scans, measured points pass through. During outages, only
motion state enters inference. Predicted ranges are clipped to `[0,7]` m
and projected onto the recorded FLU sensor rays; zero/max-range values are
excluded from the point cloud. The original VFH/PID consumes that cloud.
Raw predictions, encoded input features, use flags and timings are logged.
Dense prediction log entries are zero placeholders when reconstruction is not
used; apply the logged reconstruction-use flag when scoring them.

Telemetry, manifest, scan CSV and summary are saved under:

`data/cnn_single_episode_intermittent/<new-simulation-run-id>/`

The factory baseline model file is not edited. Integration uses a separate
copy under `results/cnn_mlp_integration/<source-episode-id>/`. A changed
experiment file or a stale artifact validation is rejected rather than
silently overwritten. The live viewer uses the selected controller cloud.

To re-export/validate without a flight:

```matlab
project = currentProject;
root = string(project.RootFolder);
[modelPath,deployment] = ares.inference.exportCnnSingleEpisodeModel( ...
    root,CnnEpisodeIdOverride,fullfile(root,".venv","Scripts","python.exe"),"");
report = validateCnnMlpSingleEpisode(root,CnnEpisodeIdOverride);
```

Dense outage errors are scoring against nominal LiDAR recorded on the flown
trajectory. They are not the notebook's held-out validation loss, and successful
numerical/instrumentation checks do not imply mission or collision success.
