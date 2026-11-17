# Full-scan SAITS: single episode, stage one only

Open `notebooks/train_saits_baseline_single_episode.ipynb`. Its 28 cells follow the same 0–9 section order as `train_cnn_mlp_baseline_single_episode.ipynb`: configuration, selected episode, packages/GPU, preprocessing/model, logging/plots, data loading, training function, tensors, training, and visualizations.

## Train

1. Select your existing GPU `.venv` kernel in VS Code.
2. In the configuration cell set `CONFIG["data"]["episode_path"]` to the raw episode folder or ID. The default matches the current CNN notebook: `f9ab72ef-e1d7-4479-8faa-03bb5b36776c`.
3. Run the notebook in order. It saves a new uniquely named folder under `results/saits_dense_single_episode/` and prints `run_dir`.

Defaults retain the CNN notebook's 80/20 chronological split, AdamW, 200 epochs, batch 16, cosine learning rate, BF16 GPU work, and best/last checkpoints. Early stopping is explicitly disabled by default, matching the current CNN loop; enable `training.early_stopping` to use its configured patience. Configuration, source/split IDs, training-only normalization, metrics CSV/JSON, training log, validation predictions, and the three comparison figures are persisted.

This is **one reconstruction training stage**. SAITS's two attention/imputation blocks belong to the same model. There is no BarrierNet, certificate, differentiable control, dynamics calibration, or closed-loop fine-tuning.

The output is the full **15 elevation × 716 azimuth** range image, in metres. Its serialized 10,740-beam vector is elevation-fastest, matching the recorded NPZ and CNN controller adapter. Predictions and error metrics are unclipped; the same existing range-to-cloud adapter clips to `[0,7]` m for VFH.

SAITS uses more input information than the state-only CNN: a 50-row causal window of measured ranges plus the same 12 encoded position/velocity/attitude variables, with an explicit observation mask. It is a SAITS endpoint adaptation using the existing MIT-attributed DMSA layers; bottleneck dense heads and a bottleneck mixture gate avoid quadratic matrices in the number of beams. It is not compatible with the old 14-sector checkpoint, and should not be represented as the unmodified upstream SAITS benchmark architecture.

Short and long synthetic scan outages alter predictors only. Clean targets stay intact. Windows, fault schedules, and padding are constructed independently inside each chronological fold. Validation never receives training-fold history or future measurements; normalization uses native training rows only. Reconstructed values do not become measured history online. The lazy CPU datasets avoid storing a duplicate dense tensor for every overlapping window.

Validation measures later scans of the **same route**. Check `validation_metrics.no_return_only` in `summary.json`: late scans can contain only the 7 m no-return encoding. Low error there does not establish obstacle reconstruction or unseen-route generalization. The saved persistence baseline and hidden-returning-beam count help interpret that limitation. Selection uses hidden-beam MSE; training curves also include measured endpoints and use a different mask scope.

## Run MATLAB after training

```matlab
openProject("A:\Projects\ARES");
SaitsSingleEpisodeIdOverride = "f9ab72ef-e1d7-4479-8faa-03bb5b36776c";
% Optional: pin the exact run_dir printed by the notebook:
% SaitsSingleEpisodeTrainingRunDirectoryOverride = "A:\Projects\ARES\results\saits_dense_single_episode\YOUR_RUN";
SaitsSingleEpisodeShowSimulationOverride = true;
run("A:\Projects\ARES\scripts\captureSaitsSingleEpisodeSingleUav.m");
```

Without an explicit run directory, the runner selects the most recently created **completed dense SAITS reconstruction-only** run for that episode. It rejects CNN/MLP, old 14-sector SAITS, and augmented/BarrierNet runs. The selected best weights are exported to versioned `data/models/saits_dense_single_episode/<episode-id>/<artifact-version>/model.mat`; `deployment.json` points to the selected export. Source-data hashes, architecture source, split/normalizer provenance, and selected epoch are verified before export. MATLAB then checks CPU FP32 reference cases before wiring the controller.

Successful SAITS summaries use `stage1_completed` or `stage1_early_stopped`. These model-specific states prevent the existing CNN runner's recursive results search from selecting a SAITS checkpoint. The CNN exporter and its current user edits are unchanged.

Automatic selection skips unreadable/malformed run metadata with a warning naming the affected file, and skips interrupted or incomplete runs. It still requires an intact completed run for the requested episode; the selected checkpoint, source hashes, normalization and best epoch are checked as before. Damaged training files are never repaired or overwritten by discovery. An explicitly pinned run does not fall back to another run: invalid metadata raises an error with its exact path. Clear `SaitsSingleEpisodeTrainingRunDirectoryOverride` to return to automatic selection, or pin an intact completed run.

Native MATLAB inference needs no Python host, ONNX importer, or Deep Learning Toolbox online. Python with Torch/NumPy/SciPy is used once for export. Available scans pass through unchanged; during outages the dense SAITS output is projected onto the recorded sensor rays. The original VFH/PID, plant, scene, scan rate, and seed-513 intermittent failures are retained. The source episode ID restores the recorded initial position and first waypoint, matching the CNN workflow. No augmented-controller or barrier-failure flag applies to this runner.

Each simulation creates `data/saits_dense_single_episode_intermittent/<UUID>/` with manifest, telemetry MAT, scan CSV, and summary. Dense predictions and full current predictor values/masks are logged in the MAT; the CSV includes scan events, sector summaries, measured pose features, executed commands, and controller-reference differences. Geometry/raw outage LiDAR is scoring-only. Existing baseline, CNN and augmented-model artifacts are preserved.

## Completely disabled LiDAR from time zero

Use the separate runner to test the same trained single-episode model without any measured LiDAR history:

```matlab
openProject("A:\Projects\ARES");
clear SaitsSingleEpisodeTrainingRunDirectoryOverride % Choose latest intact completed run.
SaitsSingleEpisodeIdOverride = "f9ab72ef-e1d7-4479-8faa-03bb5b36776c";
SaitsSingleEpisodeShowSimulationOverride = true;
run("A:\Projects\ARES\scripts\captureSaitsLidarDisabledSingleUav.m");
```

The full training-run/Python overrides above also apply. This mode sets `LidarEnabled=false` and a constant-zero availability schedule from `t=0` through the entire planned run. There is no initial healthy scan, healthy warm-up, recovery, recorded scan injection, or reconstructed scan fed back as measured. Both the first reconstruction and every later one use an unobserved LiDAR history plus independently measured motion context. The trained weights can still encode the training scenario; this is a cold-start reconstruction experiment, not evidence of general blind navigation.

The simulator retains nominal raw LiDAR **only for scoring**. It is excluded from the inference features and VFH/controller input; the viewer uses reconstructed clouds. Native pose, velocity and attitude measurements remain available. The existing VFH/PID and goal/time-limit termination are unchanged, with no barrier certificate or second training stage.

The disabled model uses its own copy under `results/saits_dense_disabled_integration/<episode-id>/`, with callbacks that restore the constant-disabled schedule on every toolbar Run. The original intermittent model and `captureLidarDisabledSingleUavRun.m` are unchanged. Use the capture script, rather than just the toolbar button, to save telemetry.

Outputs go to **`data/saits_dense_single_episode_disabled/<UUID>/`**, containing the manifest, telemetry MAT, `scans.csv`, and `summary.json`. The audit expects one continuous outage starting at time zero, zero recoveries, zero measured-LiDAR scans, zero LiDAR input masks, and reconstruction at every scan including the first. Inspect `runtimeProof.firstScanReconstructed`, `allScansReconstructed`, and `noMeasuredLidarObserved`. `observedOutageCount=1` counts the already-active initial outage; the random seed is empty because this availability schedule is deterministic. If the drone reaches its goal early, the fault end time records the actual termination time while the planned interval remains in the manifest.

## Verification scope

Focused tests cover chronology, hidden-input isolation, train-only statistics, full output layout, selected-checkpoint artifacts, and export provenance. A one-epoch reduced-width CPU execution check used the real selected episode, with its temporary weights kept outside production run/model discovery. MATLAB reference parity, online outage isolation, and model compilation are checked separately. Those are implementation checks, not trained-model performance or a completed simulation. Run full GPU training before using the MATLAB entry point for your experiment.
