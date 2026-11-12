# Run the locally trained augmented model in MATLAB

The selected deployment is `results/augmented_saits_barrier/ares_augmented_saits_barrier_local_seed42_v1/deployment` (epoch 25, checkpoint SHA-256 `23fdd45618b27c008bfb8bb003d6290cdbc08d42071ba544841c44e21810c0bf`). It is separate from the old single-episode SAITS/MEX artifact.

## Run the experiment

In MATLAB, open the ARES project and run:

```matlab
openProject("A:\Projects\ARES");
SaitsBackendOverride = "augmented-barrier";
SaitsShowSimulationOverride = true;
SaitsContinueAfterBarrierFailureOverride = true;
run("A:\Projects\ARES\scripts\captureSaitsIntermittentLidarSingleUav.m");
```

This validates the bundle, prepares an experiment copy, opens the model/viewer, runs the original seed-513 intermittent schedule, and saves a fresh UUID episode. Diagnostic continuation is now the script default. Set `SaitsContinueAfterBarrierFailureOverride=false` to restore stopping on the first invalid/infeasible filter. If MATLAB already loaded another Python interpreter, restart MATLAB once and repeat. The integration uses the existing `.venv\Scripts\python.exe` in an isolated Python host; it does not install packages.

To validate and compile **without starting a simulation**, run `scripts/prepareSaitsBarrierSingleUav.m` instead. The prepared copy is `results/saits_barrier_integration/ares_single_uav_saits_barrier.slx`; its numerical checks, source hashes, and model hash are recorded in `prepared.json`. Toolbar Run supports visualization, but use the capture script to persist telemetry.

## Outputs

Each capture writes `data/saits_barrier_intermittent/<UUID>/` containing:

- `manifest.json`: model/checkpoint/source identity, exact outage schedule, numerical validation, executed control route, and outcomes.
- `saits-barrier-intermittent-lidar-single-uav_telemetry.mat`: native plant telemetry and all scan-rate inference/control signals. Unavailable raw LiDAR is retained separately as scoring-only oracle data.
- `scans.csv`: time, commands, predictions, measured/reconstructed sectors, predictor values/masks, smooth reference, accelerations, QP status, residuals, gain, and timing.
- `summary.json`: mission/geometry outcomes, outage exposure, filter failure, and actual termination time.

The script also exposes `saitsManifest`, `saitsTelemetry`, `saitsManifestPath`, `saitsTelemetryPath`, and `saitsSummary` in the MATLAB workspace. It retains initial/early failures instead of requiring that all three planned outages finish.

## What executes

The experiment uses the trained temporal and state encoders, measured/reconstructed sector fusion, the same frozen goal controller/reference, learned-gain hard QP, and acceleration-to-guidance allocation. Its four commands replace the old VFH/PID output immediately before `Rate Transition2` and the plant. Canonical command telemetry is rewired to these executed commands. The original VFH block remains only as an unused diagnostic; it does not filter the model's commands.

Prediction and control run once per 0.1-second scan through a persistent CPU FP32 neural / FP64 QP Python runtime. The baseline plant still integrates at 1 ms. A repeated output evaluation at the same scan time returns cached outputs. Model/normalizer/history are not reloaded on each scan; history and healthy anchors reset for every simulation.

During outage, MATLAB discards the raw cloud before invoking Python. Only independent motion state, explicit missingness, and historical available sectors enter the model. No obstacle map is passed online. Displayed outage clouds are coarse sector proxies and unknown-space guards, not dense reconstructed truth.

The current held-out evaluation has 100% filter failure and 0% mission success. Its empirical sector error bounds reach approximately 5.3–6.7 m. These settings are preserved.

In diagnostic continuation mode, an invalid/infeasible filter uses the nominal goal/sector controller driven by measured/reconstructed LiDAR, with the same acceleration and slew limits. The QP is recomputed on every later scan, and feasible QP commands are used again when available. The failure latch stays logged, but no longer triggers **Stop Simulation**. This is an explicit uncertified fallback, not obstacle-constraint relaxation or a claimed feasible QP solution. The logged residual and tracking slack are recomputed for the command actually selected. Simulation can continue to the existing goal/60-second termination conditions; nonfinite runtime errors still fail explicitly.

Check `saitsSummary.goalReachedAtAnyTime` and `saitsSummary.goalReachedAtTermination` to answer whether the drone arrived. `missionSuccess` and `safetyChecksPassed` remain false if any certificate failure occurred, even if it subsequently reached the goal. `uncertifiedNominalFallbackScanCount` and the CSV's `nominal_fallback_used` identify these steps; the packet's `failure_latched` is historical, whereas `filter_feasible` is per scan.

With continuation disabled, the original separate bounded emergency output and **Stop Simulation** behavior remain. Neither mode establishes successful navigation or a safety guarantee merely by completing a capture.

## Use another completed training run

Before the same capture command, set:

```matlab
SaitsDeploymentPathOverride = "A:\Projects\ARES\results\augmented_saits_barrier\YOUR_RUN\deployment";
% Optional if using another compatible environment:
% SaitsPythonExecutableOverride = "A:\Projects\ARES\.venv\Scripts\python.exe";
```

Use the full deployment directory, not only `model.mat` or `best.pt`. Export the selected Stage-2 checkpoint with `scripts/export_augmented_saits_barrier.py` first if its deployment directory is absent. Never overwrite another experiment's bundle. Existing generated model edits are protected; preserve/move an edited experiment copy before regenerating it.

For the original single-episode path explicitly set `SaitsBackendOverride="legacy"`; its original artifact and output directory remain separate.

The Python host uses MathWorks' [out-of-process interface](https://www.mathworks.com/help/matlab/matlab_external/out-of-process-execution-of-python-functionality.html). Measured fixture timing is recorded in `prepared.json`; it is not a hard real-time guarantee or a full-flight timing measurement.
