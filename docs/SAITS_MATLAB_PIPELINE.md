# Online SAITS in the intermittent LiDAR simulation

Run `scripts/captureSaitsIntermittentLidarSingleUav.m` to apply the trained 50-scan SAITS model during the existing intermittent-availability experiment. The script runs a temporary copy of the single-UAV Simulink model and saves one new UUID run under `data/saits_intermittent/`. It restores the default LiDAR configuration afterward; completed interactive runs keep the diagram and animation open, while unattended runs close the temporary model.

This is a closed-loop integration experiment: reconstructed observations can change the avoidance reference and the executed flight commands. It is not a safety guarantee. The trained model reconstructs seven forward-sector distances and seven return fractions, not a dense 10,740-beam scan. Any effect on the controller must be assessed using the saved routing and reference-difference telemetry, together with the flight outcome.

## Prepare the trained artifact

The default artifact is:

```text
data/models/saits_endpoint/model.mat
```

Use the exported MATLAB artifact from the selected trained checkpoint. It must contain the weights and the matching training-fitted preprocessing for the project's SAITS endpoint architecture: 50 rows, 31 input features, and 14 physical output features. A training checkpoint such as `best.pt` is not interchangeable with `model.mat`; changing the extension does not export the model. Keep the selected checkpoint, export metadata, and artifact together for traceability.

The first seven physical outputs are sector distances in metres; the final seven are valid-return fractions. The inference path must use the same feature order and preprocessing as training. It must not refit normalization on the simulated flight. The raw prediction log retains unbounded model outputs; the separate controller-sector log records the bounded observations actually supplied to the adapter.

### Export the selected checkpoint in Colab

Keep the working Colab PyTorch installation. The exporter additionally needs SciPy (`%pip install scipy` only if it is missing). It uses the ARES `saits_endpoint_model.py` implementation and rejects a checkpoint whose model source, feature schema, selected epoch, or available preprocessing sidecar disagrees with its saved provenance.

From the existing checkout, export the selected `best.pt` into a **new** directory:

```python
from pathlib import Path
import subprocess
import sys

repo_root = Path("/content/ares")  # Your existing checkout.
checkpoint = Path(result.output_dir) / "best.pt"  # Or your saved training folder.
export_dir = Path("/content/saits_matlab_export")  # Choose a new name if it exists.

subprocess.run([
    sys.executable,
    str(repo_root / "scripts" / "export_saits_matlab.py"),
    "--checkpoint", str(checkpoint),
    "--output-dir", str(export_dir),
], check=True)
```

The equivalent terminal command is:

```text
python scripts/export_saits_matlab.py --checkpoint /path/to/best.pt --output-dir /path/to/new_export
```

The exporter reads the checkpoint without changing it and refuses an existing export destination. It preserves FP32 weights and the saved preprocessing, packages five reference cases, and writes `model.mat`, `export_manifest.json`, and `preprocessing.json`. The reference predictions use CPU FP32 evaluation; they do not attempt to reproduce every rounding detail of T4 FP16 autocast. Export success records MAT read-back validation but does not claim MATLAB parity before the MATLAB checks below run.

To transfer the complete package from Colab:

```python
import shutil
from google.colab import files

archive = shutil.make_archive("/content/saits_matlab_export", "zip", root_dir=export_dir)
files.download(archive)
```

Extract those three files into `data/models/saits_endpoint/` in the MATLAB checkout, retaining the sidecars. If that folder already contains a model you want to preserve, use a new named model folder and select its `model.mat` through the override below.

To use a different exported artifact, set the base-workspace override before running:

```matlab
SaitsArtifactPathOverride = "C:\path\to\exported\model.mat";
```

The override is a file-selection preference and remains in the workspace. Clear it to return to the default artifact:

```matlab
clear SaitsArtifactPathOverride
```

### Verify MATLAB inference before the flight

With the ARES project open, validate the exported package against its saved PyTorch reference cases:

```matlab
proj = currentProject;
artifactPath = fullfile(proj.RootFolder,"data","models","saits_endpoint","model.mat");
report = validateSaitsMatlab(artifactPath, ...
    BenchmarkCount=100,AbsoluteTolerance=1e-4);
assert(report.allParityChecksPassed);
disp(report);
```

Use the override path here as well if the model is in another folder. This verifies physical and normalized predictions, intermediate stages, attention, and explicit-mask versus NaN input handling before benchmarking warmed native inference. Native inference uses ordinary MATLAB numeric operations and does not require Python, an ONNX importer, or Deep Learning Toolbox in MATLAB.

The checked local artifact passed five cases with maximum physical prediction difference approximately `1.91e-6`; its measured native mean was approximately `1.02 ms` per window on the tested host. These are an artifact-specific parity result and a local wall-time measurement, not a latency bound or a T4 benchmark. Re-run validation for your exported checkpoint and retain its report.

An optional MEX build can be validated and benchmarked through the same entry point once a supported C++ compiler is configured:

```matlab
mexDirectory = fullfile(fileparts(artifactPath),"mex");
report = validateSaitsMatlab(artifactPath,BuildMex=true, ...
    MexDirectory=mexDirectory,BenchmarkCount=100,AbsoluteTolerance=1e-4);
assert(report.allParityChecksPassed && report.mexBuilt);
```

The MEX route uses the project C++ implementation with the [normal MATLAB compiler interface](https://www.mathworks.com/help/matlab/ref/mex.html) and [MATLAB BLAS](https://www.mathworks.com/help/matlab/matlab_external/calling-lapack-and-blas-functions-from-mex-files.html); it does not require MATLAB Coder or Deep Learning Toolbox. The controller looks for the validated binary and its parity/hash manifest in the artifact's `mex/` subdirectory. It verifies model, predictor source, C++ source, and binary hashes before selecting that backend. A compiled file alone is insufficient. Compiled binaries are local build outputs; rebuild them after cloning on another host. Native MATLAB remains the reference inference path if a validated binary is unavailable.

The local C++ MEX build passed the same five reference cases with maximum physical difference `1.91e-6` and a warmed mean of approximately `0.61 ms` per window. The controller logs `saits_backend_mex`, and the run summary counts actual compiled inference calls separately from total reconstruction use.

## Run from the ARES project

With `ares.prj` open in the MATLAB desktop, run:

```matlab
proj = currentProject;
SaitsShowSimulationOverride = true;
run(fullfile(proj.RootFolder,"scripts","captureSaitsIntermittentLidarSingleUav.m"));
```

The script exposes `saitsManifest`, `saitsTelemetry`, `saitsManifestPath`, `saitsTelemetryPath`, and `saitsSummary` in the calling workspace. These refer to the newly captured run. The persistent run ID and output folder are also printed.

Desktop sessions show the Simulink diagram and UAV Scenario Scope animation by default. The viewer receives the same selected point cloud as obstacle avoidance: measured clouds when healthy, absent clouds during history-warmup fallback, and the coarse sector proxy during reconstruction. The [UAV Scenario Scope](https://www.mathworks.com/help/uav/ref/uavscenarioscope.html) opens its animation automatically. [Simulation pacing](https://www.mathworks.com/help/simulink/slref/simulationpacingoptions.html) is enabled at 1x for viewing; it can slow a fast run but cannot speed up a slow one.

The diagram and animation remain open after a completed interactive capture. Running the capture script again replaces its unchanged generated model; edits made in the diagram are preserved by refusing replacement. The model's Run button reinitializes the same intermittent faults for visualization, while the capture script also saves the manifest and telemetry. For unattended runs use `SaitsShowSimulationOverride=false`; non-desktop batch sessions use that mode by default and close the temporary model afterward.

The saved model `models/ares_single_uav_top.slx` and the original script `captureIntermittentLidarDisabledSingleUav.m` keep their existing behavior. The integration loads `results/saits_integration/ares_single_uav_saits.slx` and instruments it in memory without saving its temporary changes. A running or user-edited copy is never overwritten. An existing copy file is reused only when its bytes still match the saved baseline; otherwise the capture stops so that existing edits can be preserved.

## The original faults are preserved

`ares.inference.createIntermittentSchedule` uses the original private `mt19937ar` stream with seed 513. The configuration remains three outages, 3 seconds of initial healthy time, durations of 1.5–3 seconds, at least 2 seconds of recovery between outages, and a 4-second final healthy reserve inside the 21-second fault window. The simulation horizon remains the default 60 seconds; the model may terminate earlier when it reaches the goal.

The resulting continuous-time outage intervals are:

```text
[ 3.544348603028016,  5.105822685654895)
[10.356031289257292, 12.010370739258892)
[14.974288494995000, 16.549329689818645)
```

The schedule uses zero-order hold. Native controller-rate `lidar_enabled` retains those exact commanded transitions. SAITS inference runs at the 10 Hz sensor rate, so its measured-availability signal changes at the first scan at or after an edge.

The existing fault experiment models LiDAR as unavailable to the autonomy stack. The simulated sensor continues producing nominal scans for evaluation. During an outage, those retained nominal scans are oracle data for scoring and must never supply hidden reconstruction inputs.

## Observation selection and fallback

At a healthy scan, the controller uses measured LiDAR and the history records healthy sector features. At an outage scan with enough history and a valid prediction, the controller can use the reconstructed sector observation through the VFH adapter. If history is incomplete or prediction validation fails, the original waypoint-following fallback remains available.

The first outage begins before 50 scans have accumulated. A full 50-scan window first exists at 4.9 seconds, so early outage scans must record `history-not-ready` rather than inventing history or moving the original outage. Future scans after the current prediction time are never part of a model input window.

The native LiDAR availability and the enablement of obstacle guidance are separate signals. During reconstruction use, `lidar_enabled` remains false while guidance can be enabled through the reconstructed observation. Logged `saits_guidance_enabled` represents the controller-rate OR of the original availability schedule and the held reconstruction-used signal.

The compact observation spans body-forward bearings from -60 to +60 degrees, retaining all 15 elevation rows when computing each sector's minimum valid range and return fraction. Sensor azimuths use `linspace(-179,179,716)` and the recorded 180-degree roll mount reverses bearing. A healthy no-return sector is an observed distance of 7 m and fraction 0; it is distinct from an unavailable outage input.

The adapter bounds reconstructed distances to `[0.201,7]` m and fractions to `[0,1]`. A return fraction indicates return presence, not model confidence. Each sector summary is a coarse approximation of the geometry in that angular region. Outside the forward ±60-degree field, the reconstruction provides no information; the adapter's fixed 6.95 m guard cloud makes that unknown side/rear area explicit to VFH instead of treating absent reconstructed points there as free space. This guard policy is part of the experiment and must be reported when interpreting changed trajectories.

The experiment sets VFH's point-cloud memory (`MaxAge`) to zero so that synthetic proxy points are removed on the next healthy scan. Its original value is three and is recorded alongside the experiment value in the manifest. SAITS retains its separate 50-scan history. This parameter change applies only to the temporary experiment model.

## Saved run and telemetry

Each successful simulation capture produces:

```text
data/saits_intermittent/<run-UUID>/
    manifest.json
    saits-intermittent-lidar-single-uav_telemetry.mat
    scans.csv
    summary.json
```

The manifest records the source model hash, artifact path and SHA-256, fault seed and intervals, coordinate conventions, model dimensions, adapter description, truthful outcome metrics, and runtime proof. Manifest and telemetry share the generated run UUID and the existing data-contract schema. Files are created once; previous run artifacts are not overwritten.

The MAT file retains the ordinary native flight signals and numeric additional logs. Original nominal XYZ clouds are stored separately under `telemetry.oracleLidar`, explicitly marked as scoring-only during outages. Scan-aligned nominal ranges, return masks, degraded observations, availability/missingness masks, fault transitions, and outage age remain available for EDA. Finite placeholders in missing inputs are accompanied by an observation mask; they are not measurements.

The online proof signals include:

- `saits_input_features` and `saits_input_observed_mask`: the actual 31 input features and their availability at each scan.
- `saits_history_ready`, `saits_prediction_valid`, `saits_reconstruction_used`, and `saits_fallback_reason`: whether model output was usable and selected.
- `saits_raw_prediction` and `saits_controller_sectors`: the physical predictions before controller bounds, and the observations used by the adapter.
- `saits_measured_available`, `saits_guidance_enabled`, and `saits_inference_seconds`: sensor availability, guidance selection, and inference wall time.
- `saits_selected_reference_ned_m`, `saits_waypoint_reference_ned_m`, and `saits_reference_delta_ned_m`: the actual selected position reference, same-state waypoint fallback reference, and their difference.
- `command_roll_rad`, `command_pitch_rad`, `command_yaw_rate_radps`, `command_thrust_n`, and the native position/velocity: the executed control path and resulting trajectory.

Fallback codes are 0 for healthy measurement, 1 for incomplete history, 2 for an invalid prediction, and 3 for model use. Missing or malformed proof signals cause an explicit instrumentation error. Scan CSV columns also include the independent oracle sector targets for scoring; these are labelled `OracleScoringTarget_*` and do not belong in the inference inputs.

### Inspect the saved run in MATLAB or Python

The MAT file is the complete telemetry artifact; the CSV is a compact, scan-aligned EDA view. Inspect the files that the capture just returned:

```matlab
savedRun = load(saitsTelemetryPath,"telemetry");
runFolder = fileparts(saitsTelemetryPath);
scanTable = readtable(fullfile(runFolder,"scans.csv"));
runSummary = jsondecode(fileread(fullfile(runFolder,"summary.json")));
assert(string(savedRun.telemetry.runId)==string(runSummary.runId));
disp(runSummary.runtimeProof);
```

After making that UUID folder available in the Python environment, read the compact table directly:

```python
import json
from pathlib import Path
import pandas as pd

run_dir = Path("/path/to/data/saits_intermittent/<run-UUID>")
scans = pd.read_csv(run_dir / "scans.csv", dtype={"RunId": str})
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8-sig"))
assert scans["RunId"].eq(summary["runId"]).all()
print(summary["runtimeProof"])
print(scans[["TimeS", "LidarMeasuredAvailable", "SaitsReconstructionUsed",
             "ControllerReferenceDifferenceM", "ControllerInfluencedDuringOutage"]].head())
```

This capture does not automatically create an `episode.npz` or insert the run into the existing 80-run healthy-batch index. That healthy-episode loader and its always-on assumptions are a different contract. Use this run's MAT/CSV/JSON artifacts directly; keep oracle scoring fields separate from actual input features when preparing any later model dataset. MATLAB saves the full telemetry as v7.3 MAT, so generic Python `scipy.io.loadmat` is not the reader for that telemetry file. The exported inference `model.mat`, by contrast, is the ordinary MAT-v5 package created by the exporter.

## Interpret the result

`summary.json` records configured versus observed outage counts, reconstruction-used scans, history/invalid-prediction fallbacks, inference timing, and the count of reconstructed outage scans where the selected reference differs from the waypoint fallback. The script verifies that reconstruction use requires a current outage, ready history, and valid prediction, and that the saved reference delta matches the difference between the two reference branches.

`controllerInfluencedDuringOutage=true` requires at least one reconstructed outage sample with a reference difference greater than `1e-5` m. A model call or routing flag alone does not establish a changed command. If this value is false, report that no nonzero reference influence was demonstrated in this run, even if the model was called and its outputs were routed correctly.

The flight metrics use the actual NED trajectory and the physical obstacle geometry: initializer mesh coordinates are ENU and are swapped before NED distance calculations. Clearance is signed 3D box distance minus vehicle radius, and collision/clearance checks use every logged native position sample. This is sampled collision detection, not a proof of collision freedom between samples. A run that reaches the goal can still fail safety or contain a collision; its telemetry is retained with the corresponding outcome.

The script requires all three outages and recoveries to appear in the captured sensor timeline and at least one usable reconstruction during an outage. If those instrumentation checks fail after simulation, it saves the available run artifacts and then raises `ARES:SaitsRuntimeProofFailed`. An actual simulator or inference exception still propagates, while cleanup restores nominal configuration and closes the temporary model.

One trained single-episode checkpoint—especially one selected on constant no-return validation targets—does not establish generalization or improved navigation safety. Compare the new run with the original fault baseline under the same schedule, inspect the reconstructed observations and reference differences, and report the actual mission and safety outcomes.

## Verified local compiled run

The compiled run `e6d2acb5-fcc3-4ab2-817a-fbd2ee8d4096` is saved under `data/saits_intermittent/`. Its 278 scans include all three outages and recoveries: 49 unavailable scans, 13 history-warmup fallbacks, and 36 reconstruction uses. All 36 used the compiled backend and had a nonzero difference from the same-state waypoint reference. Inference averaged 1.02 ms during this run, with a 2.38 ms maximum; this measures inference rather than the whole control step and is not a real-time bound.

The flight reached the goal with 0.270 m final error at 27.7 s, but sampled trajectory points intersected an obstacle. The run correctly records `collisionOccurred=true`, `safetyChecksPassed=false`, and `missionSuccess=false`. The pipeline and source-routing checks passed; this checkpoint has not demonstrated safe recovery from LiDAR failures. The earlier native-MATLAB comparison run remains separately saved under UUID `5658a477-4cfd-4858-9e0e-72554429d372`.

Verification also passed 17 MATLAB tests (inference, source isolation, sensor geometry, history/reset behavior, and immutable telemetry destinations) and 36 Python tests. Five exported reference cases agree with both native MATLAB and C++ MEX to within approximately 1.91e-6 in physical predictions. The trained checkpoint, baseline SLX, and original intermittent-failure script were preserved.
