# Map-temporal CNN: single-UAV MATLAB simulation

Run from the MATLAB Command Window:

```matlab
addpath("A:/Projects/ARES/src");

MapTemporalOptions = ares.mapTemporal.defaultOptions();
MapTemporalOptions.ModelPath = "A:/Projects/ARES/results/sparse_reconstruction_map_temporal/Map Temporal Sparsity Invariant CNN_20261120T002633_920339Z/best.pt";
MapTemporalOptions.StartState = [0 0 -7 0 0 0];
MapTemporalOptions.GoalState  = [30 35 -7 0 0 0];
MapTemporalOptions.LidarMode = "intermittent"; % "blackout", "healthy", or "custom"
MapTemporalOptions.NumberOfOutages = 5;
MapTemporalOptions.ShowSimulation = true;

run("A:/Projects/ARES/scripts/runMapTemporalSingleUav.m");
```

States are `[North East Down vNorth vEast vDown]` in metres and m/s. Negative Down is altitude above ground. Goal velocity must be zero: the mission terminates when both position and velocity meet the configured tolerances. Start and goal positions must be free of obstacles and inside North/East `[-5,40]`, altitude `[1,10]` m. Default required endpoint surface clearance is 1 m. These are input checks; the controller's flight is evaluated separately.

If start and goal positions coincide, the follower uses a 1 cm virtual preceding waypoint to define its segment direction. The actual initial state, learned-model goal and terminal-state test use the requested position unchanged.

`ModelPath` accepts a downloaded `.pt` checkpoint (`best.pt` or `last.pt`), a run folder containing `best.pt`, or an exported native `model.mat`. Leaving it empty selects the most recently modified `best.pt` below `results/sparse_reconstruction_map_temporal`; paste an explicit path for reproducible comparisons. Selecting `last.pt` evaluates those weights without resuming training.

For a `.pt` file, preparation uses `PythonExecutable` (default: this project's `.venv`) with PyTorch, NumPy and SciPy to export the checkpoint. Change this option if your Python environment is elsewhere. Export runs no training cells. MATLAB then checks the exported network against five PyTorch reference sequences before compiling the simulation. Python and a GPU are not used during the flight. MATLAB, Simulink and the toolboxes required by the existing single-UAV model are required. The native implementation uses ordinary single-precision MATLAB array operations and does not require Deep Learning Toolbox.

Exports are cached under `results/map_temporal_exports/<signature>/model.mat`. You can paste that native path in later runs to bypass Python export. Keep the adjacent export manifest for checkpoint provenance; the MAT file includes the weights, obstacle map and parity fixtures. Export signatures include checkpoint contents and the notebook's model definitions, so an incompatible checkpoint fails explicitly.

## Outage settings

`"blackout"` disables measured LiDAR from time zero, including the first scan. No healthy warm-up is required. `"healthy"` always selects measured LiDAR. In `"intermittent"` mode, `NumberOfOutages` controls the count, with reproducible random durations/placement:

```matlab
MapTemporalOptions.Seed = 513;
MapTemporalOptions.StopTimeS = 60;
MapTemporalOptions.OutageDurationS = [1 2];
MapTemporalOptions.OutageWindowEndS = 20;
MapTemporalOptions.InitialHealthyTimeS = 2;
MapTemporalOptions.MinimumRecoveryTimeS = 1;
MapTemporalOptions.FinalHealthyTimeS = 1;
```

Schedules use a 0.1 s scan grid. Increase `OutageWindowEndS`/`StopTimeS` if the requested outages do not fit. A mission that reaches its terminal state early can end before all scheduled outages; the summary distinguishes planned and observed counts.

For exact failure timing:

```matlab
MapTemporalOptions.LidarMode = "custom";
MapTemporalOptions.OutageIntervalsS = [2 4; 7 10; 12 25]; % [start,end) seconds
```

For a different initial orientation or motion, use `StartEulerZYXRad = [yaw pitch roll]`, `StartBodyRatesRadps = [p q r]`, and the velocity portion of `StartState`. Initial speed is limited to 2.5 m/s and roll/pitch to 25 degrees, matching the shared mission validation.

## Executed simulation

The runner creates a separate `.slx` copy under `results/map_temporal_simulation/<run-id>/`. The original `ares_single_uav_top.slx` and map-correction simulation are preserved. The new copy retains the original UAV plant, waypoint follower, VFH obstacle avoidance and PID controller, including VFH's original three-second obstacle memory. During outages the selected downloaded CNN/GRU reconstructs an organized 15-by-716 range image, converts it to the sensor-frame cloud and feeds that cloud to VFH. Healthy scans pass through unchanged; VFH can still remember obstacles from preceding scans of either type. This is a learned reconstruction experiment; no barrier controller or global route planner is added.

For the zero-velocity terminal state, the waypoint follower enables hover at its final waypoint. Once within `min(0.75, MinimumSurfaceClearanceM/2)` metres of the goal (0.5 m by default), a terminal selector latches the exact goal position and current yaw as the original PID's reference. This avoids continually commanding a moving VFH lookahead point while trying to settle. The selector does not change reconstruction and is logged as `map_temporal_goal_hold`; the canonical `desired_position_ned_m` log records the reference actually passed to the PID. A stop is requested only after the measured position and velocity satisfy both terminal tolerances.

Inference receives the checkpoint's obstacle boxes, the current UAV pose/velocity, the goal and the last 16 measured-input summaries/state records (10 Hz). Missing observations are zeroed with false availability/hit masks. Past predicted scans are never treated as measured history. Even during a blackout lasting beyond 1.6 s, current pose/map/goal remain available while old healthy measurements leave the window. This assumes localization remains available during LiDAR failure.

Every simulation Run starts with empty history. In `ShowSimulation = true`, the scenario viewer shows the selected measured/reconstructed cloud. A separate nominal-scan branch terminates in logging for offline reconstruction-error scoring; it never feeds the network, history or controller. Nonfinite states/predictions stop the simulation instead of silently switching to another algorithm. The simulated obstacle boxes must match the checkpoint's map.

To prepare/open the model and then use the Simulink Run button:

```matlab
run("A:/Projects/ARES/scripts/prepareMapTemporalSingleUav.m");
```

This initializes the scene and saves the compiled experiment. Each toolbar Run reinitializes the requested state, fault schedule and history. To also save the analyzed CSV/JSON/figure outputs after preparation, run `mapTemporalResult = ares.mapTemporal.runMission(mapTemporalPrepared);`. If inputs or source files change, prepare again.

## Results and interpretation

The runner leaves `mapTemporalPrepared` and `mapTemporalResult` in the workspace. Each run folder contains:

- `mission.mat`, `prepared.json`, and the generated `.slx`: requested inputs, schedule, native-model validation, hashes and wiring metadata.
- `scans.csv`: state/goal, fault availability, reconstruction use, history length, blackout age, inference time and terminal errors at 10 Hz.
- `simulation.mat`: full simulation output, saved before analysis so the flight is retained if reporting fails.
- `telemetry.mat`: native-rate actual position/velocity, selected/nominal range images and sanitized current input masks for analysis.
- `summary.json`: actual goal-at-rest outcome, collision/clearance and altitude results, planned/observed outage counts, range RMSE/MAE during outages, and input-isolation checks.
- `mission.png`: actual trajectory with obstacle boxes, outage timing and terminal-state errors.

The network is approximate, and VFH is a local planner. An arbitrary free start/goal can still lead to a stalled, unsafe or unsuccessful flight. Complete-blackout reconstruction uses the static known map and localization; it cannot establish unobserved moving obstacles or map changes. Neither a successful parity check nor a successful flight certifies all start/goal/failure configurations. Evaluate `Metrics.missionSuccess`, clearance, terminal errors and `Proof`, rather than judging only the rendered path. Native CPU inference may run slower than the 0.1 s simulated scan period; simulation time is independent of wall time.

In `scans.csv`, `prior_min_range_m = -1` on healthy ticks means prior inference was skipped; healthy scans use their measured ranges. Inference timings are zero on those ticks.

Optional checks (after creating a native export):

```matlab
results = runtests("A:/Projects/ARES/tests/data/TestMapTemporal.m");
assertSuccess(results);
```
