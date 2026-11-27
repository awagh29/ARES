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

The runner creates a separate `.slx` copy under `results/map_temporal_simulation/<run-id>/`. The original `ares_single_uav_top.slx` and map-correction simulation are preserved. The default experiment uses the trajectory-sampling local planner described below, with the original plant and unchanged velocity/attitude PID gains. During outages the selected downloaded CNN/GRU reconstructs the organized 15-by-716 sensor-frame cloud. Healthy scans pass through unchanged. Reconstruction weights remain frozen. No CBF or global route planner is added.

Set `EnableLocalPlanner=false` to use the earlier VFH/braking-governor experiment. That branch retains the original VFH point memory (three seconds) and position PID. Set `EnableReferenceGovernor=false` to bypass both the local planner and governor and reproduce the legacy VFH/PID experiment.

## Trajectory-sampling local planner

The planner executes at 10 Hz before the braking governor. It samples 24 horizontal headings, the direct target direction, the previous command direction, two speeds and a stop command. Heading samples include an altitude component toward the target; it does not search arbitrary vertical detours. Each candidate rolls out from the **actual velocity**, allows sensing/tracking latency, limits acceleration, and includes a braking tail. It rejects trajectories that intersect inflated selected-scan surfaces, leave the range-limited horizon or exceed mission bounds. This is an approximate rollout model, not an exact prediction of the PID/plant.

Selected points are voxelized for bounded computation. Each retained cell centre is inflated by the cell half-diagonal, so discarded points are not silently treated as free space. Required surface clearance remains 1 m by default; the extra planning buffer is a preference in the trajectory score. If already inside the additional engineering buffer, a candidate may avoid deepening that overlap while preserving the required clearance threshold. The downstream braking governor still checks requested motion and actual momentum.

The score balances route progress, extra clearance, command continuity and revisiting old locations. A route-cost guide accumulates only selected scans (including reconstructed scans during outages) on a 0.5 m horizontal grid at the requested goal altitude. This memory is separate from the reconstructor's measured-only temporal buffer. It rebuilds an eight-neighbour shortest-path cost field once per second. Seen obstacles are inflated; diagonal corner cutting is forbidden. Unknown cells have higher cost and can suggest future exploration, but a route suggestion never bypasses the current-scan rollout and braking checks. New free-space rays clear stale observations, and returns beyond 5.5 m are not retained as persistent obstacles, avoiding near-maximum-range reconstruction rings. The guide starts empty, receives no supplied obstacle boxes, and resets every flight. It is an observed-scan route memory, not a ground-truth map planner.

Little goal progress **and** little displacement over `PlannerStallTimeS` trigger stationary recovery. A longer window (four times that duration) detects persistent lack of goal progress while moving. Normal sideways bypass motion therefore does not immediately trigger recovery. The planner selects a reachable temporary escape waypoint, penalizes old loops and retains it for up to `PlannerRecoveryTimeS`. Deliberate escape motion may increase goal distance; after recovery, the normal progress window starts fresh. No feasible trajectory produces a zero-velocity request and is counted explicitly in the logs.

```matlab
MapTemporalOptions.EnableReferenceGovernor = true;
MapTemporalOptions.EnableLocalPlanner = true; % default
MapTemporalOptions.PlannerHorizonS = 2.0;
MapTemporalOptions.PlannerRolloutStepS = 0.2;
MapTemporalOptions.PlannerVoxelM = 0.2;
MapTemporalOptions.PlannerStallTimeS = 2.5;
MapTemporalOptions.PlannerProgressM = 0.35;
MapTemporalOptions.PlannerRecoveryTimeS = 6;
```

With local planning enabled, VFH is bypassed and commented out in the experiment copy. The original position PID remains diagnostic; the planner replaces its velocity request before the unchanged velocity/attitude PID loops. This prevents unused VFH obstacle memory from accumulating reconstructed surfaces and avoids letting a blocked VFH direction veto the escape direction. The executed route is recorded in `prepared.json`. The planner receives selected scan geometry, current motion, goal and mission bounds, and maintains its own observed-cell/visited-pose history. It has no input from the hidden nominal scoring scan or direct obstacle boxes. The map remains an input to the frozen reconstructor, as before. The horizontal route guide does not search vertical detours; full 3-D scan points are still used for rollout collision checks.

`planner.csv` records recovery state/count, stalled flag, feasible candidate count, preferred speed, predicted surface clearance, rolling goal progress, temporary target, infeasibility and compute time. Predicted clearance is distinct from actual native-rate clearance in the mission summary. `governor.csv` records the planner's preferred velocity and the governed command in this mode.

To compare the previously stalled short intermittent, complete-blackout and latest screenshot missions against matching archived governor-only flights:

```matlab
addpath("A:/Projects/ARES/scripts");
report = validateMapTemporalPlanner("A:/Projects/ARES", MapTemporalOptions.ModelPath);
```

The script checks selected weight and original model hashes, endpoints, outage schedules and input isolation. If no matching governor-only record exists, it runs that control first. Results and native-rate clearance/error comparison plots are written under `results/map_temporal_planner/<id>/`. Individual cases can be selected with a third argument: `"shortIntermittent"`, `"completeBlackout"` or `"screenshotIntermittent"`. Unsuccessful flights remain recorded as outcomes.

## VFH and braking-governor comparison mode

By default, the experiment adds a terminal reference selector and a braking-aware velocity governor. Within `GoalApproachRadiusM` (4 m), the selector uses the exact goal position and current yaw when the selected scan has a clear swept corridor to the goal. It returns to VFH guidance if the corridor becomes blocked. The selector is logged as `map_temporal_goal_hold`; the canonical `desired_position_ned_m` log records the reference actually passed to the position PID. A stop is requested only after measured position and velocity both satisfy their terminal tolerances.

The governor executes at 10 Hz between the original position PID's velocity output and the original velocity PID. It caps the velocity vector norm, tapers speed with goal distance in the approach phase, and uses `v*latency + v^2/(2*braking_deceleration)` to limit speed along a swept corridor. It checks actual momentum as well as requested motion, and commands zero velocity when the current speed exceeds the stopping envelope. Nominal command changes have an acceleration limit; urgent reductions bypass that smoothing. The governor rechecks the interpolated command direction. These are approximate point-cloud checks, not a safety certificate or a guarantee of actual plant speed/clearance. The original outer-loop integral gains are zero; no PID gains are retuned.

```matlab
MapTemporalOptions.EnableReferenceGovernor = true; % default; false restores legacy guidance
MapTemporalOptions.CruiseSpeedMps = 1.5;
MapTemporalOptions.BrakingDecelerationMps2 = 0.8;
MapTemporalOptions.GovernorLatencyS = 0.35;
MapTemporalOptions.GovernorMarginM = 0.35;
MapTemporalOptions.GovernorPlanningBufferM = 0.6;
MapTemporalOptions.GoalApproachRadiusM = 4;
MapTemporalOptions.GoalApproachGain = 0.7;
MapTemporalOptions.GovernorAccelerationMps2 = 0.8;
```

The clearance radius is vehicle radius plus required surface clearance plus `GovernorMarginM`. VFH's experiment-only safety distance is required surface clearance plus `GovernorMarginM` plus `GovernorPlanningBufferM`, so it steers before the braking envelope becomes tight. Geometry comes only from the selected measured/reconstructed cloud, with the recorded mounting transform applied once. The nominal scoring scan and direct map geometry are not inputs to the governor. `GovernorLatencyS` is a simulated sensing/tracking allowance, not measured wall-clock inference time. This offline simulation does not insert real-time inference delay. With the governor disabled, the legacy VFH safety distance, 0.5 m goal latch (at default clearance) and unmodified position-to-velocity PID connection are used.

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
- `governor.csv` (when enabled): original and governed velocity requests, speed cap, requested/momentum corridor distance, braking flag, approach flag and goal distance at 10 Hz.
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
results = runtests("A:/Projects/ARES/tests/data/TestMapTemporalGovernor.m");
assertSuccess(results);
```

For paired original/governed real-plant flights with identical native weights, start/goal and fault schedules:

```matlab
addpath("A:/Projects/ARES/scripts");
report = validateMapTemporalGovernor("A:/Projects/ARES", MapTemporalOptions.ModelPath);
```

This runs healthy, intermittent and cold-start blackout cases in both arms and saves summaries and comparison figures under `results/map_temporal_governor/<id>/`. It records unsuccessful missions as outcomes rather than discarding them. A third argument, for example `"intermittent"`, restricts the comparison to one mode. The report verifies that native weights and the baseline Simulink model were unchanged.

## Measured route-guided planner comparison (2026-11-26)

All three previously stalled missions completed within the same 60 s limit using the frozen epoch-42 native weights. The 1 m required clearance, 1.5 m/s cruise cap, PID gains, start/goal states and outage intervals were unchanged from the governor-only controls.

- Short intermittent, `[15,10,-7]` to `[25,25,-7]`: governor-only timed out with 15.959 m goal error and 1.330 m minimum clearance. The planner completed at 24.0 s with 1.440 m minimum clearance, encountering all five outages.
- Latest screenshot intermittent, `[0,0,-7]` to `[25,25,-7]`: governor-only timed out with 11.290 m goal error and 1.362 m minimum clearance. The planner completed at 31.7 s with 1.474 m minimum clearance, encountering all five outages.
- Complete blackout, `[0,0,-7]` to `[30,35,-7]`: governor-only timed out with 27.301 m goal error and 1.361 m minimum clearance. The planner completed at 34.7 s with 1.642 m minimum clearance, using reconstruction for every scan.
- Healthy regression, `[0,0,-7]` to `[30,35,-7]`: governor-only completed at 39.2 s with 1.417 m minimum clearance. The planner completed at 35.1 s with 1.517 m minimum clearance.

Seventeen MATLAB checks passed, covering parity, input isolation, history/reset behavior, braking, blocked-direction selection, recovery persistence/exit and bypass-motion stall detection. The comparison verified matching selected weight hashes, baseline model binary, endpoints, vehicle parameters, obstacle geometry, outage schedules and functional controller sources. The combined report and native-rate clearance/error plots are under `results/map_temporal_planner/validated_437438c7-e8c5-4baa-b7db-53540b8ab72b/`. These are deterministic MATLAB simulation cases, not validation of Gazebo, wind or swarm behavior. No CBF or reconstruction retraining was used.

Earlier sampling-only trials could still loop or chain recoveries. The final comparison uses the corrected latency/braking rollout, fresh progress windows after recovery, and selected-scan route memory. The route guide avoided prolonged stalls in these final flights; explicit recovery behavior is also covered by stationary/recovery-transition unit tests. General local-planning success or a safety certificate is not claimed.

## Measured governor-only comparison (2026-11-26, before local planning)

The selected epoch-42 native weights were frozen (SHA-256 `55d9c804ff2ef84adb65c3d23e337cdfc278ae8bd8cb138b529cb26b9c04ef0b`). Start/goal states, 60 s time limits and outage intervals matched within each pair. The original model binary and PID gains were unchanged. Success requires both the 1 m surface clearance and the existing 0.35 m position / 0.25 m/s velocity terminal tolerances.

- Healthy, `[0,0,-7]` to `[30,35,-7]`: minimum clearance improved from 0.577 m to 1.417 m. The governed flight met the terminal tolerances at 39.2 s. The original flight reached the goal but failed clearance.
- Five intermittent outages, same long mission: minimum clearance improved from 0.007 m to 1.352 m. Final position error improved from 1.704 m to 0.261 m, and final speed from 0.736 m/s to 0.232 m/s. The governed flight succeeded at 57.1 s and encountered all five outages.
- Cold-start continuous blackout, same long mission: signed clearance improved from -0.134 m (collision) to 1.361 m. The governed flight stopped near an obstacle and timed out with 27.301 m goal error; it did not succeed.
- Five intermittent outages, `[15,10,-7]` to `[25,25,-7]` (the attached error-plot scenario): clearance improved from 0.730 m to 1.330 m. The original reached the terminal tolerances but failed clearance. The governed flight timed out with 15.959 m goal error; it did not succeed.

Thus the governor alone met clearance in four tested missions and full mission criteria in two. It can stall when the selected VFH/PID request remains inside the braking envelope. Increasing the planning buffer helped the long intermittent case but did not eliminate all local stalls. These findings motivated the trajectory-sampling planner. They do not establish reliable navigation for arbitrary starts, blackouts or Gazebo integration. The comparison included no CBF, wind or swarm changes, and no reconstruction retraining. Ten MATLAB unit/parity/isolation checks passed.
