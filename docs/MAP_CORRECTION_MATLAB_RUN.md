# Map-assisted motion correction during LiDAR outages

This mode uses the existing obstacle scene and a supplied-map assumption. It
needs no neural checkpoint, Python process, or Optimization Toolbox. The
baseline `.slx` is protected; preparation creates a separate, compiled model
named `ares_single_uav_map_correction` in a unique `results/map_correction/`
directory. The existing ML experiments remain available separately.

## Set the mission and run

From MATLAB in the ARES project directory:

```matlab
addpath(fullfile(pwd,"src"));
MapCorrectionOptions = ares.navigation.defaultOptions();

% Six-component NED state: [North East Down vNorth vEast vDown], metres / m/s.
% Down is negative above ground. The goal ends in hover (zero goal velocity).
MapCorrectionOptions.StartState = [0 0 -7 0 0 0];
MapCorrectionOptions.GoalState  = [30 35 -7 0 0 0];

MapCorrectionOptions.LidarMode = "intermittent";
MapCorrectionOptions.NumberOfOutages = 5;
MapCorrectionOptions.ShowSimulation = true;

run("scripts/runMapCorrectionSingleUav.m");
```

To build/open the model without starting a flight, use
`run("scripts/prepareMapCorrectionSingleUav.m")` instead. After preparation,
the model's toolbar **Run** initializes the same mission, obstacle scene and
schedule again. It does not reset to the old fixed ML schedule. Re-run
preparation after changing inputs; already saved missions are immutable input
snapshots. Save any diagram edits before preparing another mission. Each
preparation uses a new directory and preserves the previous model and results.

In desktop mode the scope displays measured scans only when LiDAR is available.
Scans are cleared during disabled/failure intervals, including full blackout,
and reappear on recovery. Map-ray-cast observations remain internal for
correction and diagnostics. Without the MATLAB desktop, set
`ShowSimulation=false` for unattended validation.

## Complete blackout, healthy flight, or exact outage times

```matlab
% No healthy scan, warm-up history, or recovery is required:
MapCorrectionOptions.LidarMode = "blackout";
run("scripts/runMapCorrectionSingleUav.m");

% Healthy comparison with the same map planner/controller:
MapCorrectionOptions.LidarMode = "healthy";
run("scripts/runMapCorrectionSingleUav.m");

% Or supply explicit outage intervals in seconds:
MapCorrectionOptions.LidarMode = "custom";
MapCorrectionOptions.OutageIntervalsS = [0 4; 10 15; 21 26];
run("scripts/runMapCorrectionSingleUav.m");
```

`NumberOfOutages=0` in intermittent mode also produces a healthy flight.
Blackout and healthy modes take precedence over `NumberOfOutages`.
Intervals are sorted/disjoint and use 0.1-second LiDAR scan boundaries.
An outage ending at the simulation stop remains unavailable at that final
sample. Generated schedules use a private seeded random stream and do not
change the global RNG. An impossible count/duration combination fails before
flight rather than silently reducing the count.

Optional schedule inputs are `OutageDurationS=[1 2]`, `InitialHealthyTimeS=2`,
`FinalHealthyTimeS=1`, `MinimumRecoveryTimeS=1`, `OutageWindowEndS=20`,
`Seed=513` and `StopTimeS=60`. The fault window ends at the smaller of
`OutageWindowEndS` and `StopTimeS`. Flight stops on arrival at the requested
terminal state, so a short mission can encounter fewer outages than scheduled.
Both planned and observed counts are reported.

The initial attitude and body rates can also be supplied:

```matlab
MapCorrectionOptions.StartEulerZYXRad = [0 0 0]; % yaw, pitch, roll
MapCorrectionOptions.StartBodyRatesRadps = [0 0 0];
```

## Map, planner and executed correction

The same `ObstaclePositions`, `ObstacleWidth=3` and `ObstacleHeight=15` from
`scripts/initializeSingleUav.m` build the physical scene and the controller's
map. `addMesh` polygon coordinates are ENU: the planner swaps East/North and
uses Down extents `[-15,0]`. The repeated obstacle entry is deduplicated in the
map. No new obstacles or prewritten paths are supplied.

1. A 26-connected 3-D A* search generates a route from the user start to goal.
   Every edge is checked against expanded obstacle boxes, preventing diagonal
   corner cutting. Positive edge costs include length and a clearance penalty.
   Shortcuts require both a full segment collision check and nonincreasing
   sampled cost. Grid and shortcut approximations do not claim globally
   minimum-time flight.
2. A velocity reference slows near each route waypoint and the final goal.
   The nominal acceleration tracks this reference. The route remains active
   in both healthy and outage periods, avoiding a controller reset on recovery.
3. `correctMotion` projects that nominal acceleration onto hard map barrier,
   acceleration and jerk constraints. Healthy LiDAR supplies additional
   constraints for measured surfaces outside the map's geometry uncertainty
   band. Mapped faces are already protected by the full box constraints; they
   are not duplicated as point barriers. During outage, no value from the hidden raw
   cloud is read by the motion controller; the supplied map and independent
   state are sufficient. Healthy returns are transformed at their 10-Hz scan
   pose and held in world coordinates between the 20-Hz control updates.
4. The selected acceleration is allocated into the existing plant's roll,
   pitch, yaw-rate and thrust commands. These commands enter `Rate Transition2`
   directly, as in the augmented ML integration. They are not reprocessed by
   the original VFH/PID. The original follower's stop signal is bypassed; the
   new controller stops on terminal position **and velocity**, or filter failure.
5. During outage `virtualLidar` ray casts the physical boxes at the estimated
   pose, using the existing mounting offset, 180-degree roll, 7 m range and
   15-by-716 organized scan. It provides internal sector inputs; the displayed
   cloud is cleared during outages instead of showing these synthetic scans.
   Map safety constraints use the full 3-D boxes, not just seven forward sectors.

For a convex box, let `h=distance_to_box-radius` and `n` be its outward
distance normal. The filter requires

```text
n' u + 2*k*n' v + k^2*h >= acceleration_error_allowance
```

with the distance-curvature term conservatively omitted outside the convex
box. Workspace planes and healthy measured points add similar constraints.
The QP minimizes `0.5*norm(u-u_nominal)^2` with hard acceleration and slew
bounds. `projectControl` solves this three-variable projection using linear
algebra and independent active sets of size at most three; it has no toolbox
or learned-parameter dependency. An infeasible solution triggers **Stop
Simulation**. Its finite braking output is not labeled feasible or safe.

The safety radius is body radius + minimum surface clearance + geometry
uncertainty + tracking margin (default `0.1+1+0.15+0.25=1.5 m`). Planning adds
`PlanningExtraMarginM=0.6 m`. Default speed is 2.5 m/s; acceleration and jerk
are limited to 1.5 m/s² per axis and 3 m/s³ per axis. Workspace bounds are
`[-10 45; -10 50; -20 -2]` in NED and grid resolution is 1 m. A full 3-D route
may pass above the existing 15 m pillars if the selected mission favors it.
Set `WorldBoundsNedM(3,:)=[-9 -5]` to restrict flight to the old altitude band.

## Results and checks

`mapCorrectionPrepared.ModelPath` identifies the generated `.slx`.
`mapCorrectionResult` contains mission success, arrival time, native-sample
3-D obstacle clearance, filter feasibility and planned/observed outage counts.
`results/map_correction/latest_prepared.json` locates the most recent model.

Each mission directory contains:

- `mission.mat`: options, map, planned route and outage intervals.
- `prepared.json`: baseline/model/mission/source hashes, executed wiring and schedule.
- `control.csv`: time, executed commands, nominal/corrected accelerations,
  references, state, availability, outage-correction use and barrier diagnostics.
- `telemetry.mat`: native position/velocity and scan-rate control evidence.
- `summary.json`: outcome, clearance and instrumentation checks.

Generate a flight-evidence image with:

```matlab
ares.navigation.plotRun(mapCorrectionResult);
```

Run the focused unit tests and end-to-end flight checks with:

```matlab
addpath("scripts");
validation = validateMapCorrectionSingleUav(string(pwd));
```

The focused validation covers the new controller and four executed flights.
Local R2026a validation passed all 16 new unit tests. Healthy flight, five
intermittent outages, and cold-start blackout each reached the default goal at
rest in 22.90 s, with 2.851 m minimum sampled surface clearance. A blackout
flight from `[2 -3 -6 .2 0 0]` to `[26 29 -17 0 0 0]`, with initial yaw 0.3 rad,
reached the goal at rest in 20.35 s with 2.938 m minimum clearance. All filter
steps were feasible. The local generated report is
`results/map_correction/validation.json`; each case links its own saved model,
telemetry and evidence image. These results do not establish a global flight-time
optimum or success for arbitrary missions.

A separate run of the existing `TestAxisStartPlan` suite exposed two pre-existing
failures: the legacy default initial Down coordinate is `+7`, while those tests
expect `-7` and above-ground obstacle exclusions. Those legacy defaults/tests
are unchanged. This mode explicitly initializes the requested NED state.

This is a simulation implementation under a supplied-static-map and independent
localization assumption. The simulator still generates nominal raw scans, but
the new controller discards them during outage. It cannot observe an unmapped
moving object during blackout. The held-command/attitude tracking margins are
engineering bounds; sampled flight checks are not a formal continuous-time or
hardware safety certificate. The configured spherical body radius is inherited
from this project's plant assumptions and must cover the real airframe before
physical deployment.
