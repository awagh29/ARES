# LiDAR-on axis-start dataset

The collection contains 80 independent simulations in the saved
`ares_single_uav_top` scenario: 40 North-axis starts `[N,0,-7]` and 40
East-axis starts `[0,E,-7]`, in meters in NED. The final goal is always
`[30,35,-7]`. The first waypoint is set to the sampled start so each mission
explicitly runs from that start to the same goal.

## Sampling and reproducibility

`ares.data.createAxisStartPlan(20261102)` draws one random point within each
of 40 equal-length strata along each axis's feasible intervals. North spans
`(0,30)`; East spans `(0,35)`. East positions in `[7.4,12.6]` are excluded
because a current obstacle intersects that axis. Starts have the configured
vehicle radius plus minimum surface-clearance margin. The altitude,
obstacle layout, controller, sensor settings and 60-second maximum duration
are otherwise retained. Runs can terminate earlier through the existing
waypoint-follower stop logic.

The initializer's polygon vertices use the UAV scenario's default ENU mesh
frame. Their horizontal coordinates must be swapped to compare with NED
drone state. This collection leaves the meshes unchanged and uses converted
NED boxes for spawn checks and outcome evaluation. Its outcome labels do
not use the old `runSingleUav` clearance calculation or its disabled safety
assertion.

Each manifest retains the existing random UUID run ID and schema `2.0.0`.
It adds batch/axis indices, sampling and simulation seeds, exact configuration,
effective LiDAR block parameters, source hashes, software versions and
measured outcomes. No sensor fault or reconstruction model is enabled.
The effective LiDAR block has noise off, range accuracy `0.02`, maximum
range `7 m`, and sample period `0.1 s`. Initializer-only variables should not
be mistaken for these effective model parameters.

## Files

- `data/batches/<batchId>/plan.csv`: all 80 sampled starts and seeds.
- `data/batches/<batchId>/index.csv`: authoritative run IDs, paths and outcomes
  after finalization. During concurrent collection, `index_worker*.csv`
  records each worker's current progress.
- `data/batches/<batchId>/all_scans.csv`: compact combined per-scan EDA data.
- `data/batches/<batchId>/dataset_qa.json` and `summary.json`: verification and
  collection totals.
- `data/batches/<batchId>/attempts/`: immutable attempt records, including any
  simulator/instrumentation failures and their artifact paths.
- `data/raw/<runId>/manifest.json` and
  `lidar-on-axis-single-uav_telemetry.mat`: existing immutable raw-run layout.
- `data/processed/lidar_episode_v1/<runId>/`: model-neutral `episode` MAT file
  and `scans.csv`, plus `episode.npz` for Python. No GRU-specific dataset or
  training features are created.

The raw telemetry keeps native timestamps, position/velocity, attitude RPY,
body angular velocity, commands, controller reference, enable state,
avoidance status, organized XYZ scans, range vectors and return-validity
masks. State is simulator truth, not output from a state estimator.
Organized XYZ coordinates use the LiDAR sensor frame (x forward, y left,
z up); vehicle position/velocity use NED. The manifest retains the
initializer's sensor mounting location and angles.

The processed episode uses the LiDAR scan timeline. Continuous state is
interpolated linearly; angles are unwrapped before interpolation and wrapped
afterwards; commands and the local reference use previous-value hold.
Duplicate native solver timestamps retain the last sample. Extrapolation
is prohibited. Timestamp differences within `1e-9 s` of a native interval
boundary are treated as floating-point roundoff: only the interpolation
query is clipped to that boundary; original LiDAR timestamps are retained.

`episode.goal.positionNed_m` is the fixed final mission goal.
`episode.reference.desiredPositionNed_m` is the avoidance-adjusted controller
reference. They are different quantities and are both retained.

Range matrices have shape `[scans,10740]` with a `15 x 716` organized grid.
Elevation varies fastest, then azimuth (MATLAB column-major order).
No-return beams use range `7` and `validReturnMask=false`. They are not
sensor outages: availability is true and missingness false throughout these
healthy runs. Previous action means the command at the preceding LiDAR
scan. Its first row is unavailable (`NaN`, `previousAvailable=false`), not an
invented zero command. `NearestValidReturnM` in CSV is likewise `NaN` when
a scan has no valid returns.

## Reading and collecting

In MATLAB, add the project's `src` and `scripts` folders to the path:

```matlab
index = readtable(fullfile(batchPath,"index.csv"),TextType="string");
data = load(index.EpisodePath(1),"episode");
episode = data.episode;
ranges = episode.lidar.ranges_m;
validReturns = episode.lidar.validReturnMask;
scans = readtable(fullfile(batchPath,"all_scans.csv"),TextType="string");
```

For Python, `python_index.csv` adds the verified `NpzPath` for each run.
`python_export_qa.json` verifies all exports and records their hashes:

```python
import csv
import numpy as np
with open(batch_path / "python_index.csv", newline="") as stream:
    runs = list(csv.DictReader(stream))
with np.load(runs[0]["NpzPath"], allow_pickle=False) as episode:
    ranges = episode["lidar_ranges_m"]       # [scans, 10740]
    valid = episode["lidar_valid_return_mask"]
    position = episode["position_ned_m"]     # [scans, 3]
    goal = episode["goal_position_ned_m"]
```

To export a newly finalized batch, install `numpy` and `h5py`, then run
`python scripts/exportLidarBatch.py --batch <batchPath>`. Export refuses
unverified datasets and checks each NPZ array against its source. Existing
NPZ files are verified and kept. No preprocessing or precision reduction is
performed.

Create and collect a new batch serially:

```matlab
[batchPath,index] = captureLidarAxisBatch(projectRoot);
index = finalizeLidarAxisBatch(batchPath);
```

Resume an interrupted serial collection with
`captureLidarAxisBatch(projectRoot,ResumeBatchPath=batchPath)`. Failed
attempts retain their own IDs and logs; a retry receives a fresh UUID.
Completed artifacts are read back and skipped. Resume refuses changed
source code/software or a different output root.

A documented timestamp-boundary correction was applied during this batch.
Immutable `source_amendment_*.json/.mat` files retain old/new collection-code
hashes without changing the plan, scene, controller, seeds or raw runs.
Three saved raw runs were recovered after their episode exports stopped on
an approximately `7e-15 s` endpoint difference. Their original UUIDs and
attempt records are retained; `recoveries/` records the corrected exports
and verifies that raw files and manifests stayed unchanged.

For concurrent collection, prepare once with `PrepareOnly=true`, then run
`runLidarAxisWorker` in separate MATLAB processes using disjoint run indices
and worker IDs such as `worker01`. Each worker has an isolated cache and
checkpoint. Finalize only after all workers finish. A running worker must
never be launched twice with the same worker ID.

## EDA and ML cautions

Retain complete run boundaries. Split run IDs into training, calibration and
test sets before creating windows, fitting normalization, or simulating
missing observations. This dataset does not fit preprocessing or select a
learning architecture.

Mission failure does not imply bad instrumentation: collisions, clearance
violations, goal failures and timeouts remain in the dataset with explicit
labels. `SafetyChecksPassed` enforces the configured altitude and minimum
surface margin; `MissionSuccess` additionally requires reaching the goal at
termination. Collision/clearance checks evaluate every native position
sample against the known boxes and spherical vehicle radius; they are not
a proof about continuous motion between samples. Inspect both goal and
clearance metrics when selecting data for a particular analysis.

Some fields in the broad existing data contract describe future estimation,
Bayesian reconstruction and safety-filter functionality. Those components
are not present in the current model; this collection does not fabricate
their outputs.
