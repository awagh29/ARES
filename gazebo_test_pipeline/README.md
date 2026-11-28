# Gazebo test pipeline

Flies the ARES navigation solution in **Gazebo** (real rigid-body and rotor
physics) under lidar faults and wind, and scores every flight with the same
pass/fail rules as the MATLAB code: five single-drone experiments, and the same
five with **three drones and the swarm hub** (see [Swarm hub](#swarm-hub-three-drones)).

**System under test** (chosen in `config/single_drone_scenarios.yaml`):

- `map-temporal-cnn` (default): Atharva's **map-temporal sparsity-invariant
  CNN** (`src/+ares/+mapTemporal`). During lidar outages the CNN/GRU
  reconstructs the 15x716 scan. `controller.guidance` picks what flies on it:
  - `planner` (default): his trajectory-sampling local planner + braking
    governor (`TrajectoryPlanner`, `SpeedGovernor`); VFH is bypassed;
  - `governor`: VFH + position PID + braking governor;
  - `vfh`: the original waypoint follower, VFH and PID.

  The guidance runs in **MATLAB, unchanged** (Atharva's code plus MathWorks'
  own blocks), through `matlab/runGuidanceBridge.m`. The velocity/attitude PID
  runs in Python at 100 Hz and Gazebo is the vehicle. The model file is
  `controller.model_path` (`--guidance` / `--model` override both).
- `map-correction`: Atharva's barrier/map-correction controller
  (`src/+ares/+navigation`), ported to Python in `ares_gz/navigation/` and
  checked against MATLAB by `tests/test_parity.py`.

```text
 Gazebo (WSL)          Python runner (WSL)                         MATLAB (Windows)
 physics, lidar  --->  fault injection, wind, scoring  --10 Hz-->  CNN reconstruction
 1.5 kg quad     <---  PID cascade (100 Hz)            <---------  waypoint follower + VFH + goal hold
```

| # | Scenario id | Lidar | Wind |
|---|---|---|---|
| 0 | `S0_baseline` (control run) | healthy | calm |
| 1 | `S1_intermittent` | 5 random outages (MATLAB's generator and seeds) | calm |
| 2 | `S2_blackout` | off for the whole flight | calm |
| 3 | `S3_gusts_light` / `S3_gusts_medium` | healthy | gusts |
| 4 | `S4_gusts_{light,medium}_intermittent` | intermittent | gusts |
| 5 | `S5_gusts_{light,medium}_blackout` | blackout | gusts |
| 6 | the same five with three drones + hub | `run_swarm_tests.py` | see below |

Wind levels: light = 1 m/s mean + 3 m/s gust peaks + 0.3 m/s turbulence;
medium = 2 m/s mean + 6 m/s gust peaks + 0.6 m/s turbulence
(`config/single_drone_scenarios.yaml`).

## How a run works

Every 0.05 s (lock-step, so results repeat exactly):

1. Read the drone state from Gazebo and convert it to MATLAB's NED frame.
2. **Lidar fault injection:** during an outage the scan is withheld and the
   controller is told the lidar is down, exactly like the MATLAB model.
3. The controller returns an acceleration and a yaw rate. For the CNN stack:
   every 0.1 s MATLAB returns VFH's target (from the real scan, or from the
   CNN reconstruction during an outage); every 0.01 s the PID cascade turns it
   into an acceleration.
4. That acceleration becomes a velocity set-point for Gazebo's multicopter
   controller (calibrated by `scripts/diagnose.py`).
5. **Wind:** the drag from (wind - drone velocity) is applied as a force on the body.
6. Gazebo advances exactly one controller tick (1 ms physics steps).

A run **passes** when the drone ends at rest within 0.35 m of the goal, never
gets closer than 1.0 m to a building surface, stays within the altitude band,
and the controller never failed (CNN: every outage scan was reconstructed;
map-correction: every barrier-filter step was feasible). These are the rules of
`evaluateAxisRun.m` + the matching `runMission.m`.

## Setup (once, in WSL Ubuntu 26.04)

```bash
sudo apt update && sudo apt install -y ros-lyrical-ros-gz   # Gazebo Jetty + ROS 2 bridge
source /opt/ros/lyrical/setup.bash
cd "/mnt/c/Users/<you>/Desktop/14) CSCI-513 ARES/ARES/gazebo_test_pipeline"
python3 -m pytest tests -q          # port vs MATLAB + pipeline unit tests (no Gazebo needed)
python3 scripts/diagnose.py         # measures this Gazebo install, writes config/vehicle_calibration.json
```

If `diagnose.py` reports a lidar/rendering error, retry with
`--no-headless-rendering` (renders through the WSLg desktop instead of EGL).

For the CNN controller, MATLAB R2026a must be installed on Windows (the
runner starts `/mnt/d/bin/matlab.exe`; set `ARES_MATLAB` if it lives
elsewhere) and `controller.model_path` must point to the checkpoint. Check the
MATLAB link without Gazebo:

```bash
python3 scripts/bridge_smoke.py --mode intermittent --seconds 20
```

Optional, about 4x faster runs: install **MATLAB Coder** (Add-Ons, included in
the USC license), then in MATLAB run `buildVfhMex` from
`gazebo_test_pipeline/matlab`. It compiles MathWorks' VFH block and refuses to
keep the MEX unless it matches the interpreted block.

## Running the tests

```bash
python3 run_tests.py --list                              # scenario ids
python3 run_tests.py --scenario S0 --seed 1 --gui        # one run, watch it in the Gazebo window
python3 run_tests.py --all --workers 2 --video           # 9 scenarios x 2 starts x seeds, 2 at a time, with videos
python3 run_tests.py --scenario S3 S4 --seeds 1-3        # a subset
python3 run_tests.py --all --controller map-correction   # the other controller
python3 run_tests.py --report results/single_drone_map-temporal-cnn   # rebuild a report
```

Every scenario is flown from each start point in `starts:` (YAML; both at 7 m
altitude, goal (30, 35)) and each seed. `--start W1` limits it to one start.
`--workers N` runs N simulations at once, each with its own MATLAB, Gazebo and
ROS 2 domain (2 is safe on a 16 GB PC, 3 is the practical maximum); the longest
runs (blackouts) start first.

### Watching and videos (replay mode)

The live test pauses Gazebo while MATLAB computes each command, so the GUI
alternates play/pause. To see a run smoothly, replay it afterwards: the
recorded flight is re-rendered frame by frame (25 fps, nothing re-simulated).
The drone is drawn 3x larger, green while the lidar works and orange while the
CNN reconstruction is used; a HUD shows time, lidar/CNN state, wind, clearance,
a live mini-map and the final PASS/FAIL.

```bash
sudo apt install -y ffmpeg                                        # once, for MP4 encoding
python3 scripts/replay.py results/first_flight                     # every run in a campaign -> MP4s
python3 scripts/replay.py results/<campaign>/<scenario>/<run> --views overview --speed 2
python3 scripts/replay.py results/<campaign>/<scenario>/<run> --watch   # watch in the Gazebo GUI
python3 run_tests.py --scenario S1 --seed 513 --video             # run, then render its videos
```

Videos: `replay_overview.mp4` (angled view) and `replay_top.mp4` (top-down)
in each run folder, plus copies in `results/<campaign>/videos/`. On Windows
they are under `ARES\gazebo_test_pipeline\results\`.

Results go to `results/<campaign>/<scenario>/<start>/seed<N>_<id>/` (git-ignored):
`manifest.json` (all inputs), `control.csv` (every tick), `odometry_ned.csv`
(native 1 kHz path), `summary.json` (score), `world.sdf`, `logs/`. Each
campaign gets `REPORT.md`, `all_runs.csv`, `trajectories.png` and `clearance.png`.

## Swarm hub (three drones)

`run_swarm_tests.py` flies a three-drone search-and-rescue mission in one
Gazebo world, each drone running the same CNN + planner in its own MATLAB
session (in parallel). The drones talk only to the **hub**
(`ares_gz/swarm/hub.py`, star topology, simulated radio delay and loss). The hub:

- **shares a map**: fuses the drones' *measured* scans (never reconstructed
  ones, never untrusted drones) into one occupancy grid, and tracks what the
  victim cameras have searched;
- **explores**: sends drones to unsearched areas and LiDAR frontiers
  (greedy auction, targets kept 6 m apart), then to their rendezvous slots;
- **reports victims**: each report says which drone saw it and whether that
  drone's LiDAR had failed; a report from an untrusted drone is confirmed by a
  trusted one (that task pre-empts exploration);
- **trusts interpretably**: `trust = 1 - (0.35 physical + 0.30 temporal + 0.20
  peer + 0.15 historical)`, each with a plain-language reason
  (`ares_gz/swarm/trust.py`). A 1-2 s outage *flags* a drone (DEGRADED); a
  blackout longer than 3 s *quarantines* it;
- **contains and recovers**: DEGRADED = slower, no new tasks, scans not fused;
  QUARANTINED = hover 3 s, then limp to its slot at 0.6 m/s on the CNN, its task
  reassigned; LOST (no uplink 1 s) = task reassigned; a drone that hears nothing
  from the hub for 0.5 s hovers;
- **keeps the invariant**: no two drones ever closer than **3 m**. Each pair
  constraint of a control barrier function (h = |pi-pj|^2 - D^2,
  dh/dt >= -gamma*h) is split between the two drones; each filters its own
  command onboard, the reliable drone taking 80% of the duty next to an
  unreliable one, and stale positions inflate D (`ares_gz/swarm/cbf.py`).

```bash
python3 run_swarm_tests.py --list
python3 run_swarm_tests.py --scenario S0 --video          # one mission + videos
python3 run_swarm_tests.py --all --video                  # all 10 swarm scenarios
python3 run_swarm_tests.py --scenario S2 --comms lossy    # stress the radio
python3 scripts/swarm_dry_run.py --scenario S2_blackout   # hub logic only, no Gazebo/MATLAB, seconds
```

Scenarios (`config/swarm_scenarios.yaml`): intermittent outages independent per
drone *and* all at once; blackout on UAV2 (the containment case); light/medium
gusts. A mission **passes** with no crash, the 3 m invariant held at 1 kHz,
every trusted drone 1.0 m from buildings, all trusted drones at their slots,
and every victim confirmed by a drone with working LiDAR. Each run writes
`summary.json`, `hub_events.json` (every hub decision with its reason),
`trust.csv`, `swarm_map.png` and per-drone `drones/<UAV>/control.csv`; videos
show every drone's LiDAR state, hub mode, task and the live separation.

## When the MATLAB solution changes

```matlab
% in MATLAB, from the ARES folder
addpath("gazebo_test_pipeline/matlab"); exportParityFixtures();
```

then `python3 -m pytest tests/test_parity.py`. A failure names the function
whose Python port must be updated.

## Layout

```text
gazebo_test_pipeline/
├── run_tests.py                 single-drone command-line entry point
├── run_swarm_tests.py           three-drone (hub) entry point
├── config/                      scenario matrix, measured vehicle calibration
├── ares_gz/
│   ├── controllers/             systems under test: map-temporal CNN (MATLAB bridge + PID), map-correction
│   ├── matlab_bridge.py         starts MATLAB from WSL and talks to runGuidanceBridge.m
│   ├── model_artifact.py        finds the checkpoint / reuses its MATLAB export
│   ├── navigation/              Python port of src/+ares/+navigation (planner, barrier QP, controller, schedule)
│   ├── scene.py world.py        building map from initializeSingleUav.m -> Gazebo world + quadcopter
│   ├── gazebo.py                lock-step Gazebo session over ROS 2 (ros_gz_bridge)
│   ├── run.py                   one scenario run: fault injection, wind, logging, scoring
│   ├── swarm/                   hub, trust model, shared map, exploration, CBF, radio, agent, runner, replay
│   ├── wind.py metrics.py       gust/turbulence/drag model; evaluateAxisRun port
│   └── report.py scenarios.py
├── matlab/
│   ├── runGuidanceBridge.m      runs Atharva's CNN + planner/governor (or VFH) + goal hold for Python; swarm retargeting
│   ├── aresVfhStep.m, buildVfhMex.m   the model's VFH block, optionally compiled (verified) for speed
│   └── exportParityFixtures.m   reference outputs for the map-correction port
├── scripts/
│   ├── diagnose.py              one-time Gazebo calibration and sanity checks
│   ├── replay.py                smooth Gazebo replays of finished runs -> MP4 videos
│   ├── bridge_smoke.py          MATLAB-link check without Gazebo
│   ├── swarm_dry_run.py         hub logic check without Gazebo or MATLAB
│   └── wrench_check.py          wind-force application stays fast and correct over long runs
└── tests/                       parity + unit tests
```

Only files inside this folder are added; nothing else in the repository is
modified, so merging into `main` is conflict-free.

## Known limits

- The swarm CBF works on the drones' velocity commands; the 0.5 m buffer
  absorbs tracking lag, and the 3 m rule is checked on the 1 kHz Gazebo poses.
  Obstacle avoidance stays with each drone's own planner.
- Gazebo's drone is the 1.5 kg X3 example quadcopter with its own velocity
  controller; MATLAB uses UAV Toolbox's simplified guidance model. Both are
  driven by the same acceleration commands.
- Wind is applied as body drag; rotor-wind aerodynamics are not modelled.
