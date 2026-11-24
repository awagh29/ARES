# Gazebo test pipeline

Flies the ARES single-drone navigation solution in **Gazebo** (real rigid-body
and rotor physics) under lidar faults and wind, and scores every flight with
the same pass/fail rules as the MATLAB code.

**System under test:** Atharva's map-correction controller
(`src/+ares/+navigation`, the "barrier function based path reconstruction" run
by `scripts/runMapCorrectionSingleUav.m`). It is ported line by line to Python
in `ares_gz/navigation/`, and `tests/test_parity.py` checks the port against
outputs produced by the real MATLAB code.

| # | Scenario id | Lidar | Wind |
|---|---|---|---|
| 0 | `S0_baseline` (control run) | healthy | calm |
| 1 | `S1_intermittent` | 5 random outages (MATLAB's generator and seeds) | calm |
| 2 | `S2_blackout` | off for the whole flight | calm |
| 3 | `S3_gusts_light` / `S3_gusts_medium` | healthy | gusts |
| 4 | `S4_gusts_{light,medium}_intermittent` | intermittent | gusts |
| 5 | `S5_gusts_{light,medium}_blackout` | blackout | gusts |
| 6 | three drones + hub | later | later |

Wind levels: light = 1 m/s mean + 3 m/s gust peaks + 0.3 m/s turbulence;
medium = 2 m/s mean + 6 m/s gust peaks + 0.6 m/s turbulence
(`config/single_drone_scenarios.yaml`).

## How a run works

Every 0.05 s (lock-step, so results repeat exactly):

1. Read the drone state from Gazebo and convert it to MATLAB's NED frame.
2. **Lidar fault injection:** during an outage the scan is withheld and the
   controller is told the lidar is down, exactly like the MATLAB model.
3. The controller returns a safe acceleration (its barrier filter keeps
   1.5 m from every building) and a yaw rate.
4. That acceleration becomes a velocity set-point for Gazebo's multicopter
   controller (calibrated by `scripts/diagnose.py`).
5. **Wind:** the drag from (wind - drone velocity) is applied as a force on the body.
6. Gazebo advances exactly 50 physics steps (1 ms each).

A run **passes** when the drone ends at rest within 0.35 m of the goal, never
gets closer than 1.0 m to a building surface, stays within the altitude band,
and every barrier-filter step was feasible. These are the same rules as
`evaluateAxisRun.m` + `runMission.m`.

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

## Running the tests

```bash
python3 run_tests.py --list                              # scenario ids
python3 run_tests.py --scenario S0 --seed 1 --gui        # one run, watch it in the Gazebo window
python3 run_tests.py --all                               # 9 scenarios x seeds 1-10
python3 run_tests.py --scenario S3 S4 --seeds 1-3        # a subset
python3 run_tests.py --report results/single_drone       # rebuild the report
```

Results go to `results/<campaign>/<scenario>/seed<N>_<id>/` (git-ignored):
`manifest.json` (all inputs), `control.csv` (every tick), `odometry_ned.csv`
(native 1 kHz path), `summary.json` (score), `world.sdf`, `logs/`. Each
campaign gets `REPORT.md`, `all_runs.csv`, `trajectories.png` and `clearance.png`.

## When the MATLAB solution changes

```matlab
% in MATLAB, from the ARES folder
addpath("gazebo_test_pipeline/matlab"); exportParityFixtures();
```

then `python3 -m pytest tests/test_parity.py`. A failure names the function
whose Python port must be updated.

## Layout

```
gazebo_test_pipeline/
├── run_tests.py                 command-line entry point
├── config/                      scenario matrix, measured vehicle calibration
├── ares_gz/
│   ├── navigation/              Python port of src/+ares/+navigation (planner, barrier QP, controller, schedule)
│   ├── scene.py world.py        building map from initializeSingleUav.m -> Gazebo world + quadcopter
│   ├── gazebo.py                lock-step Gazebo session over ROS 2 (ros_gz_bridge)
│   ├── run.py                   one scenario run: fault injection, wind, logging, scoring
│   ├── wind.py metrics.py       gust/turbulence/drag model; evaluateAxisRun port
│   └── report.py scenarios.py
├── matlab/exportParityFixtures.m  reference outputs from the real MATLAB code
├── scripts/diagnose.py          one-time Gazebo calibration and sanity checks
└── tests/                       parity + unit tests
```

Only files inside this folder are added; nothing else in the repository is
modified, so merging into `main` is conflict-free.

## Known limits

- The map is fixed and complete (every building is in the controller's map),
  so the controller never needs lidar to avoid them. Lidar faults are expected
  to matter little in calm air; wind is the main stress.
- Gazebo's drone is the 1.5 kg X3 example quadcopter with its own velocity
  controller; MATLAB uses UAV Toolbox's simplified guidance model. Both are
  driven by the same acceleration commands.
- Wind is applied as body drag; rotor-wind aerodynamics are not modelled.
