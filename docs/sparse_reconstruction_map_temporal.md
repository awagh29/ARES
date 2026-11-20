# Map-wide temporal sparse reconstruction notebook

Open `notebooks/train_sparse_reconstruction_map_temporal.ipynb` and run its cells
in order. It is a separate copy of the single-episode baseline with the same
29 cells, cell order and sections 0–9. The original notebook is unchanged.
All model, dataset, generation, training and inference definitions are embedded;
the notebook does not depend on the temporary authoring files.

## Defaults and training

- Combine the audited batch `c3eaab07-d882-4825-98a5-84c278cf8935` (80 actual
  MATLAB flights) with 192 additional synthetic routes.
- Sample synthetic starts/goals at North/East −5–40 m and altitude 1–10 m
  (NED Down −10 to −1 m). Keep vehicle centers at least 0.7 m beyond each
  building face: 0.1 m vehicle radius plus 0.6 m surface clearance.
- Connect endpoints using collision-free building-footprint visibility graphs.
  Interpolate altitude and speed, and vary heading, roll and pitch. These are
  kinematic histories, **not** validated MATLAB controller rollouts.
- Split complete trajectories by source and recorded start axis, approximately
  80/10/10, before constructing causal windows or introducing failures. Defaults
  produce 218/27/27 train/validation/test trajectories.
- Use 16 time steps at 10 Hz. The current scan retains full 15×716 resolution;
  older observations are compact 3×90 summaries of nearness, hit fraction and
  availability. Missing past frames have explicit presence flags.
- Train with measured-scan noise, pose bias/noise/drift, variable beam/sector
  loss, repeated blackout bursts, long blackouts and flights blacked out from
  their first step. No fixed outage count is assumed.
- Retain range-weighted MSE, AdamW, cosine scheduling, checkpoints,
  logs and plots. Early stopping is configurable and disabled by default, as in
  the baseline. Training faults and pose errors change every epoch; validation
  faults and pose errors remain fixed.

The current configured epoch count is 1,000. Before running, select a CUDA-enabled PyTorch
>=2.5 kernel. Select FP16 if the GPU does not support BF16. The default batch size
is 16; reduce it if GPU memory is insufficient. CPU mode is available for small
checks. NumPy, pandas, matplotlib and PyTorch are the only required packages.

Paths are detected from the checkout on Windows or Linux. `ARES_REPO_ROOT` can
select a checkout and `ARES_RUN_ROOT` can put generated files outside it. Without
overrides, generated files stay under the checkout. See
[CARC_MAP_TEMPORAL.md](CARC_MAP_TEMPORAL.md) for your `/scratch1/awagh` setup,
environment creation, raw-data preparation and `salloc` commands.

Section 5 audits sources and prepares reusable, fingerprinted scan caches under
`<ARES_RUN_ROOT>/data/processed/map_temporal_lidar_v1`. Allow several GB of disk space. The
builder streams one trajectory at a time, reports progress and verifies completed
cache files. It never changes raw telemetry. Section 8 starts training. No
production training or full synthetic cache generation was performed in creating
this notebook.

## Inputs and architecture

The supplied obstacle map is an intentional runtime input. Manifest ENU obstacle
centers are converted to NED and duplicate buildings are removed. Each obstacle
is represented by `[center_north, center_east, center_down, full_size_north,
full_size_east, full_size_down]`. Sensor rays and the roll-180 mounting with a
`[0,0,-0.4]` m translation match the existing 15×716 sensor. No ground plane is
invented: the source scene contains the building meshes only.

The sparse convolution/U-Net branch processes current available LiDAR. A map
branch receives expected ranges independently rendered from the supplied map
and **estimated** current pose. A shared obstacle encoder processes body-relative
centers, closest box points and dimensions. A causal GRU combines observed scan
history, pose/velocity/attitude, body-relative mission goal, lag and blackout age.
Its context conditions the decoder. The output is a bounded learned correction
to the map prediction on missing beams; available measurements are preserved.

Clean synthetic labels are rendered at true poses. Pose error augmentation is
applied separately to model inputs, so the model is not simply handed its clean
target during blackout. Recorded LiDAR is used as the target for actual flights.
Future frames, hidden clean returns and predictions are never fed into historical
observation buffers. The first step needs no healthy-scan warm-up.

## Validation and saved results

Every run has a new directory under
`results/sparse_reconstruction_map_temporal`. It contains configuration, source
hashes, splits, saved validation faults, metrics, best/last checkpoints and plots.
`best_validation_report.json` compares reconstruction and map-only prediction.
`holdout_report.json` runs the best-validation checkpoint on held-out flights
under assigned, full cold-start, long-blackout and repeated-burst fault profiles.
The test folds do not select the checkpoint.

Reports separate recorded/synthetic data, full blackouts, cold starts and
blackouts exceeding the history horizon. They include overall and missing-beam
errors plus the fraction of hidden obstacle returns overestimated by more than
0.5 m. Empty subgroups report null values. Inspect these comparisons before
deciding whether additional actual MATLAB trajectories are needed.

## Three-UAV use and limits

`MultiUavReconstruction(model, CONFIG)` shares model weights while keeping a
separate causal buffer per UAV ID. Call `step(uav_id, timestamp_s, ranges, valid,
available, state9, goal3, boxes_Ox6)` at each 10 Hz scan time. State order is NED
position, NED velocity, roll/pitch/yaw radians. Scans use `[15,716]` sensor order;
supported no-return beams are 7 m. Supply estimated pose and the known map even
during full LiDAR blackout. Reset a UAV's buffer at mission start. A timing gap
clears its history; another UAV's frames cannot enter it.

Generalization is to new flights on this known static map. Long blackout behavior
still depends on localization and map accuracy. A reconstruction score does not
establish arbitrary-goal mission success, collision safety, or inter-UAV avoidance.
Moving UAVs and new obstacles are absent from the static map. MATLAB closed-loop
validation remains a separate next step after training. This notebook does not
change the simulation or existing controller. Its new architecture is incompatible
with the existing single-episode MATLAB exporter and requires a separate export
integration before Simulink deployment.

## Checks

`tests/python/test_map_temporal_sparse_notebook.py` checks notebook flow, actual
recorded-scan geometry/order, synthetic route clearance, whole-flight splits,
causal and per-UAV history isolation, input masking, repeated/long/cold-start
blackouts, cache integrity, gradients, a tiny training/artifact run, and the
default architecture's CUDA forward/backward path.
