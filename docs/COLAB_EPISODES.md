# Start with the episodes in your existing Colab notebook

This uses the `episode.npz` files created by `scripts/exportLidarBatch.py`.
No simulations or MATLAB conversion are needed. Keep your existing repository
checkout and CI setup. Once the new starter script is synced into that checkout,
add the following cells to your notebook.

## 1. Load the dataset

Change only `REPO_ROOT` if your checkout has a different location:

```python
from pathlib import Path
import sys

REPO_ROOT = Path("/content/ares")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from colab_episode_starter import EpisodeDataset, make_run_splits, lidar_grid

data = EpisodeDataset(REPO_ROOT)
runs = data.runs
scans = data.load_scan_table()

print(f"{len(runs)} runs; {len(scans):,} scans")  # 80 runs; 19,185 scans
display(runs[["RunIndex", "RunId", "Axis", "ScanCount",
              "MissionSuccess", "CollisionOccurred", "SafetyChecksPassed"]].head())
display(scans.head())
```

The loader needs NumPy and pandas. If either import is missing, install it in
a notebook cell with `%pip install numpy pandas`.

The original index stores `A:\Projects\ares\...` paths. The starter reconstructs
the NPZ location from the checkout and run ID, without editing that index.
Other original path columns and NPZ source-path metadata are provenance from
the collection machine; do not open those Windows paths in Colab.

## 2. Inspect one episode

```python
run_id = runs.iloc[0]["RunId"]
episode = data.load_episode(run_id)

print(episode["lidar_ranges_m"].shape)  # [scans, 10740]
print(episode["position_ned_m"].shape)  # [scans, 3]: North, East, Down
print(episode["action_current"].shape)  # [scans, 4]

# A few compact EDA summaries. Do not call dropna() on every field.
display(scans.groupby("Axis")[["ValidReturnFraction", "DistanceToFinalGoalM"]].describe())
```

Each episode contains time, sample indices, LiDAR ranges and masks, position,
velocity, attitude, body angular velocity, actions, mission goal, controller
reference, run ID, and metadata. The first previous action is unavailable:
its values are NaN and `previous_action_available[0]` is false. No-return
LiDAR beams have range 7 and a false valid-return mask; they are not outages.
All 80 runs remain available, including the 33 clearance violations and two
sampled collisions. `Status=complete` means the data collection completed;
it does not mean the flight met every safety requirement.

## 3. Optional first plots

```python
import matplotlib.pyplot as plt

position = episode["position_ned_m"]
scan_index = len(episode["time_s"]) // 2
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(position[:, 1], position[:, 0])
axes[0].scatter(position[0, 1], position[0, 0], label="Start")
goal = episode["goal_position_ned_m"][0]
axes[0].scatter(goal[1], goal[0], marker="*", label="Goal")
axes[0].set(xlabel="East (m)", ylabel="North (m)", title=run_id)
axes[0].axis("equal")
axes[0].legend()
image = axes[1].imshow(lidar_grid(episode, scan_index), aspect="auto", vmin=0, vmax=7)
axes[1].set(xlabel="Azimuth index", ylabel="Elevation index", title="LiDAR range (m)")
fig.colorbar(image, ax=axes[1], label="m")
plt.tight_layout()
plt.show()
```

`lidar_grid` creates a plotting copy with the correct elevation-fastest channel
ordering. It displays no returns as NaN while preserving the original arrays.
Install `matplotlib` only if it is missing from your runtime.

## 4. Make run-level splits before ML

```python
splits = make_run_splits(runs, seed=42)
display(splits.groupby(["Split", "Axis"]).size().unstack())

train_ids = splits.loc[splits["Split"].eq("train"), "RunId"].tolist()
validation_ids = splits.loc[splits["Split"].eq("validation"), "RunId"].tolist()
calibration_ids = splits.loc[splits["Split"].eq("calibration"), "RunId"].tolist()
test_ids = splits.loc[splits["Split"].eq("test"), "RunId"].tolist()

# Read only selected arrays, one complete trajectory at a time.
for run_id, values in data.iter_episodes(
    train_ids, fields=["position_ned_m", "velocity_ned_mps", "action_current"]
):
    # Build your model-independent features here.
    print(run_id, values["position_ned_m"].shape)
    break
```

The split contains 56 training, 8 validation, 8 calibration, and 8 test runs,
with equal North/East representation in each. Preserve the assignment for
your experiment. Fit scalers on training runs only; create windows and
synthetic sensor failures after splitting. Keep calibration for uncertainty
calibration and test runs for final evaluation. Because every run uses the
same scene, this is a held-out-start split, not evidence of generalization to
unseen maps. No ML architecture, normalization, or synthetic outage is imposed.

Do not load all 80 dense LiDAR arrays into a single notebook variable. The
compact CSV table is suitable for broad EDA; stream NPZ episodes as needed.
The entire NPZ export is approximately 100 MiB compressed, but decompressed
arrays occupy substantially more memory.

Colab's runtime files are temporary. Save experiment outputs and your split
assignment through your existing persistent-storage workflow. A notebook
opened from GitHub does not by itself put the dataset in its runtime. If the
loader reports missing files, make the `data/batches` and
`data/processed/lidar_episode_v1` folders available in the checkout. If it
detects Git LFS pointers, retrieve their actual contents using your existing
GitHub authentication setup.

References: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[NumPy NPZ loading](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
