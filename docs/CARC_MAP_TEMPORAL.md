# Run the map-temporal notebook on USC CARC

This workflow uses your operating directory `/scratch1/awagh`. The notebook finds
its cloned checkout on Linux or Windows; there are no hard-coded Windows paths.
The command-line launcher executes the notebook's code cells in order, so the
model and training loop have one source of truth. Its original 29-cell flow and
your current **1,000 epochs** are preserved.

## Locations

- Checkout and recorded source data: `/scratch1/awagh/ARES`
- Generated files: `/scratch1/awagh/ares-training`
- Results: `/scratch1/awagh/ares-training/results/sparse_reconstruction_map_temporal/<run>`
- Reusable synthetic cache: `/scratch1/awagh/ares-training/data/processed/map_temporal_lidar_v1/<fingerprint>`
- Rebuilt recorded exports, when needed: `/scratch1/awagh/ares-training/exports/<batch>`
- Launcher logs, notebook snapshots and effective settings: `/scratch1/awagh/ares-training/launches/<job-id>_<timestamp>`

Scratch is temporary storage and subject to purging. Copy important checkpoints
and reports to your allocated project storage or local machine after training.
USC describes scratch and project storage in its
[Discovery guide](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery).

## 1. Clone and create an environment (once)

Replace `YOUR_REPOSITORY_URL` below with your Git remote. Push the updated
notebook, `scripts/carc`, this guide, `data/raw`, and the existing `data/batches`
metadata. The latter supplies the finalized batch index and QA reports.

```bash
ssh awagh@discovery.usc.edu
module purge
module load conda
conda create --prefix /scratch1/awagh/conda-envs/ares-map-temporal -c conda-forge python=3.11 pip git-lfs -y
conda activate /scratch1/awagh/conda-envs/ares-map-temporal

GIT_LFS_SKIP_SMUDGE=1 git clone YOUR_REPOSITORY_URL /scratch1/awagh/ARES
cd /scratch1/awagh/ARES
git lfs install --local
git lfs pull --include="data/raw/**"
```

The repository marks `.mat` files for Git LFS. Download the actual raw MAT files;
small pointer text files are insufficient. Existing tracked `episode.npz` files
will also clone normally. `.gitignore` does not untrack files already committed.
If those exports are absent, the preparation step below rebuilds them directly
from the 80 raw telemetry MAT files, without MATLAB. It never rewrites tracked
data or the old Windows export reports.

CARC documents `module load conda` and activation in its
[Conda guide](https://www.carc.usc.edu/user-guides/hpc-systems/software/conda).
If that module is unavailable in your allocation's software stack, use
`module spider conda` to locate the available version.

## 2. Reserve an interactive GPU allocation

From the login node:

```bash
myaccount
salloc --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00
```

This uses your default allocation account. If you have multiple accounts, append
`--account=YOUR_PROJECT_ID` using the account shown by `myaccount`. The `gpu`
partition and GPU request follow USC's
[data-science example](https://www.carc.usc.edu/user-guides/data-science/data-science-hpc).
If your group's guide requires a different partition or GPU type, use that value.
The four-hour limit is an initial request, not a claim that 1,000 epochs fit in it.
Use the first epoch timing to choose the production wall time your account allows.

Wait for allocation and confirm the prompt has moved off `discovery1` onto the
allocated compute node. If it has not, enter `srun --pty bash -l` within the granted
allocation. Then run:

```bash
module purge
module load conda
conda activate /scratch1/awagh/conda-envs/ares-map-temporal
cd /scratch1/awagh/ARES
export ARES_REPO_ROOT=/scratch1/awagh/ARES
export ARES_RUN_ROOT=/scratch1/awagh/ares-training
nvidia-smi
```

Do preparation, GPU checks and training on the allocated compute node. USC's
[job guide](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs)
explains allocations, accounts and job monitoring. Keep the interactive session
alive for training; leaving the allocation releases its resources.

## 3. Install Python packages (once, inside the allocation)

For an existing working CUDA PyTorch >=2.5 environment, keep that installation
and install the remaining packages:

```bash
python -m pip install -r scripts/carc/requirements.txt
```

For a new environment, one explicit supported baseline is PyTorch 2.5.1 with
CUDA 12.1, provided the allocated node's driver supports CUDA 12.1:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r scripts/carc/requirements.txt
```

That wheel is listed in the official
[PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v251).
Use the [current PyTorch selector](https://pytorch.org/get-started/locally/) if the
allocated GPU/driver needs a different supported build. The launcher checks CUDA
and uses BF16 when supported, otherwise FP16. It respects Slurm's GPU visibility;
the single assigned GPU is device 0 inside this process. No separate CUDA toolkit
module is required just to use a compatible prebuilt PyTorch wheel.

## 4. Prepare data and check the notebook

```bash
python -u scripts/carc/run_map_temporal.py --prepare-data --check-only
```

This verifies recorded export hashes, the obstacle map, Python packages, GPU and
the model's cold-blackout forward path. It does **not** generate the synthetic
corpus or start training. When exports are missing, raw conversion does run here.
The raw reader preserves beam ordering and follows `buildLidarEpisode.m`:
duplicate native timestamps keep the last value, motion is interpolated,
angles are unwrapped then wrapped, commands are held, and extrapolation is refused.
A separate audit records the regenerated files' new hashes.

## 5. Run training

For a one-epoch timing run, including first-time synthetic cache generation:

```bash
python -u scripts/carc/run_map_temporal.py --prepare-data --epochs 1
```

For the notebook's complete configured run:

```bash
python -u scripts/carc/run_map_temporal.py --prepare-data
```

The full command preserves the notebook's epoch count, optimizer, scheduler,
splits and architecture. Use `--batch-size 8` if GPU memory is insufficient, or
`--amp-dtype fp16` to request that precision explicitly. Every launch has a new
log folder. New training creates a results folder; resume reuses the original
results folder. Initial
synthetic generation uses about 3.2 GB for clean scan arrays; allow extra room for
recorded exports, checkpoints and the Python environment.

The launcher saves `input_notebook.ipynb`, `effective_config.json`, `console.log`
and `status.json`; training saves its existing best/last checkpoints and reports.
The source notebook remains unchanged when the launcher executes it. For a new
run, the launcher saves within-epoch continuation state every five minutes, at
every epoch, and before validation. It attempts a graceful stop five minutes
before the current Slurm allocation expires. These are training checkpoints,
not a second training implementation. No CARC job is submitted automatically.

## 6. Pick up training in the next salloc allocation

After your present allocation finishes, update the checkout with these changes,
request a new allocation, and activate the same environment. Your checkpoint and
cache directories must remain in place:

```bash
cd /scratch1/awagh/ARES
git pull
salloc --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00
module purge
module load conda
conda activate /scratch1/awagh/conda-envs/ares-map-temporal
cd /scratch1/awagh/ARES
python -u scripts/carc/resume_map_temporal.py
```

The resume command detects `/scratch1/awagh/ares-training` when it exists. If your
run is elsewhere, set `ARES_RUN_ROOT` or pass `--run-root /your/run/root`.
Use the allocation account/partition appropriate for your project, as above.

The script prints its selected checkpoint and saved progress before proceeding.
Automatic selection uses the newest training run with saved checkpoints. Inside
that run it selects the furthest readable training state from `last.pt`,
`last.prev.pt`, `best.pt` and periodic `epoch_*.pt` files. Temporary writes are
ignored. Corrupt candidates are reported and skipped. It does not silently fall
back to a different experiment. To choose a specific run or checkpoint:

```bash
python -u scripts/carc/resume_map_temporal.py --resume "/scratch1/awagh/ares-training/results/sparse_reconstruction_map_temporal/YOUR_RUN_FOLDER"
python -u scripts/carc/resume_map_temporal.py --resume "/path/to/run/last.pt"
```

Normally use `last.pt`, which represents the most recent training state. `best.pt`
can be older: it is selected by validation quality. Both older checkpoint types
include optimizer/scheduler/scaler state, so either can continue training from
its own saved epoch. **The old script did not save unfinished epoch progress.**
Any updates after its saved epoch must be redone; that information cannot be
recovered from `best.pt`. New version-2 checkpoints also save the next batch,
partial loss totals, the augmentation epoch, best weights, and Python/NumPy/
PyTorch/CUDA random states. The resume path skips saved batches without loading
their scan windows or repeating their optimizer updates.

The original model, optimizer, learning rate, cosine scheduler position and
horizon, scaler, best validation state, patience counter, metric history,
training seeds, flight splits and data fingerprint are restored. There is no
fresh optimizer or learning-rate warm-up. Configuration mismatches fail clearly:
do not change batch size, precision, data generation or masking when continuing
the same experiment. A saved BF16 run needs a BF16-capable GPU. Floating-point
results can still differ across GPU models/software versions; retain your
environment for the closest numerical continuation.

Continuations append to the same `training.log` and rebuild CSV/JSON metric rows
from the authoritative checkpoint history, so a killed write cannot duplicate an
epoch. Each allocation also receives its own launcher log/snapshot. An OS lock
prevents two trainers from writing the same run; it releases automatically on
process exit or kill. Checkpoints are replaced atomically and `last.prev.pt`
retains another complete save. Model/data cells still run to reconstruct the
environment and verify/reuse the cache before restoring training state.

Useful timing controls (also supported by the new-run launcher):

```bash
python -u scripts/carc/resume_map_temporal.py --checkpoint-minutes 2
python -u scripts/carc/resume_map_temporal.py --stop-after-minutes 220
python -u scripts/carc/resume_map_temporal.py --walltime-margin-minutes 10
```

By default, remaining allocation time comes from Slurm's
[`squeue --format=%L`](https://slurm.schedmd.com/squeue.html). If that query is
unavailable, periodic saves still work; use `--stop-after-minutes` for a soft
session limit. SIGTERM, SIGINT (Ctrl+C), and SIGUSR1 request a save after the
current optimizer batch. A hard SIGKILL cannot save: resume uses the last complete
file, losing only progress since that save. A long running batch/validation pass
can exceed the margin, so the periodic checkpoint remains the recovery point.

The epoch target means **total epochs**, not additional epochs. A saved 1,000-epoch
run continues toward 1,000. If an earlier one-epoch timing run should now continue
to 1,000, use `--epochs 1000`; its original scheduler horizon must already cover
that target. The scheduler is never silently restarted or stretched. When all
epochs are already done, training updates are skipped and final reports can be
regenerated. Interrupted validation/report generation can be rerun without
repeating completed training updates.

## Optional: use a notebook interface

The command-line workflow above needs no Jupyter server. If you prefer CARC
OnDemand/Jupyter, use a GPU-backed session with the same environment, set
`ARES_REPO_ROOT` and `ARES_RUN_ROOT` before starting its kernel, prepare missing
exports with `scripts/carc/prepare_map_temporal_data.py`, then run the notebook's
cells normally. Paths will be detected without editing the notebook.
