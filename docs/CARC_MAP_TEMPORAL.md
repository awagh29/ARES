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
log folder, and each training invocation has a new results folder. Initial
synthetic generation uses about 3.2 GB for clean scan arrays; allow extra room for
recorded exports, checkpoints and the Python environment.

The launcher saves `input_notebook.ipynb`, `effective_config.json`, `console.log`
and `status.json`; training saves its existing best/last checkpoints and reports.
The source notebook remains unchanged. Neither this launcher nor the notebook
automatically resumes an interrupted training run: `last.pt` is retained, but a
new invocation starts a new run. Set sufficient allocation time before the full
run. No CARC job has been submitted by these repository changes.

## Optional: use a notebook interface

The command-line workflow above needs no Jupyter server. If you prefer CARC
OnDemand/Jupyter, use a GPU-backed session with the same environment, set
`ARES_REPO_ROOT` and `ARES_RUN_ROOT` before starting its kernel, prepare missing
exports with `scripts/carc/prepare_map_temporal_data.py`, then run the notebook's
cells normally. Paths will be detected without editing the notebook.
