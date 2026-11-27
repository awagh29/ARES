#!/bin/bash
#SBATCH --job-name=ares_map_temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch1/awagh/ares-training/slurm_logs/%x_%j.out
#SBATCH --error=/scratch1/awagh/ares-training/slurm_logs/%x_%j.err

set -euo pipefail

# Pass a checkpoint/run path as $1, or resume the latest saved run by default.
# "fresh" is an explicit opt-in only. Continuations pass the total target as $2.
MODE="${1:-auto}"
TARGET_EPOCHS="${2:-800}"

# 1. Load Conda (no separate CUDA module needed per guide; PyTorch wheel bundles CUDA)
module purge
module load conda

# Initialize and activate the scratch Conda prefix cleanly in non-interactive batch mode
eval "$(conda shell.bash hook)"
conda activate /scratch1/awagh/conda-envs/ares-map-temporal

# 2. Set repository & run root environment variables
export ARES_REPO_ROOT=/scratch1/awagh/ARES
export ARES_RUN_ROOT=/scratch1/awagh/ares-training
export PYTHONUNBUFFERED=1

cd "$ARES_REPO_ROOT"
mkdir -p "$ARES_RUN_ROOT/slurm_logs"

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Start Time:    $(date)"
echo "Mode:          $MODE"
echo "=========================================="
nvidia-smi

# 3. Verify environment and recorded data/cache readiness before launching
echo "=== Verifying Python, PyTorch CUDA & Data Exports ==="
python -u -c "
import torch
print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device 0:', torch.cuda.get_device_name(0), '| VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
"
python -u scripts/carc/run_map_temporal.py --prepare-data --check-only --epochs "$TARGET_EPOCHS"

# 4. Continue the saved run to the total target and chain time-limited allocations.
# Use the repository path: Slurm executes a spooled copy of this batch script.
CONTINUATION_SCRIPT="$ARES_REPO_ROOT/run_ares_map_temporal.slurm"

if [ "$MODE" = "fresh" ]; then
    echo "=== Starting NEW $TARGET_EPOCHS-epoch run with auto-batch-size ==="
    TRAIN_COMMAND=(python -u scripts/carc/run_map_temporal.py --prepare-data)
else
    echo "=== Resuming $MODE toward $TARGET_EPOCHS total epochs ==="
    TRAIN_COMMAND=(python -u scripts/carc/resume_map_temporal.py --resume "$MODE")
fi

# Replacing the shell lets Slurm's B:USR1 warning reach Python's save handler.
exec "${TRAIN_COMMAND[@]}" \
    --epochs "$TARGET_EPOCHS" \
    --auto-batch-size \
    --gpu-memory-fraction 0.90 \
    --checkpoint-minutes 5 \
    --walltime-margin-minutes 10 \
    --submit-continuation "$CONTINUATION_SCRIPT"
