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

# Optional: pass a specific checkpoint/run path as $1, or "fresh" to force a new run
MODE="${1:-auto}"

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
python -u scripts/carc/run_map_temporal.py --prepare-data --check-only

# 4. Launch training (auto-detects whether to resume an existing run or start fresh)
RESULTS_BASE="$ARES_RUN_ROOT/results/sparse_reconstruction_map_temporal"

if [ "$MODE" = "fresh" ]; then
    echo "=== Starting NEW 1,000-epoch run with auto-batch-size ==="
    python -u scripts/carc/run_map_temporal.py \
        --prepare-data \
        --auto-batch-size \
        --gpu-memory-fraction 0.90 \
        --checkpoint-minutes 5 \
        --walltime-margin-minutes 10
elif [ "$MODE" != "auto" ]; then
    echo "=== Resuming SPECIFIC checkpoint/run: $MODE ==="
    python -u scripts/carc/resume_map_temporal.py \
        --resume "$MODE" \
        --auto-batch-size \
        --gpu-memory-fraction 0.90 \
        --checkpoint-minutes 5 \
        --walltime-margin-minutes 10
elif [ -d "$RESULTS_BASE" ] && [ -n "$(ls -A "$RESULTS_BASE" 2>/dev/null)" ]; then
    echo "=== Auto-resuming LATEST run in $RESULTS_BASE ==="
    python -u scripts/carc/resume_map_temporal.py \
        --auto-batch-size \
        --gpu-memory-fraction 0.90 \
        --checkpoint-minutes 5 \
        --walltime-margin-minutes 10
else
    echo "=== No existing run found; starting NEW 1,000-epoch run ==="
    python -u scripts/carc/run_map_temporal.py \
        --prepare-data \
        --auto-batch-size \
        --gpu-memory-fraction 0.90 \
        --checkpoint-minutes 5 \
        --walltime-margin-minutes 10
fi

echo "=== Job finished cleanly at $(date) ==="