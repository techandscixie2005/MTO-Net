#!/bin/bash
#SBATCH --job-name=mto_transfer
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=outputs/logs/slurm/stage_transfer_%j.out
#SBATCH --error=outputs/logs/slurm/stage_transfer_%j.err

# ===========================================================================
# MTO-Net Stage Transfer Stability (Intra-Stage-A)
# ===========================================================================
# Since Stage B/C labels are only direct spectrum CSVs (no physical Hessian/
# frequencies/normal modes), we run intra-Stage-A transfer:
#   mu-only -> evaluate alpha
#   alpha-only -> evaluate mu
#   mu+alpha -> mu+alpha (baseline, re-uses existing Stage A results)
# ===========================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh 2>/dev/null
module load miniforge3/25.11.0-1 2>/dev/null

PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "============================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================"

$PYTHON -u scripts/eval/run_stage_transfer.py \
  --data-dir data/qm9s/subset_medium \
  --output-dir outputs/stage_transfer \
  --seeds 0 1 2 \
  --epochs 20 \
  --batch-size 8 \
  --feature-dim 128 \
  --maxl 3 \
  --num-block 3

echo "Done: $(date)"
