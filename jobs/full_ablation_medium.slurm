#!/bin/bash
#SBATCH --job-name=mto_ablation
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/slurm/ablation_med_%j.out
#SBATCH --error=outputs/logs/slurm/ablation_med_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh 2>/dev/null
module load miniforge3/25.11.0-1 2>/dev/null

PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "============================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Commit: $(git log --oneline -1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================"

# Medium-scale config (matching Stage A architecture but medium data)
DATA_DIR="data/qm9s/subset_medium"
CKPT_BASE="outputs/baselines"
METRICS_DIR="outputs/metrics/baselines"
EPOCHS=20
BATCH_SIZE=8
FEATURE_DIM=128
MAXL=3
NUM_BLOCK=3
SEEDS=(0 1 2)

mkdir -p "$CKPT_BASE" "$METRICS_DIR" outputs/logs/slurm

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "========================================"
  echo "  ABLATION RUN — seed=$seed"
  echo "  Date: $(date)"
  echo "========================================"

  $PYTHON -u scripts/eval/run_baselines.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "${CKPT_BASE}/seed${seed}" \
    --metrics-dir "${METRICS_DIR}" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --feature-dim $FEATURE_DIM \
    --maxl $MAXL \
    --num-block $NUM_BLOCK \
    --lr 1e-3

  echo "  Seed $seed done at $(date)"
  echo ""
done

echo "============================================"
echo "  ALL ABLATIONS COMPLETE"
echo "  Date: $(date)"
echo "============================================"
