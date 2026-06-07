#!/bin/bash
#SBATCH --job-name=mto-smoke-A
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/smoke_stage_a_%j.out
#SBATCH --error=outputs/logs/smoke_stage_a_%j.err

export PYTHONUNBUFFERED=1
cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1

PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"
echo "Date: $(date)"

# Smoke test: 500 molecules, small model, 5 epochs
$PYTHON -u scripts/train_stage.py \
  --stage stage_a \
  --data-dir data/qm9s \
  --epochs 5 --seed 0 --batch-size 8 \
  --feature-dim 32 --maxl 1 --num-block 1 \
  --checkpoint-dir outputs/checkpoints

echo "Done: $(date)"
ls -lh outputs/checkpoints/
