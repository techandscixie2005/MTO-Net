#!/bin/bash
#SBATCH --job-name=mto-csm
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm/smoke_stage_c_%j.out
#SBATCH --error=outputs/logs/slurm/smoke_stage_c_%j.err

export PYTHONUNBUFFERED=1
cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "Stage C Smoke Test"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"
echo "Date: $(date)"

$PYTHON -u scripts/train_stage.py \
  --data-dir data/qm9s \
  --stage stage_c \
  --epochs 2 \
  --seed 101 \
  --batch-size 4 \
  --max-mols 16 \
  --feature-dim 64 \
  --maxl 2 \
  --num-block 2 \
  --rc 5.0 \
  --lr 1e-3 \
  --activity-mode simple \
  --spectral-downsample 50 \
  --save-cache \
  --plot-mto \
  --checkpoint-dir outputs/smoke_stage_c/checkpoints

echo "Done at $(date)"
