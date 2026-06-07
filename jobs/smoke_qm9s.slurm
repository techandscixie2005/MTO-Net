#!/bin/bash
#SBATCH --job-name=mto-smoke-A
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=outputs/logs/smoke_stage_a_%j.out
#SBATCH --error=outputs/logs/smoke_stage_a_%j.err

export PYTHONUNBUFFERED=1
cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "Host: $(hostname)"
echo "Date: $(date)"

# Point at 500-mol subset (6.5MB, loads in <1s)
$PYTHON -u scripts/train_stage.py \
  --stage stage_a \
  --data-dir data/qm9s/subset_smoke \
  --epochs 5 --seed 0 --batch-size 8 \
  --feature-dim 32 --maxl 1 --num-block 1 \
  --checkpoint-dir outputs/checkpoints

echo "Done: $(date)"
ls -lh outputs/checkpoints/stage_a_seed0/
