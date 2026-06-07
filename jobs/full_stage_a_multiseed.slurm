#!/bin/bash
#SBATCH --job-name=mto-full-A
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/full_stage_a_%j.out
#SBATCH --error=outputs/logs/full_stage_a_%j.err

export PYTHONUNBUFFERED=1
cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1

PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"
echo "Date: $(date)"

for seed in 0 1 2; do
  echo "=== Full Stage A seed=$seed ==="
  $PYTHON -u scripts/train_stage.py --stage stage_a --data-dir data/qm9s --epochs 50 --seed $seed --batch-size 8 --checkpoint-dir outputs/checkpoints
  echo "=== Seed $seed done at $(date) ==="
  ls -lh outputs/checkpoints/
done

echo "Done: $(date)"
