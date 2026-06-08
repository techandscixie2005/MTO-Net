#!/bin/bash
#SBATCH --job-name=mto-med-A
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/medium_stage_a_%j.out
#SBATCH --error=outputs/logs/medium_stage_a_%j.err

export PYTHONUNBUFFERED=1
cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "Host: $(hostname)"
echo "Date: $(date)"

for seed in 0 1 2; do
  echo "=== Medium Stage A seed=$seed ==="
  $PYTHON -u scripts/train_stage.py --stage stage_a --data-dir data/qm9s/subset_medium --epochs 20 --seed $seed --batch-size 8 --feature-dim 128 --maxl 3 --num-block 3 --checkpoint-dir outputs/checkpoints
  echo "=== Seed $seed done at $(date) ==="
done

echo "All done: $(date)"
ls -lh outputs/checkpoints/ 2>/dev/null
