#!/bin/bash
#SBATCH --job-name=mto_baseline
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm/baseline_%j.out
#SBATCH --error=outputs/logs/slurm/baseline_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh 2>/dev/null
module load miniforge3/25.11.0-1 2>/dev/null

PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Commit: $(git log --oneline -1)"

$PYTHON -u scripts/eval/run_baselines.py \
  --data-dir data/qm9s/subset_smoke \
  --epochs 10 --batch-size 8 --feature-dim 64 \
  --checkpoint-dir outputs/baselines \
  --metrics-dir outputs/metrics/baselines

echo "Done: $(date)"
