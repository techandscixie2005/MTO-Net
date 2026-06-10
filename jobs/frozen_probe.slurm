#!/bin/bash
#SBATCH --job-name=mto_frozen
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/slurm/frozen_probe_%j.out
#SBATCH --error=outputs/logs/slurm/frozen_probe_%j.err

export PYTHONUNBUFFERED=1

echo "=== FROZEN PROBE START ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"

cd /data/home/scwc008/run/xxy/MTO || { echo "FAIL: cd to MTO"; exit 1; }

source /usr/share/modules/init/sh 2>/dev/null || true
module load miniforge3/25.11.0-1 2>/dev/null || true

PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

echo "Python: $($PYTHON --version)"
echo "DetaNet: $(ls third_party/DetaNet/detanet_model/ 2>/dev/null | head -3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'no-gpu-detected')"
echo "============================================"

$PYTHON -u scripts/eval/run_frozen_probe.py \
  --data-dir data/qm9s/subset_medium \
  --stage-a-ckpt outputs/checkpoints/stage_a_seed1/best.pt \
  --output-dir outputs/frozen_probe \
  --seeds 0 1 2 \
  --epochs 20 \
  --batch-size 8 \
  --feature-dim 128 \
  --maxl 3 \
  --num-block 3

echo "=== FROZEN PROBE DONE ==="
echo "Date: $(date)"
