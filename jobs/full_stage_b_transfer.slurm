#!/bin/bash
#SBATCH --job-name=mto-full-B
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/full_stage_b_%j.out
#SBATCH --error=outputs/logs/full_stage_b_%j.err

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1

echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"
echo "Date: $(date)"

echo "=== Full Stage B (transfer from Stage A seed 0) ==="
conda run -n py310-torch270-vllm090 python scripts/train_stage.py \
  --stage stage_b \
  --data-dir data/qm9s \
  --epochs 100 --seed 0 --batch-size 8 \
  --checkpoint-dir outputs/checkpoints

echo "Done: $(date)"
