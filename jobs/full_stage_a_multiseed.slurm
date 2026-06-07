#!/bin/bash
#SBATCH --job-name=mto-full-A
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/full_stage_a_%j.out
#SBATCH --error=outputs/logs/full_stage_a_%j.err

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090

echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"

for seed in 0 1 2; do
  echo "=== Full Stage A seed=$seed ==="
  python scripts/train_stage.py \
    --stage stage_a \
    --data-dir data/qm9s/processed \
    --epochs 50 --seed $seed --batch-size 16 \
    --checkpoint-dir outputs/checkpoints
done
