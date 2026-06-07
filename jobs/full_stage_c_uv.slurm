#!/bin/bash
#SBATCH --job-name=mto-full-C
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/full_stage_c_%j.out
#SBATCH --error=outputs/logs/full_stage_c_%j.err

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090

echo "Host: $(hostname)"
echo "Full Stage C (transfer from best Stage B)"

python scripts/train_stage.py \
  --stage stage_c \
  --data-dir data/qm9s/processed \
  --epochs 100 --seed 0 --batch-size 16 \
  --checkpoint-dir outputs/checkpoints
