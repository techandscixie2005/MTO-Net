#!/bin/bash
#SBATCH --job-name=mto-med-B
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/medium_stage_b_%j.out
#SBATCH --error=outputs/logs/medium_stage_b_%j.err

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090

echo "Host: $(hostname)"
echo "Stage B (transfer from Stage A seed 0)"

python scripts/train_stage.py \
  --stage stage_b \
  --data-dir data/qm9s/subset_medium/processed \
  --epochs 100 --seed 0 \
  --checkpoint-dir outputs/checkpoints
