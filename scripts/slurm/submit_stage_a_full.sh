#!/bin/bash
#SBATCH --job-name=mto-sa-full
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --output=outputs/logs/slurm/stage_a_full_%A_%a.out
#SBATCH --error=outputs/logs/slurm/stage_a_full_%A_%a.err

export PYTHONUNBUFFERED=1
cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
PYTHON=/data/apps/miniforge3/25.11.0-1/envs/py310-torch270-vllm090/bin/python

SEED=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "MTO-Net Stage A Full QM9S - Seed ${SEED}"
echo "============================================"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"
echo "Date: $(date)"
echo ""

$PYTHON -u scripts/train_stage.py \
  --data-dir data/qm9s \
  --stage stage_a \
  --epochs 50 \
  --seed $SEED \
  --batch-size 8 \
  --feature-dim 128 \
  --maxl 3 \
  --num-block 3 \
  --rc 5.0 \
  --lr 1e-3 \
  --activity-mode simple \
  --split-file outputs/splits/qm9s_split_stage_a.json \
  --checkpoint-dir outputs/checkpoints \
  --save-cache \
  --plot-mto

echo ""
echo "Stage A seed ${SEED} done at $(date)"
