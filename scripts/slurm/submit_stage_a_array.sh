#!/bin/bash
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --job-name=mto_sa
#SBATCH --array=0-4
#SBATCH --output=outputs/logs/slurm/stage_a_%A_%a.out
#SBATCH --error=outputs/logs/slurm/stage_a_%A_%a.err
set -euo pipefail
cd /data/home/scwc008/run/xxy/MTO
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090
SEED=$SLURM_ARRAY_TASK_ID
python scripts/train_stage.py --data-dir data/qm9s --stage stage_a --epochs 50 --seed $SEED --checkpoint-dir outputs/checkpoints/stage_a_seed${SEED}
