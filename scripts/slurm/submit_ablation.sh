#!/bin/bash
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --job-name=mto_abl
#SBATCH --output=outputs/logs/slurm/ablation_%j.out
#SBATCH --error=outputs/logs/slurm/ablation_%j.err
set -euo pipefail
cd /data/home/scwc008/run/xxy/MTO
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090
python scripts/train_stage.py --data-dir data/qm9s --stage stage_a --epochs 50 --seed 0 --checkpoint-dir outputs/checkpoints/ablation_direct --ablation direct_readout
