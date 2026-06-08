#!/bin/bash
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --job-name=mto_smoke
#SBATCH --output=outputs/logs/slurm/smoke_%j.out
#SBATCH --error=outputs/logs/slurm/smoke_%j.err
set -euo pipefail
cd /data/home/scwc008/run/xxy/MTO
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090
mkdir -p outputs/smoke outputs/reports outputs/figures/debug/smoke
python scripts/train_stage.py --data-dir data/qm9s --stage stage_a --epochs 2 --max-mols 32 --batch-size 4 --checkpoint-dir outputs/smoke/checkpoints --feature-dim 64 --maxl 1 --num-block 1
