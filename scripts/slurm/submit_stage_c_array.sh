#!/bin/bash
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --job-name=mto_sc
#SBATCH --array=0-4
#SBATCH --output=outputs/logs/slurm/stage_c_%A_%a.out
#SBATCH --error=outputs/logs/slurm/stage_c_%A_%a.err
set -euo pipefail
cd /data/home/scwc008/run/xxy/MTO
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090
echo "Stage C requires UV labels not present in QM9S. Skipping."
