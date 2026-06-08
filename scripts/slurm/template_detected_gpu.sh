#!/bin/bash
# Auto-detected GPU template for MTO-Net
# Detected: A800 partition (gpu_a800)
# Generated: 2026-06-08

#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/slurm/%x_%j.out
#SBATCH --error=outputs/logs/slurm/%x_%j.err

set -euo pipefail

cd /data/home/scwc008/run/xxy/MTO

module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090

echo "=== Job Info ==="
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")'
nvidia-smi || true

exec python scripts/train_stage.py "$@"
