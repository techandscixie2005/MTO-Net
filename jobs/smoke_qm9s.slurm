#!/bin/bash
#SBATCH --job-name=mto-smoke
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/smoke_%j.out
#SBATCH --error=outputs/logs/smoke_%j.err

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"

python scripts/run_all_smoke.py
