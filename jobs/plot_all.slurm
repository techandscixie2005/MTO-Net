#!/bin/bash
#SBATCH --job-name=mto-plot
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/plot_all_%j.out
#SBATCH --error=outputs/logs/plot_all_%j.err

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090

BEST_CKPT="outputs/checkpoints/stage_c_seed0/best.pt"

python scripts/plot_mto_maps.py --checkpoint "$BEST_CKPT" --num-mols 32 --top-slots 5
python scripts/plot_seed_subspace.py --stage stage_a --top-r 5
python scripts/plot_stage_transfer.py --mols selected
