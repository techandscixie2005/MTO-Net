#!/bin/bash
#SBATCH --job-name=mto-eval
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/eval_all_%j.out
#SBATCH --error=outputs/logs/eval_all_%j.err

cd /data/home/scwc008/run/xxy/MTO
source /usr/share/modules/init/sh
module load miniforge3/25.11.0-1
source activate py310-torch270-vllm090

echo "Host: $(hostname)"

for ckpt in outputs/checkpoints/full_stage_*_seed0_best.pt; do
  stage=$(basename $(dirname $ckpt))
  echo "Evaluating $stage"
  python scripts/eval_model.py \
    --checkpoint "$ckpt" \
    --data-dir data/qm9s/processed \
    --split test \
    --output-metrics "outputs/metrics/${stage}_seed0_test_metrics.json"
done
