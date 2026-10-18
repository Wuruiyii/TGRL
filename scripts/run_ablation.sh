#!/bin/bash
set -e
CONFIGS=(A B C D E F G H)
for cfg in "${CONFIGS[@]}"; do
    echo "=== Ablation config $cfg ==="
    python train.py --config ../src/configs/ablation_${cfg}.yaml \
        --instance-dir ../data/instances/15x10 \
        --checkpoint-dir ../checkpoints/ablation_${cfg} \
        --seed 0
    python evaluate.py --checkpoint ../checkpoints/ablation_${cfg}/best.pt \
        --instance-dir ../data/instances/15x10 \
        --profile D4 \
        --output ../experiments/ablation/results_${cfg}.csv
done
echo "Full TGRL"
python evaluate.py --checkpoint ../checkpoints/pretrained/tgrl_15x10.pt \
    --instance-dir ../data/instances/15x10 \
    --profile D4 \
    --output ../experiments/ablation/results_I.csv
