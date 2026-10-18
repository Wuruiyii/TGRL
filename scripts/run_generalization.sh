#!/bin/bash
set -e
SCALES=("10x5" "30x15" "50x20")
for scale in "${SCALES[@]}"; do
    echo "=== Zero-shot on $scale ==="
    python evaluate.py --checkpoint ../checkpoints/pretrained/tgrl_15x10.pt \
        --instance-dir ../data/instances/${scale} \
        --profile D4 \
        --output ../experiments/generalization/zeroshot_${scale}.csv
done
