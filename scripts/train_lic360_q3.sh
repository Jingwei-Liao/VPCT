#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/train.py" \
    --dataset "$PROJECT_ROOT/data" \
    --train-split train \
    --test-split test \
    --model vpct-cheng2020-attn \
    --quality 3 \
    --epochs 100 \
    --batch-size 16 \
    --test-batch-size 1 \
    --num-workers 12 \
    --prefetch-factor 4 \
    --save-root "$PROJECT_ROOT/checkpoints/q3" \
    --vp-fov 90 90 \
    --num-viewports 6 \
    --random-vp-rotate \
    --cuda \
    --vpct-layers 1 \
    "$@"