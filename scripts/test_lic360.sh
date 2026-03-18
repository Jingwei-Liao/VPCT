#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"


TEST_SPLIT="8k_test"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
NUM_VIEWPORTS="${NUM_VIEWPORTS:-4}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

quality=1
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/q${quality}/vpct-cheng2020-attn_q${quality}_vpct1_bs16_nvp6_fov90x90_lr0p0001/checkpoint_best_loss.pth.tar"
python "$PROJECT_ROOT/test.py" \
    -d "$PROJECT_ROOT/data" \
    --test-split "$TEST_SPLIT" \
    --model vpct-cheng2020-attn \
    --quality "$quality" \
    --vpct-layers 1 \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --vp-fov 90 90 \
    --num-viewports "$NUM_VIEWPORTS" \
    --checkpoint "$CHECKPOINT_PATH" \
    --cuda \
    --amp \
    --amp-dtype "$AMP_DTYPE" \
    "$@"

quality=2
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/q${quality}/vpct-cheng2020-attn_q${quality}_vpct1_bs16_nvp6_fov90x90_lr0p0001/checkpoint_best_loss.pth.tar"
python "$PROJECT_ROOT/test.py" \
    -d "$PROJECT_ROOT/data" \
    --test-split "$TEST_SPLIT" \
    --model vpct-cheng2020-attn \
    --quality "$quality" \
    --vpct-layers 1 \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --vp-fov 90 90 \
    --num-viewports "$NUM_VIEWPORTS" \
    --checkpoint "$CHECKPOINT_PATH" \
    --cuda \
    --amp \
    --amp-dtype "$AMP_DTYPE" \
    "$@"

quality=3
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/q${quality}/vpct-cheng2020-attn_q${quality}_vpct1_bs16_nvp6_fov90x90_lr0p0001/checkpoint_best_loss.pth.tar"
python "$PROJECT_ROOT/test.py" \
    -d "$PROJECT_ROOT/data" \
    --test-split "$TEST_SPLIT" \
    --model vpct-cheng2020-attn \
    --quality "$quality" \
    --vpct-layers 1 \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --vp-fov 90 90 \
    --num-viewports "$NUM_VIEWPORTS" \
    --checkpoint "$CHECKPOINT_PATH" \
    --cuda \
    --amp \
    --amp-dtype "$AMP_DTYPE" \
    "$@"

quality=4
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/q${quality}/vpct-cheng2020-attn_q${quality}_vpct1_bs16_nvp6_fov90x90_lr0p0001/checkpoint_best_loss.pth.tar"
python "$PROJECT_ROOT/test.py" \
    -d "$PROJECT_ROOT/data" \
    --test-split "$TEST_SPLIT" \
    --model vpct-cheng2020-attn \
    --quality "$quality" \
    --vpct-layers 1 \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --vp-fov 90 90 \
    --num-viewports "$NUM_VIEWPORTS" \
    --checkpoint "$CHECKPOINT_PATH" \
    --cuda \
    --amp \
    --amp-dtype "$AMP_DTYPE" \
    "$@"

quality=5
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/q${quality}/vpct-cheng2020-attn_q${quality}_vpct1_bs16_nvp6_fov90x90_lr0p0001/checkpoint_best_loss.pth.tar"
python "$PROJECT_ROOT/test.py" \
    -d "$PROJECT_ROOT/data" \
    --test-split "$TEST_SPLIT" \
    --model vpct-cheng2020-attn \
    --quality "$quality" \
    --vpct-layers 1 \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --vp-fov 90 90 \
    --num-viewports "$NUM_VIEWPORTS" \
    --checkpoint "$CHECKPOINT_PATH" \
    --cuda \
    --amp \
    --amp-dtype "$AMP_DTYPE" \
    "$@"

quality=6
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/q${quality}/vpct-cheng2020-attn_q${quality}_vpct1_bs16_nvp6_fov90x90_lr0p0001/checkpoint_best_loss.pth.tar"
python "$PROJECT_ROOT/test.py" \
    -d "$PROJECT_ROOT/data" \
    --test-split "$TEST_SPLIT" \
    --model vpct-cheng2020-attn \
    --quality "$quality" \
    --vpct-layers 1 \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --vp-fov 90 90 \
    --num-viewports "$NUM_VIEWPORTS" \
    --checkpoint "$CHECKPOINT_PATH" \
    --cuda \
    --amp \
    --amp-dtype "$AMP_DTYPE" \
    "$@"
