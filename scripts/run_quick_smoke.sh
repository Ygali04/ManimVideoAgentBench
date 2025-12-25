#!/bin/bash
# Ultra-quick smoke test - generation only, no eval
# Use this to quickly verify video generation works

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="output/runs/${TIMESTAMP}_quick"
GEN_MODEL="${GEN_MODEL:-gpt4o_mini}"

echo "Quick smoke test (generation only)"
echo "Model: $GEN_MODEL"
echo "Output: $RUN_ROOT"

# Activate venv
[ -f ".venv/bin/activate" ] && source .venv/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

GEN_DIR="$RUN_ROOT/gen/$GEN_MODEL/N=0"
mkdir -p "$GEN_DIR"

START=$(date +%s)

python generate_video.py \
    --model "$GEN_MODEL" \
    --theorems_path data/thb_easy/math.json \
    --sample_size 1 \
    --max_retries 0 \
    --output_dir "$GEN_DIR" \
    --max_scene_concurrency 3

END=$(date +%s)
echo ""
echo "Completed in $((END - START))s"
echo "Output: $GEN_DIR"

# Show result
find "$GEN_DIR" -name "*_combined.mp4" -exec ls -lh {} \;

