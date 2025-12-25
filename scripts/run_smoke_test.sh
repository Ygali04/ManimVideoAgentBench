#!/bin/bash
# Optimized smoke test script for TheoremExplainAgent
# Uses cheap/fast models to quickly verify the pipeline works

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${1:-smoke}"
RUN_ROOT="output/runs/${TIMESTAMP}_${RUN_NAME}"

# Fast/cheap models for smoke testing
GEN_MODEL="${GEN_MODEL:-gpt4o_mini}"              # $0.15/M - fast, reliable for structured output
EVAL_TEXT_MODEL="${EVAL_TEXT_MODEL:-gemini20_flash_lite}"   # $0.07/M - cheap
EVAL_IMAGE_MODEL="${EVAL_IMAGE_MODEL:-gemini20_flash_lite}" # $0.07/M - cheap, has vision
EVAL_VIDEO_MODEL="${EVAL_VIDEO_MODEL:-gemini20_flash_lite}" # $0.07/M - cheap, has video

# Smoke test settings - minimal for speed
MAX_RETRIES="${MAX_RETRIES:-0}"  # N=0 for fastest test (no fix loop)
SAMPLE_SIZE="${SAMPLE_SIZE:-1}"  # Just 1 theorem
THEOREMS_PATH="${THEOREMS_PATH:-data/thb_easy/math.json}"
TARGET_FPS="${TARGET_FPS:-1}"    # Minimal frames for video eval

echo "============================================"
echo "TheoremExplainAgent Smoke Test"
echo "============================================"
echo "Run root: $RUN_ROOT"
echo "Gen model: $GEN_MODEL"
echo "Eval models: text=$EVAL_TEXT_MODEL, image=$EVAL_IMAGE_MODEL, video=$EVAL_VIDEO_MODEL"
echo "Max retries (N): $MAX_RETRIES"
echo "Sample size: $SAMPLE_SIZE"
echo "Theorems: $THEOREMS_PATH"
echo "============================================"

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Ensure PYTHONPATH includes current directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Create output directories
GEN_DIR="$RUN_ROOT/gen/$GEN_MODEL/N=$MAX_RETRIES"
mkdir -p "$GEN_DIR"

echo ""
echo "[1/3] Generation Phase"
echo "----------------------"
START_GEN=$(date +%s)

python generate_video.py \
    --model "$GEN_MODEL" \
    --theorems_path "$THEOREMS_PATH" \
    --sample_size "$SAMPLE_SIZE" \
    --max_retries "$MAX_RETRIES" \
    --output_dir "$GEN_DIR" \
    --max_scene_concurrency 2 \
    --max_topic_concurrency 1

END_GEN=$(date +%s)
GEN_TIME=$((END_GEN - START_GEN))
echo "Generation completed in ${GEN_TIME}s"

# Find the generated theorem directory
THEOREM_DIR=$(find "$GEN_DIR" -name "*_combined.mp4" -exec dirname {} \; | head -1)

if [ -z "$THEOREM_DIR" ]; then
    echo "ERROR: No combined video found. Generation may have failed."
    echo "Checking for partial outputs..."
    find "$GEN_DIR" -name "*.mp4" | head -5
    exit 1
fi

echo "Found theorem at: $THEOREM_DIR"

echo ""
echo "[2/3] Evaluation Phase"
echo "----------------------"
EVAL_DIR="$RUN_ROOT/eval/judges=${EVAL_TEXT_MODEL}__${EVAL_VIDEO_MODEL}"
mkdir -p "$EVAL_DIR"

START_EVAL=$(date +%s)

# Run evaluation with fast settings
python evaluate.py \
    --bulk_evaluate \
    --eval_type all \
    --file_path "$(dirname "$THEOREM_DIR")" \
    --output_folder "$EVAL_DIR" \
    --model_text "$EVAL_TEXT_MODEL" \
    --model_image "$EVAL_IMAGE_MODEL" \
    --model_video "$EVAL_VIDEO_MODEL" \
    --target_fps "$TARGET_FPS" \
    --retry_limit 2 \
    --max_workers 2

END_EVAL=$(date +%s)
EVAL_TIME=$((END_EVAL - START_EVAL))
echo "Evaluation completed in ${EVAL_TIME}s"

echo ""
echo "[3/3] Summary"
echo "-------------"
TOTAL_TIME=$((GEN_TIME + EVAL_TIME))
echo "Generation time: ${GEN_TIME}s"
echo "Evaluation time: ${EVAL_TIME}s"
echo "Total time: ${TOTAL_TIME}s"
echo ""
echo "Outputs:"
echo "  Generation: $GEN_DIR"
echo "  Evaluation: $EVAL_DIR"
echo ""

# Show generated video
echo "Generated video:"
ls -lh "$THEOREM_DIR"/*_combined.mp4 2>/dev/null || echo "  (none found)"

# Show evaluation results
echo ""
echo "Evaluation results:"
find "$EVAL_DIR" -name "*.json" -exec echo "  {}" \; -exec cat {} \; 2>/dev/null | head -50 || echo "  (none found)"

echo ""
echo "============================================"
echo "Smoke test completed successfully!"
echo "============================================"

