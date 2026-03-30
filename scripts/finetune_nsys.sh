#!/bin/bash
# Run fine-tune with nsys profiling. Only the training loop (between cudaProfilerStart/Stop)
# is captured; model/dataset loading is excluded from the profile.
#
# Usage: ./scripts/finetune_nsys.sh [OPTIONS]
# Example: ./scripts/finetune_nsys.sh --model-id Qwen/Qwen3-VL-8B-Instruct --dataset-id lmms-lab/ChartQA \
#   --output-dir ./checkpoints/notebook-finetune --nsys-out ./reports/finetune_profile

set -e

run_and_report() {
    local cmd_desc="$1"
    shift

    local log_file
    log_file=$(mktemp 2>/dev/null || mktemp -t finetune_nsys.log)

    set +e
    "$@" 2>&1 | tee "$log_file"
    local cmd_status=${PIPESTATUS[0]}
    set -e

    if [[ $cmd_status -ne 0 ]]; then
        echo "[!] ${cmd_desc} failed with exit code ${cmd_status}"
        if [[ $cmd_status -eq 139 ]]; then
            echo "[!] Exit code 139 usually means a segmentation fault (SIGSEGV)."
        fi
        echo "[!] Last 40 log lines:"
        tail -n 40 "$log_file"
    fi

    rm -f "$log_file"
    return "$cmd_status"
}

# Defaults (override with env or args)
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_ID="${DATASET_ID:-lmms-lab/ChartQA}"
SPLIT="${SPLIT:-test[:85%]}"
EPOCHS="${EPOCHS:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VISION_MAX_PIXELS="${VISION_MAX_PIXELS:-262144}"
VISION_MIN_PIXELS="${VISION_MIN_PIXELS:-50176}"
IMAGE_MAX_SIDE="${IMAGE_MAX_SIDE:-768}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/notebook-finetune}"
CACHE_DIR="${CACHE_DIR:-./data}"
NSYS_OUT="${NSYS_OUT:-./reports/finetune_profile}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --dataset-id) DATASET_ID="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --vision-max-pixels) VISION_MAX_PIXELS="$2"; shift 2 ;;
        --vision-min-pixels) VISION_MIN_PIXELS="$2"; shift 2 ;;
        --image-max-side) IMAGE_MAX_SIDE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --nsys-out) NSYS_OUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$NSYS_OUT")"
mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")/.." || exit 1
PYTHON_EXEC=$(uv run which python 2>/dev/null || which python)

if ! command -v nsys &> /dev/null; then
    echo "nsys not found. Run fine-tune without profiling."
    run_and_report "Fine-tune" "$PYTHON_EXEC" -m src.jobs.finetune_job \
        --model-id "$MODEL_ID" \
        --dataset-id "$DATASET_ID" \
        --split "$SPLIT" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --vision-max-pixels "$VISION_MAX_PIXELS" \
        --vision-min-pixels "$VISION_MIN_PIXELS" \
        --image-max-side "$IMAGE_MAX_SIDE" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR"
    exit $?
fi

echo "[+] nsys profiling (training only, --capture-range=cudaProfilerApi)"
echo "[+] Report: ${NSYS_OUT}.nsys-rep"
echo "---------------------------------------------------------"

# osrt (OS runtime) is not available in Windows nsys --trace; use cuda,nvtx (optional: cublas,cuDNN).
run_and_report "nsys profile + fine-tune" nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o "$NSYS_OUT" \
    "$PYTHON_EXEC" -m src.jobs.finetune_job \
        --model-id "$MODEL_ID" \
        --dataset-id "$DATASET_ID" \
        --split "$SPLIT" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --vision-max-pixels "$VISION_MAX_PIXELS" \
        --vision-min-pixels "$VISION_MIN_PIXELS" \
        --image-max-side "$IMAGE_MAX_SIDE" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR"
    exit $?
