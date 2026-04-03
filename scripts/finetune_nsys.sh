#!/bin/bash
# Run fine-tune with nsys profiling. Only the training loop (between cudaProfilerStart/Stop)
# is captured; model/dataset loading is excluded from the profile.
#
# Usage: ./scripts/finetune_nsys.sh [OPTIONS]
# Example: ./scripts/finetune_nsys.sh --model-id Qwen/Qwen3-VL-8B-Instruct --dataset-id lmms-lab/ChartQA \
#   --output-dir ./checkpoints/notebook-finetune --nsys-out ./reports/finetune_profile

set -e

# Defaults (override with env or args)
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_ID="${DATASET_ID:-lmms-lab/ChartQA}"
SPLIT="${SPLIT:-test[:85%]}"
EPOCHS="${EPOCHS:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/notebook-finetune}"
CACHE_DIR="${CACHE_DIR:-./data}"
NSYS_OUT="${NSYS_OUT:-./reports/finetune_profile}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --dataset-id) DATASET_ID="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
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
    exec "$PYTHON_EXEC" -m src.jobs.finetune_job \
        --model-id "$MODEL_ID" \
        --dataset-id "$DATASET_ID" \
        --split "$SPLIT" \
        --epochs "$EPOCHS" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR"
fi

echo "[+] nsys profiling (training only, --capture-range=cudaProfilerApi)"
echo "[+] Report: ${NSYS_OUT}.nsys-rep"
echo "---------------------------------------------------------"

# osrt (OS runtime) is not available in Windows nsys --trace; use cuda,nvtx (optional: cublas,cuDNN).
exec nsys profile \
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
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR"
