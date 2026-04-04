#!/bin/bash
# Run fine-tune with nsys profiling. Only the training loop (between cudaProfilerStart/Stop)
# is captured; model/dataset loading is excluded from the profile.
#
# Usage: ./scripts/finetune_nsys.sh [OPTIONS]
#
# QLoRA: Python defaults to NO vision sparsity (full tokens; better for QLoRA). Enable explicitly:
#   ./scripts/finetune_nsys.sh --qlora --sparse
#   → default nsys report path: ./reports/qlora_sparsity_profile.nsys-rep (override with --nsys-out or NSYS_OUT)
#
# QLoRA without vision sparsity (same as plain --qlora now):
#   ./scripts/finetune_nsys.sh --qlora
#   → default report: ./reports/finetune_profile.nsys-rep
#
# Aliases:
#   --sparse     ->  --token-sparsity   (enable vision sparsity)
#   --no-sparse  ->  --no-token-sparsity
#
# Env overrides for report basename (no extension; nsys adds .nsys-rep):
#   NSYS_OUT_QLORA_SPARSE  default ./reports/qlora_sparsity_profile
#   NSYS_OUT_DEFAULT       default ./reports/finetune_profile
#   NSYS_OUT               if set, wins unless you pass --nsys-out
#
# Example:
#   ./scripts/finetune_nsys.sh --qlora --sparse --sparsity-keep-ratio 0.25

set -e

# Defaults (override with env or args)
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_ID="${DATASET_ID:-lmms-lab/ChartQA}"
SPLIT="${SPLIT:-test[:85%]}"
EPOCHS="${EPOCHS:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/notebook-finetune}"
CACHE_DIR="${CACHE_DIR:-./data}"

NSYS_OUT_DEFAULT="${NSYS_OUT_DEFAULT:-./reports/finetune_profile}"
NSYS_OUT_QLORA_SPARSE="${NSYS_OUT_QLORA_SPARSE:-./reports/qlora_sparsity_profile}"
NSYS_OUT_FROM_ENV="${NSYS_OUT:-}"
EXPLICIT_NSYS_OUT=false

# Extra args for Python (e.g. --vlm-opt --vlm-opt-merger-backend compile_wrap)
VLM_JOB_EXTRA=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --dataset-id) DATASET_ID="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --nsys-out) NSYS_OUT="$2"; EXPLICIT_NSYS_OUT=true; shift 2 ;;
        --vlm-opt) VLM_JOB_EXTRA+=(--vlm-opt); shift ;;
        --vlm-opt-merger-backend) VLM_JOB_EXTRA+=(--vlm-opt-merger-backend "$2"); shift 2 ;;
        --vlm-opt-liger) VLM_JOB_EXTRA+=(--vlm-opt-liger); shift ;;
        --vlm-opt-torch-compile-model) VLM_JOB_EXTRA+=(--vlm-opt-torch-compile-model); shift ;;
        --gradient-checkpointing) VLM_JOB_EXTRA+=(--gradient-checkpointing); shift ;;
        --dataloader-num-workers) VLM_JOB_EXTRA+=(--dataloader-num-workers "$2"); shift 2 ;;
        --gradient-accumulation-steps) VLM_JOB_EXTRA+=(--gradient-accumulation-steps "$2"); shift 2 ;;
        --qlora) VLM_JOB_EXTRA+=(--qlora); shift ;;
        --qlora-8bit) VLM_JOB_EXTRA+=(--qlora-8bit); shift ;;
        --lora-r) VLM_JOB_EXTRA+=(--lora-r "$2"); shift 2 ;;
        --lora-alpha) VLM_JOB_EXTRA+=(--lora-alpha "$2"); shift 2 ;;
        --lora-dropout) VLM_JOB_EXTRA+=(--lora-dropout "$2"); shift 2 ;;
        --lora-target-modules) VLM_JOB_EXTRA+=(--lora-target-modules "$2"); shift 2 ;;
        --token-sparsity) VLM_JOB_EXTRA+=(--token-sparsity); shift ;;
        --no-token-sparsity) VLM_JOB_EXTRA+=(--no-token-sparsity); shift ;;
        --sparse) VLM_JOB_EXTRA+=(--token-sparsity); shift ;;
        --no-sparse) VLM_JOB_EXTRA+=(--no-token-sparsity); shift ;;
        --sparsity-keep-ratio|--keep-ratio) VLM_JOB_EXTRA+=(--sparsity-keep-ratio "$2"); shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$EXPLICIT_NSYS_OUT" == true ]]; then
    :
else
    has_qlora=false
    want_sparse=false
    for a in "${VLM_JOB_EXTRA[@]}"; do
        [[ "$a" == "--qlora" ]] && has_qlora=true
        if [[ "$a" == "--token-sparsity" || "$a" == "--sparse" ]]; then
            want_sparse=true
        fi
    done
    if [[ -n "$NSYS_OUT_FROM_ENV" ]]; then
        NSYS_OUT="$NSYS_OUT_FROM_ENV"
    elif $has_qlora && $want_sparse; then
        NSYS_OUT="$NSYS_OUT_QLORA_SPARSE"
    else
        NSYS_OUT="$NSYS_OUT_DEFAULT"
    fi
fi

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
        --cache-dir "$CACHE_DIR" \
        "${VLM_JOB_EXTRA[@]}"
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
        --cache-dir "$CACHE_DIR" \
        "${VLM_JOB_EXTRA[@]}"
