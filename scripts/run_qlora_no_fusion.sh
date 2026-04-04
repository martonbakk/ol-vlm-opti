#!/usr/bin/env bash
# QLoRA fine-tune without vlm_opt kernel fusion; vision image sparsity on by default.
# Usage (from repo root): bash scripts/run_qlora_no_fusion.sh
# Override: MODEL_ID=... DATASET_ID=... bash scripts/run_qlora_no_fusion.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_ID="${DATASET_ID:-lmms-lab/ChartQA}"
SPLIT="${SPLIT:-test[:1%]}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qlora-no-fusion}"
CACHE_DIR="${CACHE_DIR:-./data}"

PYTHON_EXEC="${PYTHON_EXEC:-$(command -v python3 || command -v python)}"

exec "$PYTHON_EXEC" -m src.jobs.finetune_job \
  --model-id "$MODEL_ID" \
  --dataset-id "$DATASET_ID" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --batch-size 1 \
  --qlora \
  --gradient-checkpointing \
  --epochs 1.0
