"""Single entry: enable/disable VLM backend optimizations from config.

QLoRA / PEFT and ``VramTrainConfig`` live in the separate ``vlm_lora`` package (see ``peft_qlora.py``);
``finetune_job`` wires that path. Kernel fusion stays in ``VlmOptConfig`` here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.vlm_opt.config import VlmOptConfig
from src.vlm_opt.io import vlm_print
from src.vlm_opt.liger_optional import try_apply_liger_to_qwen3_language_model
from src.vlm_opt.patch_qwen3vl import patch_qwen3vl_vision_mergers

logger = logging.getLogger(__name__)


def apply_vlm_optimizations(model: nn.Module, cfg: VlmOptConfig) -> nn.Module:
    """
    Apply configured optimizations to a loaded model (in-place where possible).

    Order:
        1. Optional Liger patches (global / text backbone) when enabled.
        2. Fused vision PatchMerger blocks when enabled.
        3. Optional ``torch.compile`` on the full module (experimental).
    """
    if not cfg.enabled:
        vlm_print("apply_vlm_optimizations: cfg.enabled=False - nothing to do.")
        return model

    vlm_print(
        "apply_vlm_optimizations: START "
        f"(merger_backend={cfg.merger_fused_backend!s}, "
        f"liger={cfg.liger_language_model}, "
        f"torch_compile_full_model={cfg.torch_compile_full_model})"
    )
    vlm_print(
        "NOTE: Fused PatchMerger mainly cuts kernel/mem traffic & launch overhead; "
        "it does NOT remove weights - peak VRAM often changes little. "
        "For lower VRAM use: smaller --batch-size, --gradient-checkpointing, "
        "--no-bf16 off (keep bf16), lower --vision-max-pixels / --image-max-side, QLoRA, smaller model."
    )

    if cfg.liger_language_model:
        vlm_print("Step 1/3: Liger language-model patches ...")
        ok = try_apply_liger_to_qwen3_language_model(model)
        vlm_print(f"Step 1/3: Liger -> {'ACTIVE' if ok else 'NOT ACTIVE (skip or failed)'}")

    if cfg.fused_vision_merger:
        vlm_print("Step 2/3: Vision PatchMerger fusion ...")
        n = patch_qwen3vl_vision_mergers(model, cfg.merger_fused_backend)
        vlm_print(f"Step 2/3: fused {n} PatchMerger module(s) (see per-layer backend below).")

    if cfg.torch_compile_full_model and hasattr(torch, "compile"):
        mode = cfg.torch_compile_mode
        vlm_print(f"Step 3/3: torch.compile full model (mode={mode}) ...")
        model = torch.compile(model, mode=mode, fullgraph=False)  # type: ignore[assignment]
        vlm_print(f"Step 3/3: torch.compile -> ACTIVE (mode={mode})")
    elif cfg.torch_compile_full_model:
        vlm_print("Step 3/3: torch.compile skipped (torch.compile not available).")

    vlm_print("apply_vlm_optimizations: DONE.")
    return model
