"""BitsAndBytes load config + PEFT QLoRA wrapping (no kernel fusion here)."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.vlm_lora.config import VramTrainConfig
from src.vlm_lora.io import lora_print

logger = logging.getLogger(__name__)

# Typical Qwen LLM projection layer leaf names (language_model.layers.*)
_DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Qwen3-VL: ``visual.merger`` and ``visual.deepstack_merger_list.*`` (Qwen3VLVisionPatchMerger) use
# these linears to map vision features into the LLM hidden size. Training only the LM without them
# leaves the image-to-text bridge frozen.
_PROJECTOR_LINEAR: tuple[str, ...] = (
    "linear_fc1",
    "linear_fc2",
)

# ``visual.blocks.*`` MLPs also use linear_fc1/linear_fc2; we train merger/deepstack only by excluding
# blocks. Vision attention uses ``qkv``/``proj`` (not q_proj), so LLM-only targets do not hit blocks.
_EXCLUDE_VISION_BLOCKS_REGEX: str = r".*\.visual\.blocks(\..*)?$"


def _default_qlora_target_candidates() -> tuple[str, ...]:
    return _DEFAULT_LORA_TARGETS + _PROJECTOR_LINEAR


def _language_model_root(model: nn.Module) -> nn.Module | None:
    """Return ``language_model`` for Qwen-style VLMs (ForConditionalGeneration → .model.language_model)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return getattr(inner, "language_model")
    if hasattr(model, "language_model"):
        return getattr(model, "language_model")
    return None


def _has_visual_tower(model: nn.Module) -> bool:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        return True
    return hasattr(model, "visual")


def _collect_visual_projector_leaves(model: nn.Module, want: set[str]) -> set[str]:
    """``linear_fc*`` under ``visual.merger`` and ``visual.deepstack_merger_list`` (not ``visual.blocks``)."""
    found: set[str] = set()
    inner = getattr(model, "model", None)
    visual = getattr(inner, "visual", None) if inner is not None else None
    if visual is None:
        visual = getattr(model, "visual", None)
    if visual is None:
        return found
    for name, _mod in visual.named_modules():
        if ".blocks." in name:
            continue
        if ".merger." not in name and "deepstack_merger_list" not in name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in want:
            found.add(leaf)
    return found


def resolve_lora_target_modules(
    model: nn.Module,
    candidates: tuple[str, ...] | None = None,
    *,
    language_only: bool = True,
) -> list[str]:
    """Detect leaf names for LoRA: ``language_model`` projections plus vision projector linears when present."""
    cand = candidates or _default_qlora_target_candidates()
    want = set(cand)
    root: nn.Module = model
    if language_only:
        lm = _language_model_root(model)
        if lm is not None:
            root = lm
    found: set[str] = set()
    for name, _mod in root.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in want:
            found.add(leaf)
    found |= _collect_visual_projector_leaves(model, want)
    out = sorted(found, key=lambda s: (cand.index(s) if s in cand else 999, s))
    return out


def build_bitsandbytes_config(
    *,
    load_in_4bit: bool,
    bf16_compute: bool,
) -> Any:
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if bf16_compute else torch.float16
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def apply_qlora_peft(
    model: nn.Module,
    cfg: VramTrainConfig,
    *,
    gradient_checkpointing: bool,
) -> nn.Module:
    """
    Wrap ``model`` with 4-bit (or 8-bit) prep + LoRA. Call after ``from_pretrained`` on GPU.

    Expects ``cfg.qlora`` True; otherwise returns ``model`` unchanged.
    """
    if not cfg.qlora:
        return model

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    lora_print(
        "QLoRA: prepare_model_for_kbit_training "
        f"(use_gradient_checkpointing={gradient_checkpointing})"
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )

    targets = (
        list(cfg.lora_target_modules)
        if cfg.lora_target_modules is not None
        else resolve_lora_target_modules(model, _default_qlora_target_candidates())
    )
    if not targets:
        targets = list(_default_qlora_target_candidates())
        lora_print(
            "QLoRA: could not auto-detect target modules; falling back to default name list "
            f"(may error if names differ): {targets}"
        )
    else:
        lora_print(f"QLoRA: LoRA target_modules (detected) = {targets}")

    exclude_vision_blocks = _has_visual_tower(model) and _language_model_root(model) is not None
    if exclude_vision_blocks:
        lora_print(
            "QLoRA: excluding PEFT under `visual.blocks` so `linear_fc1`/`linear_fc2` LoRA applies to "
            "merger and deepstack_merger_list only (not every vision MLP block)."
        )

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
        exclude_modules=_EXCLUDE_VISION_BLOCKS_REGEX if exclude_vision_blocks else None,
    )
    model = get_peft_model(model, peft_config)
    trainable = getattr(model, "print_trainable_parameters", None)
    if callable(trainable):
        trainable()
    lora_print("QLoRA: get_peft_model applied (see trainable % above).")
    return model
