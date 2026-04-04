"""QLoRA / PEFT and VRAM-oriented training helpers (separate from ``vlm_opt`` kernel fusion)."""

from src.vlm_lora.config import VramTrainConfig
from src.vlm_lora.io import lora_print
from src.vlm_lora.peft_qlora import (
    apply_qlora_peft,
    build_bitsandbytes_config,
    resolve_lora_target_modules,
)

__all__ = [
    "VramTrainConfig",
    "apply_qlora_peft",
    "build_bitsandbytes_config",
    "resolve_lora_target_modules",
    "lora_print",
]
