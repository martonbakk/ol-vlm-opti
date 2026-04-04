"""QLoRA / PEFT training switches (separate from ``vlm_opt`` kernel-fusion config)."""

from pydantic import BaseModel, Field


class VramTrainConfig(BaseModel):
    """
    Lower-VRAM training: quantized base + LoRA on the LLM and vision projector (merger / deepstack merger).

    QLoRA: 4-bit (or 8-bit) NF4 base weights + LoRA adapters. Pair with ``TrainingArguments(bf16=True)``,
    small batch size, and optional gradient checkpointing.
    """

    qlora: bool = Field(
        default=False,
        description="Enable bitsandbytes quant + PEFT LoRA on language layers and vision projectors (not raw vision blocks).",
    )
    load_in_4bit: bool = Field(
        default=True,
        description="If True, NF4 4-bit; if False, 8-bit quantization (still LoRA, not full fine-tune).",
    )
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: tuple[str, ...] | None = Field(
        default=None,
        description="If None, auto-detect LLM projections plus vision merger linear_fc1/linear_fc2 (Qwen3-VL).",
    )
