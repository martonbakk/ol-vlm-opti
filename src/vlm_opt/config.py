"""Switches for VLM backend optimizations (kernel fusion, module swap, optional Liger)."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class MergerFusedBackend(str, Enum):
    """How to run fused Linear+GELU in the vision projector (PatchMerger)."""

    auto = "auto"  # Triton if importable, else PyTorch
    pytorch = "pytorch"  # F.linear + F.gelu (Inductor can fuse under torch.compile)
    triton = "triton"  # Custom Triton epilogue (may be unavailable on Windows)
    compile_wrap = "compile_wrap"  # Tiny nn.Module wrapped with torch.compile (per-module CUDA graphs)


class VlmOptConfig(BaseModel):
    """Toggle backend optimizations without editing call sites."""

    enabled: bool = Field(default=False, description="Master switch; if False, no patches are applied.")
    fused_vision_merger: bool = Field(
        default=True,
        description="Replace PatchMerger linear_fc1 + GELU with a single fused module.",
    )
    merger_fused_backend: MergerFusedBackend = Field(
        default=MergerFusedBackend.auto,
        description="Implementation for fused Linear+GELU.",
    )
    liger_language_model: bool = Field(
        default=False,
        description="If True, try applying Liger-Kernel monkey patches to the text backbone (optional dep).",
    )
    torch_compile_full_model: bool = Field(
        default=False,
        description="Wrap root model with torch.compile(mode='reduce-overhead'); experimental.",
    )
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="reduce-overhead",
    )
