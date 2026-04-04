"""VLM backend optimizations: fused projector, optional Triton/Liger, monkey patching."""

from src.vlm_opt.apply import apply_vlm_optimizations
from src.vlm_opt.config import MergerFusedBackend, VlmOptConfig
from src.vlm_opt.fused_linear_gelu import FusedLinearGELU
from src.vlm_opt.io import vlm_print
from src.vlm_opt.patch_qwen3vl import is_merger_fusion_active, patch_qwen3vl_vision_mergers

__all__ = [
    "VlmOptConfig",
    "MergerFusedBackend",
    "FusedLinearGELU",
    "apply_vlm_optimizations",
    "patch_qwen3vl_vision_mergers",
    "is_merger_fusion_active",
    "vlm_print",
]
