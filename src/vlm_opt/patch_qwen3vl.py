"""Runtime module swap for Qwen3-VL vision ``PatchMerger`` (fused Linear+GELU)."""

from __future__ import annotations

import types
from typing import Any, cast

import torch.nn as nn

from src.vlm_opt.config import MergerFusedBackend
from src.vlm_opt.fused_linear_gelu import FusedLinearGELU
from src.vlm_opt.io import vlm_print

_MERGER_PATCHED_ATTR = "_ol_fused_merger_patched"


def _patch_single_merger(merger: nn.Module, backend: MergerFusedBackend, *, label: str) -> None:
    if getattr(merger, _MERGER_PATCHED_ATTR, False):
        return
    if not hasattr(merger, "linear_fc1") or not hasattr(merger, "act_fn") or not hasattr(merger, "linear_fc2"):
        raise TypeError(f"Expected Qwen3VLVisionPatchMerger-like module, got {type(merger)}")

    old_linear = merger.linear_fc1
    if not isinstance(old_linear, nn.Linear):
        raise TypeError("merger.linear_fc1 must be nn.Linear before fusion patch.")

    fused = FusedLinearGELU.from_linear_and_gelu(old_linear)
    fused.set_backend(backend)
    # Keep parameter name ``linear_fc1`` so checkpoints / PEFT still line up with ``*.linear_fc1.weight``.
    merger.linear_fc1 = fused
    if "act_fn" in merger._modules:
        merger._modules.pop("act_fn")

    def _new_forward(self: Any, x: Any) -> Any:
        # Dynamic PatchMerger attributes (norm, linear_fc1, linear_fc2, hidden_size, …).
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.linear_fc1(x))
        return x

    merger.forward = types.MethodType(_new_forward, merger)  # type: ignore[method-assign]
    setattr(merger, _MERGER_PATCHED_ATTR, True)
    fc1 = merger.linear_fc1
    desc = fc1.describe_backend() if isinstance(fc1, FusedLinearGELU) else str(type(fc1))
    vlm_print(f"PatchMerger '{label}': fusion APPLIED - linear_fc1 now FusedLinearGELU ({desc})")


def patch_qwen3vl_vision_mergers(root: nn.Module, backend: MergerFusedBackend | str = MergerFusedBackend.auto) -> int:
    """
    Replace ``linear_fc1`` + ``act_fn`` with :class:`FusedLinearGELU` (assigned to ``linear_fc1``).

    ``root`` should be the loaded HF model (e.g. ``Qwen3VLForConditionalGeneration``): resolves ``model.visual``.

    Returns:
        Number of merger modules patched.
    """
    if isinstance(backend, str):
        backend = MergerFusedBackend(backend)

    visual = _resolve_visual(root)
    count = 0
    merger = getattr(visual, "merger", None)
    if merger is not None:
        _patch_single_merger(cast(nn.Module, merger), backend, label="visual.merger")
        count += 1
    ds_list = getattr(visual, "deepstack_merger_list", None)
    if ds_list is not None:
        for i, m in enumerate(ds_list):
            _patch_single_merger(cast(nn.Module, m), backend, label=f"visual.deepstack_merger_list[{i}]")
            count += 1
    return count


def _resolve_visual(root: nn.Module) -> nn.Module:
    inner = getattr(root, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        return cast(nn.Module, getattr(inner, "visual"))
    if hasattr(root, "visual"):
        return cast(nn.Module, getattr(root, "visual"))
    raise AttributeError(
        "Could not find vision tower (expected ``model.visual`` or ``visual``). "
        "Pass the top-level Qwen3-VL image-text model."
    )


def is_merger_fusion_active(root: nn.Module) -> bool:
    try:
        visual = _resolve_visual(root)
    except AttributeError:
        return False
    m = getattr(visual, "merger", None)
    return bool(m is not None and getattr(m, _MERGER_PATCHED_ATTR, False))
