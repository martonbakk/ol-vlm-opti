"""
Vision token sparsity for Qwen2-VL / Qwen3-VL: down-weight less important merged patch
embeddings per image while keeping the **same length and row order** as the vision grid.

**Important:** row ``i`` must stay at index ``i`` so placeholders / RoPE / grid stay aligned.
(Older versions concatenated top-k rows + zeros at the end, which permuted positions and
blew up the loss.)

Dropped rows are filled with the mean of the **kept** rows (per chunk), not zeros.

Patches ``get_image_features`` (pooler + deepstack), not ``visual.forward``.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn

_PATCH_ATTR = "_ol_qwen_vl_image_sparsity_patched"


def _resolve_qwen_vl_model(model: nn.Module) -> nn.Module:
    """Unwrap PEFT / wrapper to the inner module that defines ``get_image_features``."""
    m: Any = model
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        inner = m.base_model.model
        if hasattr(inner, "get_image_features"):
            return inner
    if hasattr(m, "model") and hasattr(m.model, "get_image_features"):
        return m.model
    if hasattr(m, "get_image_features"):
        return m
    raise TypeError(
        f"Cannot find Qwen VL inner model with get_image_features (got {type(model).__name__})."
    )


def _infer_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    p = next(model.parameters(), None)
    if p is not None:
        return p.device, p.dtype
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return d, torch.float32


def _topk_indices_kept(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Row indices to keep (sorted ascending) for x of shape (S, D)."""
    s = x.size(0)
    if keep_ratio >= 1.0:
        return torch.arange(s, device=x.device, dtype=torch.long)
    k = max(1, int(s * keep_ratio))
    imp = torch.norm(x, p=2, dim=-1)
    idx = torch.topk(imp, k=k, largest=True).indices
    return torch.sort(idx).values


def _prune_chunk_same_length(x: torch.Tensor, idx_kept: torch.Tensor) -> torch.Tensor:
    """
    Same shape (S, D), **same row order** as the merged vision sequence (grid scan order).

    Rows in ``idx_kept`` keep ``x[idx]``; all other rows are set to ``mean(x[idx_kept], dim=0)``
    so placeholders still line up with spatial indices and the LM does not see a permuted image.
    """
    s, _d = x.shape
    k = idx_kept.numel()
    if k >= s:
        return x
    out = x.clone()
    mean_kept = x[idx_kept].mean(dim=0)
    keep = torch.zeros(s, dtype=torch.bool, device=x.device)
    keep[idx_kept] = True
    out[~keep] = mean_kept
    return out


def apply_qwen3_sparsity(model: nn.Module, keep_ratio: float = 0.5) -> nn.Module:
    """
    Patch ``get_image_features`` to drop less important **merged** patch tokens per image,
    padding back to the original length so multimodal placeholder counts stay valid.

    Safe to call **before** PEFT QLoRA wrap; the same inner module is kept, so the patch
    remains after ``get_peft_model``.
    """
    vl = _resolve_qwen_vl_model(model)
    if getattr(vl, _PATCH_ATTR, False):
        print(f"[token_pruner] Qwen VL image sparsity already active (keep_ratio={keep_ratio})")
        return model

    orig = vl.get_image_features

    def _patched_get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        out = orig(pixel_values, image_grid_thw=image_grid_thw, **kwargs)

        if image_grid_thw is None:
            return out

        merge_sq = self.visual.spatial_merge_size**2
        split_sizes = (image_grid_thw.prod(-1) // merge_sq).tolist()
        pooler = out.pooler_output
        # After original: tuple of per-image merged embeds
        if isinstance(pooler, tuple):
            chunks = list(pooler)
        else:
            flat = pooler
            chunks = list(torch.split(flat, split_sizes))

        new_chunks: list[torch.Tensor] = []
        idx_per_chunk: list[torch.Tensor] = []

        for ch in chunks:
            idx = _topk_indices_kept(ch, keep_ratio)
            idx_per_chunk.append(idx)
            new_chunks.append(_prune_chunk_same_length(ch, idx))

        out.pooler_output = tuple(new_chunks)

        # Qwen3-VL DeepStack: same sequence layout as merged pooler (concat length)
        ds_list = getattr(out, "deepstack_features", None)
        if ds_list:
            new_ds = []
            for ds in ds_list:
                parts = torch.split(ds, split_sizes, dim=0)
                rebuilt = []
                for i, part in enumerate(parts):
                    rebuilt.append(_prune_chunk_same_length(part, idx_per_chunk[i]))
                new_ds.append(torch.cat(rebuilt, dim=0))
            out.deepstack_features = new_ds

        return out

    vl.get_image_features = types.MethodType(_patched_get_image_features, vl)
    setattr(vl, _PATCH_ATTR, True)

    _, dt = _infer_device_dtype(vl)
    print(
        f"[OK] Qwen VL image sparsity active on get_image_features (keep_ratio={keep_ratio}, dtype={dt})"
    )
    return model
