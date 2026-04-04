"""Fused Linear + GELU for Qwen3-VL ``PatchMerger`` (matches ``nn.Linear`` + ``nn.GELU()`` exact)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vlm_opt.io import vlm_print

if TYPE_CHECKING:
    from src.vlm_opt.config import MergerFusedBackend

logger = logging.getLogger(__name__)

_TRITON_KERNEL_OK = False
_triton_fused_linear_gelu_forward = None

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_linear_gelu_row_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        n_base = tl.program_id(1) * BLOCK_N
        n_idx = n_base + tl.arange(0, BLOCK_N)
        mask_n = n_idx < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < K
            xk = tl.load(
                x_ptr + m * stride_xm + k_idx * stride_xk,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            w_blk = tl.load(
                w_ptr + n_idx[:, None] * stride_wn + k_idx[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w_blk * xk[None, :], axis=1)
        b = tl.load(b_ptr + n_idx, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + b
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865475244))
        tl.store(out_ptr + m * N + n_idx, y, mask=mask_n)

    def _triton_fused_linear_gelu_forward_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        M, K = x.shape
        N, Kw = weight.shape
        assert K == Kw and bias.shape[0] == N
        x_ = x.contiguous()
        w_ = weight.contiguous()
        b_ = bias.contiguous()
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (M, triton.cdiv(N, BLOCK_N))
        _fused_linear_gelu_row_kernel[grid](
            x_,
            w_,
            b_,
            out,
            M,
            N,
            K,
            x_.stride(0),
            x_.stride(1),
            w_.stride(0),
            w_.stride(1),
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return out.to(dtype=x.dtype)

    _triton_fused_linear_gelu_forward = _triton_fused_linear_gelu_forward_impl
    _TRITON_KERNEL_OK = True
except Exception as e:
    logger.debug("vlm_opt: Triton fused kernel not available: %s", e)


def _gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """Match ``nn.GELU()`` default (approximate='none')."""
    return F.gelu(x, approximate="none")


def _is_triton_usable() -> bool:
    if not _TRITON_KERNEL_OK or _triton_fused_linear_gelu_forward is None:
        return False
    if os.environ.get("VLM_OPT_DISABLE_TRITON", "").strip().lower() in ("1", "true", "yes"):
        return False
    if not torch.cuda.is_available():
        return False
    return True


class FusedLinearGELU(nn.Module):
    """
    Single submodule replacing ``nn.Linear`` + ``nn.GELU()`` with the same numerics (exact GELU).

    Weight layout matches ``nn.Linear.weight``: ``[out_features, in_features]``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._use_triton_fwd = False
        self._compiled_forward = None
        self._effective_backend = "uninitialized"

    def describe_backend(self) -> str:
        """Human-readable resolved backend after ``set_backend``."""
        return self._effective_backend

    @classmethod
    def from_linear_and_gelu(cls, linear: nn.Linear) -> FusedLinearGELU:
        m = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            m.weight.copy_(linear.weight)
            if linear.bias is not None and m.bias is not None:
                m.bias.copy_(linear.bias)
        return m

    def set_backend(self, backend: MergerFusedBackend | str) -> None:
        from src.vlm_opt.config import MergerFusedBackend

        if isinstance(backend, str):
            backend = MergerFusedBackend(backend)
        self._compiled_forward = None
        self._use_triton_fwd = False

        if backend == MergerFusedBackend.auto:
            backend = MergerFusedBackend.triton if _is_triton_usable() else MergerFusedBackend.pytorch

        if backend == MergerFusedBackend.triton:
            self._use_triton_fwd = _is_triton_usable()
            if self._use_triton_fwd:
                self._effective_backend = (
                    "triton (custom fused kernel in eval only; training uses PyTorch eager + autograd)"
                )
            else:
                self._effective_backend = "pytorch eager (Triton unavailable or VLM_OPT_DISABLE_TRITON set)"
                logger.info("vlm_opt: Triton unavailable or disabled; using PyTorch Linear+GELU.")
        elif backend == MergerFusedBackend.compile_wrap:
            mode = os.environ.get("VLM_OPT_COMPILE_MODE", "reduce-overhead")
            self._compiled_forward = torch.compile(
                self._forward_eager,
                mode=mode,
                fullgraph=False,
            )
            self._effective_backend = f"compile_wrap (torch.compile mode={mode} on Linear+GELU forward)"
        else:
            self._effective_backend = "pytorch eager (F.linear + F.gelu exact)"

        vlm_print(
            f"FusedLinearGELU [{self.in_features}->{self.out_features}]: ACTIVE -> {self._effective_backend}"
        )

    def _forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu_exact(F.linear(x, self.weight, self.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is not None:
            return self._compiled_forward(x)
        if (
            not self.training
            and self._use_triton_fwd
            and _triton_fused_linear_gelu_forward is not None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            try:
                return _triton_fused_linear_gelu_forward(x, self.weight, self.bias)
            except Exception as e:
                logger.warning("vlm_opt: Triton fused forward failed (%s); falling back to PyTorch.", e)
        return self._forward_eager(x)
