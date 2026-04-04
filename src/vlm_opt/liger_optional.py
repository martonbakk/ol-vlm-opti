"""Optional Liger-Kernel integration for the text backbone (RMSNorm / RoPE fused ops)."""

from __future__ import annotations

import logging
from typing import Any

from src.vlm_opt.io import vlm_print

logger = logging.getLogger(__name__)


def try_apply_liger_to_qwen3_language_model(model: Any) -> bool:
    """
    Try to apply Liger-Kernel monkey patches for Qwen3 **text** blocks.

    Requires the ``liger-kernel`` package. Qwen3-VL-specific patches may differ by version; failures are logged only.

    Returns:
        True if at least one patch entrypoint completed without raising.
    """
    try:
        from liger_kernel.transformers import monkey_patch as mp
    except ImportError:
        vlm_print("Liger: NOT ACTIVE (liger-kernel package not installed).")
        logger.info("vlm_opt: liger-kernel not installed; skip Liger integration.")
        return False

    for name in ("apply_liger_kernel_to_qwen3", "apply_liger_kernel_to_qwen2"):
        fn = getattr(mp, name, None)
        if callable(fn):
            try:
                fn()
                vlm_print(f"Liger: ACTIVE (monkey_patch.{name}() succeeded).")
                logger.info("vlm_opt: applied Liger monkey patch: %s", name)
                return True
            except Exception as e:
                vlm_print(f"Liger: NOT ACTIVE ({name} raised: {e!s}).")
                logger.warning("vlm_opt: Liger patch %s failed: %s", name, e)
    vlm_print("Liger: NOT ACTIVE (no apply_liger_kernel_to_qwen3 / _qwen2 entrypoint worked).")
    logger.info("vlm_opt: No working Liger apply_* entrypoint found.")
    return False
