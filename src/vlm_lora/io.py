"""Console output for ``vlm_lora`` (separate prefix from ``vlm_opt`` fusion)."""

from __future__ import annotations

import sys


def lora_print(msg: str) -> None:
    line = f"[vlm_lora] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="replace").decode(enc), flush=True)
