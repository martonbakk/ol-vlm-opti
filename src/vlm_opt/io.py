"""Console output for vlm_opt (always visible; not tied to logging config)."""

from __future__ import annotations

import sys


def safe_print(line: str) -> None:
    """
    Print *line* even when stdout uses a legacy Windows code page (e.g. cp1252)
    that cannot encode Unicode punctuation like arrows or em dashes.
    """
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="replace").decode(enc), flush=True)


def vlm_print(msg: str) -> None:
    """Single prefix so grepping logs is easy."""
    safe_print(f"[vlm_opt] {msg}")
