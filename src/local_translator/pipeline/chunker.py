from __future__ import annotations

import re


def segment_text(text: str) -> list[str]:
    if not text.strip():
        return [text]
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text) if c.strip()]
    return chunks or [text]
