from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from local_translator.config import LLMSettings

PostEditMode = Literal["safe", "smart"]

_PLACEHOLDER_RE = re.compile(r"__LT_[A-Z0-9_]+__")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class SegmentFeatures:
    char_count: int
    placeholder_count: int
    technical_token_count: int
    sentence_count: int
    glossary_hits: int

    @property
    def placeholder_ratio(self) -> float:
        return self.placeholder_count / max(1, self.char_count)


@dataclass(slots=True)
class LLMDecision:
    use_llm: bool
    mode: PostEditMode | None = None
    reason: str = ""


def extract_segment_features(segment: str, translated_segment: str, glossary: dict[str, str]) -> SegmentFeatures:
    char_count = len(translated_segment)
    placeholder_count = len(_PLACEHOLDER_RE.findall(translated_segment))
    technical_token_count = sum(token in translated_segment for token in ["http", "`", "${", "{{", "--"])
    sentence_count = max(1, len([p for p in _SENTENCE_SPLIT_RE.split(translated_segment) if p.strip()]))
    glossary_hits = sum(1 for term in glossary.values() if term and term in translated_segment)
    return SegmentFeatures(
        char_count=char_count,
        placeholder_count=placeholder_count,
        technical_token_count=technical_token_count,
        sentence_count=sentence_count,
        glossary_hits=glossary_hits,
    )


def decide_llm_postedit(
    segment: str,
    translated_segment: str,
    glossary: dict[str, str],
    llm_settings: LLMSettings,
) -> LLMDecision:
    mode = llm_settings.postedit_mode
    if mode == "off":
        return LLMDecision(use_llm=False, reason="mode_off")

    features = extract_segment_features(segment, translated_segment, glossary)

    if features.char_count < llm_settings.skip_short_characters:
        return LLMDecision(use_llm=False, reason="short_segment")

    if features.placeholder_ratio > llm_settings.skip_high_placeholder_ratio:
        return LLMDecision(use_llm=False, reason="placeholder_dense")

    if mode in {"safe", "smart"}:
        return LLMDecision(use_llm=True, mode=mode, reason="forced_mode")

    if features.technical_token_count > 0 or features.placeholder_count >= 2:
        return LLMDecision(use_llm=True, mode="safe", reason="technical_or_placeholder")

    if features.char_count >= llm_settings.smart_min_chars or features.sentence_count > 1:
        return LLMDecision(use_llm=True, mode="smart", reason="long_or_multisentence")

    return LLMDecision(use_llm=True, mode="safe", reason="default_safe")


@dataclass(slots=True)
class LLMChunk:
    start_index: int
    end_index: int
    segment_indices: list[int]


def build_llm_chunks(segments: list[str], target_chars: int, max_chars: int) -> list[LLMChunk]:
    """Build conservative contiguous chunks for future chunk-level post-editing.

    This planner is intentionally side-effect free and can be adopted incrementally.
    """
    chunks: list[LLMChunk] = []
    current_indices: list[int] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_indices, current_len
        if not current_indices:
            return
        chunks.append(
            LLMChunk(
                start_index=current_indices[0],
                end_index=current_indices[-1],
                segment_indices=current_indices.copy(),
            )
        )
        current_indices = []
        current_len = 0

    for idx, segment in enumerate(segments):
        seg_len = len(segment)
        if current_indices and (current_len + seg_len > max_chars):
            flush()

        current_indices.append(idx)
        current_len += seg_len

        if current_len >= target_chars:
            flush()

    flush()
    return chunks
