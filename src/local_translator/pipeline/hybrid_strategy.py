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
    segment_indices: list[int]
    source_text: str
    draft_text: str
    char_count: int
    placeholder_density: float


def build_llm_chunks(segments: list[str], metadata: list[dict[str, object]]) -> list[LLMChunk]:
    """Build deterministic, conservative contiguous chunks for LLM post-editing."""
    chunks: list[LLMChunk] = []
    current_indices: list[int] = []
    current_source: list[str] = []
    current_draft: list[str] = []
    current_char_count = 0
    current_placeholder_count = 0
    max_segments = 4
    max_chars = 560

    def flush() -> None:
        nonlocal current_indices, current_source, current_draft, current_char_count, current_placeholder_count
        if not current_indices:
            return
        chunks.append(
            LLMChunk(
                segment_indices=current_indices.copy(),
                source_text="\n\n".join(current_source),
                draft_text="\n\n".join(current_draft),
                char_count=current_char_count,
                placeholder_density=current_placeholder_count / max(1, current_char_count),
            )
        )
        current_indices = []
        current_source = []
        current_draft = []
        current_char_count = 0
        current_placeholder_count = 0

    for idx, _segment in enumerate(segments):
        item = metadata[idx]
        should_merge = bool(item.get("can_chunk", False))
        if not should_merge:
            flush()
            continue

        seg_source = str(item.get("source", ""))
        seg_draft = str(item.get("draft", ""))
        seg_char_count = len(seg_draft)
        seg_placeholder_count = int(item.get("placeholder_count", 0))
        if (
            current_indices
            and (
                len(current_indices) >= max_segments
                or (current_char_count + seg_char_count) > max_chars
            )
        ):
            flush()
        current_indices.append(idx)
        current_source.append(seg_source)
        current_draft.append(seg_draft)
        current_char_count += seg_char_count
        current_placeholder_count += seg_placeholder_count

    flush()
    return chunks
