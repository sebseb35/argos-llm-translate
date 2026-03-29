from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EngineMode(str, Enum):
    ARGOS = "argos"
    HYBRID = "hybrid"
    LLM = "llm"


@dataclass(slots=True)
class Segment:
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    translatable: bool = True


@dataclass(slots=True)
class TranslationReport:
    segment_count: int
    translated_count: int
    skipped_count: int
    elapsed_seconds: float
    fallback_count: int = 0
    glossary_replacements: int = 0
    llm_calls: int = 0
    llm_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    llm_safe_segments: int = 0
    llm_smart_segments: int = 0
    llm_chunks_built: int = 0
    chunk_fallbacks: int = 0
    segment_fallbacks: int = 0
    avg_chunk_size: float = 0.0
    max_chunk_size: int = 0
    avg_llm_latency_per_segment: float = 0.0
    avg_llm_latency_per_chunk: float = 0.0
    llm_calls_saved_by_chunking: int = 0
    validation_failures: dict[str, int] = field(default_factory=dict)
    placeholder_mismatch_count: int = 0
    glossary_placeholder_mismatch_count: int = 0
    routing_reasons: dict[str, int] = field(default_factory=dict)
    chunk_boundary_reasons: dict[str, int] = field(default_factory=dict)
    chunk_merge_reasons: dict[str, int] = field(default_factory=dict)
    routing_trace: list[dict[str, object]] = field(default_factory=list)
    chunk_trace: list[dict[str, object]] = field(default_factory=list)
