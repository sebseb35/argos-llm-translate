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
    errors: list[str] = field(default_factory=list)
