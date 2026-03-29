from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PostEditPolicy = Literal["off", "auto", "safe", "smart"]

from local_translator.models.types import EngineMode

LanguageCode = Literal["fr", "en"]


@dataclass(slots=True)
class LLMSettings:
    enabled: bool = False
    model_path: Path | None = None
    n_ctx: int = 1024
    n_threads: int | None = 2
    n_batch: int = 64
    temperature: float = 0.0
    max_tokens: int = 256
    strict_validation: bool = True
    fallback_to_argos: bool = True
    max_expansion_ratio: float = 1.4
    postedit_mode: PostEditPolicy = "auto"
    skip_short_characters: int = 48
    skip_high_placeholder_ratio: float = 0.12
    smart_min_chars: int = 160
    enable_chunking: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    source_lang: LanguageCode
    target_lang: LanguageCode
    engine_mode: EngineMode = EngineMode.ARGOS
    glossary_path: Path | None = None
    llm: LLMSettings = field(default_factory=LLMSettings)
    report: bool = False
