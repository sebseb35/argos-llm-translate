from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from local_translator.models.types import EngineMode

LanguageCode = Literal["fr", "en"]


@dataclass(slots=True)
class LLMSettings:
    enabled: bool = False
    model_path: Path | None = None
    n_ctx: int = 2048
    temperature: float = 0.1
    max_tokens: int = 256


@dataclass(slots=True)
class RuntimeConfig:
    source_lang: LanguageCode
    target_lang: LanguageCode
    engine_mode: EngineMode = EngineMode.ARGOS
    glossary_path: Path | None = None
    llm: LLMSettings = field(default_factory=LLMSettings)
    report: bool = False
