from __future__ import annotations

import time
from dataclasses import dataclass

from local_translator.config import RuntimeConfig
from local_translator.engines.argos_engine import ArgosEngine
from local_translator.engines.llm_engine import LLMEngine
from local_translator.glossary.store import apply_glossary, load_glossary
from local_translator.models.types import EngineMode, TranslationReport
from local_translator.pipeline.chunker import segment_text
from local_translator.pipeline.postedit import post_edit_segment


@dataclass(slots=True)
class TranslationResult:
    text: str
    report: TranslationReport


class TranslationPipeline:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.argos = ArgosEngine(config.source_lang, config.target_lang)
        self.glossary = load_glossary(config.glossary_path)
        self.llm = None
        if config.engine_mode in {EngineMode.HYBRID, EngineMode.LLM} and config.llm.enabled:
            self.llm = LLMEngine(
                model_path=str(config.llm.model_path) if config.llm.model_path else None,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )

    def translate_text(self, text: str) -> TranslationResult:
        started = time.perf_counter()
        segments = segment_text(text)
        outputs: list[str] = []
        errors: list[str] = []

        for segment in segments:
            try:
                if self.config.engine_mode == EngineMode.LLM:
                    raw = segment
                else:
                    raw = self.argos.translate(segment)

                with_glossary = apply_glossary(raw, self.glossary)
                if self.config.engine_mode in {EngineMode.HYBRID, EngineMode.LLM}:
                    final = post_edit_segment(self.llm, segment, with_glossary, self.glossary)
                else:
                    final = with_glossary
                outputs.append(final)
            except Exception as exc:  # best effort fallback
                errors.append(str(exc))
                outputs.append(segment)

        elapsed = time.perf_counter() - started
        report = TranslationReport(
            segment_count=len(segments),
            translated_count=len(outputs) - len(errors),
            skipped_count=0,
            elapsed_seconds=elapsed,
            errors=errors,
        )
        return TranslationResult(text=" ".join(outputs), report=report)
