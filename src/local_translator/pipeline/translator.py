from __future__ import annotations

import time
from dataclasses import dataclass

from local_translator.config import RuntimeConfig
from local_translator.engines.argos_engine import ArgosEngine
from local_translator.engines.llm_engine import LLMEngine
from local_translator.glossary.store import (
    apply_glossary_with_stats,
    load_glossary,
    protect_glossary_terms_with_stats,
    restore_glossary_terms_with_stats,
)
from local_translator.models.types import EngineMode, TranslationReport
from local_translator.pipeline.chunker import segment_text
from local_translator.pipeline.postedit import post_edit_segment_with_metrics


@dataclass(slots=True)
class TranslationResult:
    text: str
    report: TranslationReport


class TranslationPipeline:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.argos = ArgosEngine(config.source_lang, config.target_lang)
        self.glossary = load_glossary(config.glossary_path, config.source_lang, config.target_lang)
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
        fallback_count = 0
        glossary_replacements = 0

        for segment in segments:
            try:
                prepared_segment, glossary_token_map, pre_replacements = protect_glossary_terms_with_stats(
                    segment, self.glossary
                )
                glossary_replacements += pre_replacements

                if self.config.engine_mode == EngineMode.LLM:
                    raw = prepared_segment
                else:
                    raw = self.argos.translate(prepared_segment)

                with_glossary, replacements = apply_glossary_with_stats(raw, self.glossary)
                glossary_replacements += replacements
                if self.config.engine_mode in {EngineMode.HYBRID, EngineMode.LLM}:
                    outcome = post_edit_segment_with_metrics(
                        self.llm,
                        segment,
                        with_glossary,
                        self.glossary,
                        self.config.llm,
                    )
                    final = outcome.text
                    fallback_count += int(outcome.fallback_used)
                    glossary_replacements += outcome.glossary_replacements
                else:
                    final = with_glossary
                final, post_restore_replacements = restore_glossary_terms_with_stats(final, glossary_token_map)
                glossary_replacements += post_restore_replacements
                final, post_replacements = apply_glossary_with_stats(final, self.glossary)
                glossary_replacements += post_replacements
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
            fallback_count=fallback_count,
            glossary_replacements=glossary_replacements,
            errors=errors,
        )
        return TranslationResult(text=" ".join(outputs), report=report)
