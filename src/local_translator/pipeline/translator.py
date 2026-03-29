from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from local_translator.config import RuntimeConfig
from local_translator.engines.argos_engine import ArgosEngine
from local_translator.engines.llm_engine import LLMEngine
from local_translator.glossary.store import (
    apply_glossary_with_stats,
    load_glossary,
    normalize_restored_text,
    protect_glossary_terms_with_stats,
    restore_glossary_terms_with_stats,
)
from local_translator.models.types import EngineMode, TranslationReport
from local_translator.pipeline.chunker import segment_text
from local_translator.pipeline.hybrid_strategy import build_llm_chunks, decide_llm_postedit
from local_translator.pipeline.postedit import (
    apply_postedit_candidate,
    format_chunk_payload,
    parse_chunk_output,
    post_edit_segment_with_metrics,
)

LOGGER = logging.getLogger(__name__)


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
                n_ctx=config.llm.n_ctx,
                n_threads=config.llm.n_threads,
                n_batch=config.llm.n_batch,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )

    def translate_text(self, text: str) -> TranslationResult:
        started = time.perf_counter()
        segments = segment_text(text)
        outputs: list[str] = [""] * len(segments)
        errors: list[str] = []
        fallback_count = 0
        glossary_replacements = 0
        llm_calls = 0
        llm_skipped = 0
        llm_calls_saved = 0
        prepared_sources: list[str] = []
        prepared_drafts: list[str] = []
        glossary_maps: list[dict[str, str]] = []
        llm_metadata: list[dict[str, object]] = []
        llm_decisions: list[object] = []

        for idx, segment in enumerate(segments):
            segment_started = time.perf_counter()
            try:
                stage_started = time.perf_counter()
                prepared_segment, glossary_token_map, pre_replacements = protect_glossary_terms_with_stats(
                    segment, self.glossary
                )
                glossary_replacements += pre_replacements
                protect_elapsed = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                if self.config.engine_mode == EngineMode.LLM:
                    raw = prepared_segment
                else:
                    raw = self.argos.translate(prepared_segment)
                argos_elapsed = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                with_glossary, replacements = apply_glossary_with_stats(raw, self.glossary)
                glossary_replacements += replacements
                glossary_elapsed = time.perf_counter() - stage_started

                # Normalize Argos-normalized glossary placeholders (e.g. "LT GLOSSARY TERM 0000")
                # back to canonical target terms before LLM post-editing. This avoids turning
                # these artifacts into fragmented protected tokens during strict validation.
                with_glossary, pre_postedit_restore_replacements = restore_glossary_terms_with_stats(
                    with_glossary, glossary_token_map
                )
                glossary_replacements += pre_postedit_restore_replacements
                prepared_sources.append(segment)
                prepared_drafts.append(with_glossary)
                glossary_maps.append(glossary_token_map)
                glossary_entries = self.glossary.entries if hasattr(self.glossary, "entries") else self.glossary
                decision = decide_llm_postedit(segment, with_glossary, glossary_entries, self.config.llm)
                llm_decisions.append(decision)
                technical_signal = any(token in with_glossary for token in ["http", "```", "`", "${", "{{", "--"])
                placeholder_count = with_glossary.count("__LT_")
                can_chunk = (
                    decision.use_llm
                    and bool(decision.mode)
                    and len(with_glossary) >= self.config.llm.skip_short_characters
                    and placeholder_count <= 1
                    and not technical_signal
                )
                llm_metadata.append(
                    {
                        "source": segment,
                        "draft": with_glossary,
                        "mode": decision.mode,
                        "can_chunk": can_chunk and self.config.llm.enable_chunking,
                        "placeholder_count": placeholder_count,
                    }
                )
                postedit_elapsed = 0.0
                restore_elapsed = 0.0
                segment_elapsed = time.perf_counter() - segment_started
                LOGGER.debug(
                    "Segment %d prep timings | protect=%.3fs argos=%.3fs glossary=%.3fs postedit=%.3fs restore=%.3fs total=%.3fs",
                    idx,
                    protect_elapsed,
                    argos_elapsed,
                    glossary_elapsed,
                    postedit_elapsed,
                    restore_elapsed,
                    segment_elapsed,
                )
            except Exception as exc:  # best effort fallback
                errors.append(str(exc))
                prepared_sources.append(segment)
                prepared_drafts.append(segment)
                glossary_maps.append({})
                llm_metadata.append({"source": segment, "draft": segment, "mode": None, "can_chunk": False, "placeholder_count": 0})
                llm_decisions.append(None)

        if self.config.engine_mode in {EngineMode.HYBRID, EngineMode.LLM}:
            handled = set()
            if self.config.llm.enable_chunking:
                chunks = build_llm_chunks(prepared_drafts, llm_metadata)
                for chunk in chunks:
                    if len(chunk.segment_indices) <= 1:
                        continue
                    mode = llm_metadata[chunk.segment_indices[0]].get("mode")
                    if not mode:
                        continue
                    chunk_started = time.perf_counter()
                    llm_calls += 1
                    source_payload, draft_payload = format_chunk_payload(
                        chunk.segment_indices,
                        prepared_sources,
                        prepared_drafts,
                    )
                    LOGGER.debug(
                        "LLM chunk created | segments=%s chars=%d placeholder_density=%.3f mode=%s",
                        chunk.segment_indices,
                        chunk.char_count,
                        chunk.placeholder_density,
                        mode,
                    )
                    try:
                        raw = self.llm.post_edit(
                            source_payload,
                            draft_payload,
                            glossary=self.glossary.entries if hasattr(self.glossary, "entries") else self.glossary,
                            mode=mode,
                        )
                        parsed, parse_reason = parse_chunk_output(raw, len(chunk.segment_indices))
                        if parse_reason or parsed is None:
                            raise ValueError(parse_reason or "chunk_parse_failed")
                        for rel_idx, seg_idx in enumerate(chunk.segment_indices):
                            outcome = apply_postedit_candidate(
                                source_segment=prepared_sources[seg_idx],
                                translated_segment=prepared_drafts[seg_idx],
                                candidate_protected=parsed[rel_idx],
                                glossary=self.glossary,
                                llm_settings=self.config.llm,
                            )
                            if outcome.fallback_used:
                                raise ValueError(outcome.failure_reason or "chunk_validation_failed")
                            prepared_drafts[seg_idx] = outcome.text
                            glossary_replacements += outcome.glossary_replacements
                            handled.add(seg_idx)
                        llm_calls_saved += max(0, len(chunk.segment_indices) - 1)
                        LOGGER.debug(
                            "LLM chunk success | segments=%s elapsed=%.3fs",
                            chunk.segment_indices,
                            time.perf_counter() - chunk_started,
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "LLM chunk fallback to per-segment | segments=%s reason=%s elapsed=%.3fs",
                            chunk.segment_indices,
                            str(exc),
                            time.perf_counter() - chunk_started,
                        )
                        for seg_idx in chunk.segment_indices:
                            decision = llm_decisions[seg_idx]
                            if not decision or not decision.use_llm or not decision.mode:
                                llm_skipped += 1
                                continue
                            llm_calls += 1
                            outcome = post_edit_segment_with_metrics(
                                self.llm,
                                prepared_sources[seg_idx],
                                prepared_drafts[seg_idx],
                                self.glossary,
                                self.config.llm,
                                mode=decision.mode,
                            )
                            prepared_drafts[seg_idx] = outcome.text
                            fallback_count += int(outcome.fallback_used)
                            glossary_replacements += outcome.glossary_replacements
                            handled.add(seg_idx)

            for idx, decision in enumerate(llm_decisions):
                if idx in handled:
                    continue
                if decision and decision.use_llm and decision.mode:
                    llm_calls += 1
                    outcome = post_edit_segment_with_metrics(
                        self.llm,
                        prepared_sources[idx],
                        prepared_drafts[idx],
                        self.glossary,
                        self.config.llm,
                        mode=decision.mode,
                    )
                    prepared_drafts[idx] = outcome.text
                    fallback_count += int(outcome.fallback_used)
                    glossary_replacements += outcome.glossary_replacements
                else:
                    llm_skipped += 1
                    reason = decision.reason if decision else "segment_error"
                    LOGGER.debug("Segment %d skipped LLM post-edit (%s)", idx, reason)

        for idx, final in enumerate(prepared_drafts):
            glossary_token_map = glossary_maps[idx]
            final, post_restore_replacements = restore_glossary_terms_with_stats(final, glossary_token_map)
            glossary_replacements += post_restore_replacements
            final, post_replacements = apply_glossary_with_stats(final, self.glossary)
            glossary_replacements += post_replacements
            outputs[idx] = normalize_restored_text(final)

        elapsed = time.perf_counter() - started
        if llm_calls_saved:
            LOGGER.debug("Chunking saved %d LLM calls", llm_calls_saved)
        report = TranslationReport(
            segment_count=len(segments),
            translated_count=len(outputs) - len(errors),
            skipped_count=0,
            elapsed_seconds=elapsed,
            fallback_count=fallback_count,
            glossary_replacements=glossary_replacements,
            llm_calls=llm_calls,
            llm_skipped=llm_skipped,
            errors=errors,
        )
        return TranslationResult(text=" ".join(outputs), report=report)
