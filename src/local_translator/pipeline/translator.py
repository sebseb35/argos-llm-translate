from __future__ import annotations

import logging
import time
from collections import Counter
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
        llm_safe_segments = 0
        llm_smart_segments = 0
        llm_segment_latencies: list[float] = []
        llm_chunk_latencies: list[float] = []
        chunk_sizes: list[int] = []
        chunk_fallbacks = 0
        segment_fallbacks = 0
        validation_failures: Counter[str] = Counter()
        routing_reasons: Counter[str] = Counter()
        chunk_boundary_reasons: Counter[str] = Counter()
        chunk_merge_reasons: Counter[str] = Counter()
        routing_trace: list[dict[str, object]] = []
        chunk_trace: list[dict[str, object]] = []

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
                routing_reasons[decision.reason] += 1
                routing_trace.append(
                    {
                        "segment_index": idx,
                        "mode": decision.mode if decision.use_llm else "skip",
                        "reason": decision.reason,
                        "char_count": len(with_glossary),
                        "placeholder_count": with_glossary.count("__LT_"),
                    }
                )
                if decision.mode == "safe":
                    llm_safe_segments += 1
                elif decision.mode == "smart":
                    llm_smart_segments += 1

                technical_signal = any(token in with_glossary for token in ["http", "```", "`", "${", "{{", "--"])
                placeholder_count = with_glossary.count("__LT_")
                can_chunk = (
                    decision.use_llm
                    and bool(decision.mode)
                    and len(with_glossary) >= self.config.llm.chunk_min_chars_for_merge
                    and placeholder_count <= self.config.llm.chunk_max_placeholders_per_segment
                    and not technical_signal
                    and self.config.llm.enable_chunking
                )
                llm_metadata.append(
                    {
                        "source": segment,
                        "draft": with_glossary,
                        "mode": decision.mode,
                        "can_chunk": can_chunk,
                        "placeholder_count": placeholder_count,
                    }
                )
                segment_elapsed = time.perf_counter() - segment_started
                LOGGER.debug(
                    "Segment %d prep timings | protect=%.3fs argos=%.3fs glossary=%.3fs total=%.3fs",
                    idx,
                    protect_elapsed,
                    argos_elapsed,
                    glossary_elapsed,
                    segment_elapsed,
                )
            except Exception as exc:  # best effort fallback
                errors.append(str(exc))
                prepared_sources.append(segment)
                prepared_drafts.append(segment)
                glossary_maps.append({})
                llm_metadata.append({"source": segment, "draft": segment, "mode": None, "can_chunk": False, "placeholder_count": 0})
                llm_decisions.append(None)
                routing_reasons["segment_error"] += 1
                routing_trace.append({"segment_index": idx, "mode": "skip", "reason": "segment_error", "char_count": len(segment), "placeholder_count": 0})

        if self.config.engine_mode in {EngineMode.HYBRID, EngineMode.LLM}:
            handled = set()
            if self.config.llm.enable_chunking:
                chunks = build_llm_chunks(prepared_drafts, llm_metadata, self.config.llm)
                for chunk in chunks:
                    if len(chunk.segment_indices) <= 1:
                        continue
                    mode = llm_metadata[chunk.segment_indices[0]].get("mode")
                    if not mode:
                        continue
                    chunk_sizes.append(len(chunk.segment_indices))
                    chunk_merge_reasons[chunk.merge_reason] += 1
                    if chunk.boundary_reason:
                        chunk_boundary_reasons[chunk.boundary_reason] += 1
                    chunk_trace.append(
                        {
                            "segment_indices": chunk.segment_indices,
                            "routing_mode": mode,
                            "character_count": chunk.char_count,
                            "placeholder_count": sum(int(llm_metadata[i].get("placeholder_count", 0)) for i in chunk.segment_indices),
                            "merge_reason": chunk.merge_reason,
                            "boundary_reason": chunk.boundary_reason,
                        }
                    )
                    chunk_started = time.perf_counter()
                    llm_calls += 1
                    source_payload, draft_payload = format_chunk_payload(
                        chunk.segment_indices,
                        prepared_sources,
                        prepared_drafts,
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
                            validation_failures[parse_reason or "chunk_parse_failed"] += 1
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
                                validation_failures[outcome.failure_reason or "chunk_validation_failed"] += 1
                                raise ValueError(outcome.failure_reason or "chunk_validation_failed")
                            prepared_drafts[seg_idx] = outcome.text
                            glossary_replacements += outcome.glossary_replacements
                            handled.add(seg_idx)
                        llm_calls_saved += max(0, len(chunk.segment_indices) - 1)
                        llm_chunk_latencies.append(time.perf_counter() - chunk_started)
                    except Exception as exc:
                        chunk_fallbacks += 1
                        llm_chunk_latencies.append(time.perf_counter() - chunk_started)
                        LOGGER.warning(
                            "LLM chunk fallback to per-segment | segments=%s reason=%s",
                            chunk.segment_indices,
                            str(exc),
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
                            llm_segment_latencies.append(outcome.llm_latency_seconds)
                            prepared_drafts[seg_idx] = outcome.text
                            fallback_count += int(outcome.fallback_used)
                            segment_fallbacks += int(outcome.fallback_used)
                            glossary_replacements += outcome.glossary_replacements
                            if outcome.failure_reason:
                                validation_failures[outcome.failure_reason] += 1
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
                    llm_segment_latencies.append(outcome.llm_latency_seconds)
                    prepared_drafts[idx] = outcome.text
                    fallback_count += int(outcome.fallback_used)
                    segment_fallbacks += int(outcome.fallback_used)
                    glossary_replacements += outcome.glossary_replacements
                    if outcome.failure_reason:
                        validation_failures[outcome.failure_reason] += 1
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
            llm_safe_segments=llm_safe_segments,
            llm_smart_segments=llm_smart_segments,
            llm_chunks_built=len(chunk_sizes),
            chunk_fallbacks=chunk_fallbacks,
            segment_fallbacks=segment_fallbacks,
            avg_chunk_size=(sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
            max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
            avg_llm_latency_per_segment=(sum(llm_segment_latencies) / len(llm_segment_latencies)) if llm_segment_latencies else 0.0,
            avg_llm_latency_per_chunk=(sum(llm_chunk_latencies) / len(llm_chunk_latencies)) if llm_chunk_latencies else 0.0,
            llm_calls_saved_by_chunking=llm_calls_saved,
            validation_failures=dict(validation_failures),
            placeholder_mismatch_count=validation_failures.get("placeholder_mismatch", 0),
            glossary_placeholder_mismatch_count=validation_failures.get("glossary_placeholder_mismatch", 0),
            routing_reasons=dict(routing_reasons),
            chunk_boundary_reasons=dict(chunk_boundary_reasons),
            chunk_merge_reasons=dict(chunk_merge_reasons),
            routing_trace=routing_trace,
            chunk_trace=chunk_trace,
        )
        return TranslationResult(text=" ".join(outputs), report=report)
