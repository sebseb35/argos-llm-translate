from __future__ import annotations

import json
from pathlib import Path

from local_translator.models.types import TranslationReport


def format_report(report: TranslationReport) -> str:
    lines = [
        "Execution report",
        f"Segments processed: {report.segment_count}",
        f"LLM skipped: {report.llm_skipped}",
        f"SAFE segments: {report.llm_safe_segments}",
        f"SMART segments: {report.llm_smart_segments}",
        f"LLM calls: {report.llm_calls}",
        f"Chunks built: {report.llm_chunks_built} (avg size={report.avg_chunk_size:.2f}, max size={report.max_chunk_size})",
        f"Chunk fallbacks: {report.chunk_fallbacks}",
        f"Segment fallbacks: {report.segment_fallbacks}",
        f"Fallbacks: {report.fallback_count}",
        f"Validation failures: {report.validation_failures}",
        f"Placeholder mismatches: {report.placeholder_mismatch_count}",
        f"Glossary placeholder mismatches: {report.glossary_placeholder_mismatch_count}",
        f"Avg LLM latency per segment: {report.avg_llm_latency_per_segment:.3f}s",
        f"Avg LLM latency per chunk: {report.avg_llm_latency_per_chunk:.3f}s",
        f"Estimated calls saved by chunking: {report.llm_calls_saved_by_chunking}",
        f"Errors: {len(report.errors)}",
        f"Glossary replacements: {report.glossary_replacements}",
        f"Time: {report.elapsed_seconds:.3f}s",
    ]
    return "\n".join(lines)


def report_to_dict(report: TranslationReport) -> dict[str, object]:
    return {
        "segment_count": report.segment_count,
        "translated_count": report.translated_count,
        "skipped_count": report.skipped_count,
        "elapsed_seconds": round(report.elapsed_seconds, 6),
        "error_count": len(report.errors),
        "errors": report.errors,
        "fallback_count": report.fallback_count,
        "glossary_replacements": report.glossary_replacements,
        "llm_calls": report.llm_calls,
        "llm_skipped": report.llm_skipped,
        "llm_safe_segments": report.llm_safe_segments,
        "llm_smart_segments": report.llm_smart_segments,
        "llm_chunks_built": report.llm_chunks_built,
        "avg_chunk_size": round(report.avg_chunk_size, 3),
        "max_chunk_size": report.max_chunk_size,
        "chunk_fallbacks": report.chunk_fallbacks,
        "segment_fallbacks": report.segment_fallbacks,
        "avg_llm_latency_per_segment": round(report.avg_llm_latency_per_segment, 6),
        "avg_llm_latency_per_chunk": round(report.avg_llm_latency_per_chunk, 6),
        "llm_calls_saved_by_chunking": report.llm_calls_saved_by_chunking,
        "validation_failures": report.validation_failures,
        "placeholder_mismatch_count": report.placeholder_mismatch_count,
        "glossary_placeholder_mismatch_count": report.glossary_placeholder_mismatch_count,
        "routing_reasons": report.routing_reasons,
        "chunk_boundary_reasons": report.chunk_boundary_reasons,
        "chunk_merge_reasons": report.chunk_merge_reasons,
        "routing_trace": report.routing_trace,
        "chunk_trace": report.chunk_trace,
    }


def summarize_reports(report_payloads: list[dict[str, object]]) -> dict[str, object]:
    total_segments = sum(int(item.get("segment_count", 0)) for item in report_payloads)
    total_llm_skipped = sum(int(item.get("llm_skipped", 0)) for item in report_payloads)
    total_safe = sum(int(item.get("llm_safe_segments", 0)) for item in report_payloads)
    total_smart = sum(int(item.get("llm_smart_segments", 0)) for item in report_payloads)
    total_chunk_fallbacks = sum(int(item.get("chunk_fallbacks", 0)) for item in report_payloads)
    total_segment_fallbacks = sum(int(item.get("segment_fallbacks", 0)) for item in report_payloads)
    total_saved = sum(int(item.get("llm_calls_saved_by_chunking", 0)) for item in report_payloads)
    avg_chunk = 0.0
    total_chunks = sum(int(item.get("llm_chunks_built", 0)) for item in report_payloads)
    if total_chunks:
        weighted = sum(float(item.get("avg_chunk_size", 0.0)) * int(item.get("llm_chunks_built", 0)) for item in report_payloads)
        avg_chunk = weighted / total_chunks

    validation_failures: dict[str, int] = {}
    for item in report_payloads:
        failures = item.get("validation_failures", {})
        if isinstance(failures, dict):
            for key, value in failures.items():
                validation_failures[str(key)] = validation_failures.get(str(key), 0) + int(value)

    return {
        "reports": len(report_payloads),
        "segments": total_segments,
        "llm_skip_rate": round(total_llm_skipped / max(1, total_segments), 4),
        "safe_share": round(total_safe / max(1, total_segments), 4),
        "smart_share": round(total_smart / max(1, total_segments), 4),
        "chunk_fallbacks": total_chunk_fallbacks,
        "segment_fallbacks": total_segment_fallbacks,
        "avg_chunk_size": round(avg_chunk, 3),
        "llm_calls_saved_by_chunking": total_saved,
        "validation_failures": dict(sorted(validation_failures.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def write_report_json(report: TranslationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")
