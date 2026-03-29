from __future__ import annotations

import json
from pathlib import Path

from local_translator.models.types import TranslationReport


def format_report(report: TranslationReport) -> str:
    lines = [
        "Execution report",
        f"Segments processed: {report.segment_count}",
        f"Fallbacks: {report.fallback_count}",
        f"Errors: {len(report.errors)}",
        f"Glossary replacements: {report.glossary_replacements}",
        f"LLM calls: {report.llm_calls}",
        f"LLM skipped: {report.llm_skipped}",
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
    }


def write_report_json(report: TranslationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")
