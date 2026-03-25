from __future__ import annotations

from local_translator.engines.llm_engine import LLMEngine


def post_edit_segment(
    llm_engine: LLMEngine | None,
    source_segment: str,
    translated_segment: str,
    glossary: dict[str, str],
) -> str:
    if llm_engine is None:
        return translated_segment
    try:
        return llm_engine.post_edit(source_segment, translated_segment, glossary=glossary)
    except Exception:
        return translated_segment
