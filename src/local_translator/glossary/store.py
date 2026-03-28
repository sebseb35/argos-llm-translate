from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


class GlossaryError(ValueError):
    """Raised when glossary loading or validation fails."""


@dataclass(frozen=True, slots=True)
class Glossary:
    source_language: str | None
    target_language: str | None
    entries: dict[str, str]

    @classmethod
    def empty(cls) -> "Glossary":
        return cls(source_language=None, target_language=None, entries={})


def _read_payload(path: Path) -> dict:
    payload = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise GlossaryError("PyYAML is required to read YAML glossary files") from exc
        data = yaml.safe_load(payload)
    elif suffix == ".json":
        data = json.loads(payload)
    else:
        raise GlossaryError(f"Unsupported glossary format: {path}. Use .yaml/.yml/.json")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise GlossaryError("Glossary file must contain an object at top level")
    return data


def _normalize_entries(raw_entries: object) -> dict[str, str]:
    if not isinstance(raw_entries, dict):
        raise GlossaryError("Glossary 'entries' must be an object of source->target mappings")

    normalized: dict[str, str] = {}
    for raw_source, raw_target in raw_entries.items():
        source = str(raw_source).strip()
        target = str(raw_target).strip()
        if not source:
            raise GlossaryError("Glossary entry contains an empty source term")
        if not target:
            raise GlossaryError(f"Glossary entry '{raw_source}' has an empty target term")
        normalized[source] = target

    return normalized


def _parse_glossary_object(data: dict) -> Glossary:
    # Backward-compatible format: a plain key-value object.
    if "entries" not in data:
        return Glossary(source_language=None, target_language=None, entries=_normalize_entries(data))

    source_language = data.get("source_language")
    target_language = data.get("target_language")

    if source_language is not None and not isinstance(source_language, str):
        raise GlossaryError("'source_language' must be a string when provided")
    if target_language is not None and not isinstance(target_language, str):
        raise GlossaryError("'target_language' must be a string when provided")

    entries = _normalize_entries(data.get("entries"))
    return Glossary(
        source_language=source_language.lower() if isinstance(source_language, str) else None,
        target_language=target_language.lower() if isinstance(target_language, str) else None,
        entries=entries,
    )


def load_glossary(path: Path | None, source_lang: str | None = None, target_lang: str | None = None) -> Glossary:
    if path is None:
        return Glossary.empty()

    data = _read_payload(path)
    glossary = _parse_glossary_object(data)

    if source_lang and glossary.source_language and glossary.source_language != source_lang.lower():
        raise GlossaryError(
            f"Glossary source language mismatch: glossary={glossary.source_language}, requested={source_lang.lower()}"
        )
    if target_lang and glossary.target_language and glossary.target_language != target_lang.lower():
        raise GlossaryError(
            f"Glossary target language mismatch: glossary={glossary.target_language}, requested={target_lang.lower()}"
        )

    return glossary


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    # Use word boundaries for terms containing word chars to avoid replacing inside other words.
    if re.search(r"\w", term):
        return re.compile(rf"(?<!\w){escaped}(?!\w)")
    return re.compile(escaped)


def apply_glossary_with_stats(text: str, glossary: Glossary | dict[str, str]) -> tuple[str, int]:
    if isinstance(glossary, Glossary):
        entries = glossary.entries
    else:
        entries = glossary

    if not entries:
        return text, 0

    ordered_terms = sorted(entries.keys(), key=lambda term: (-len(term), term))
    out = text
    placeholders: dict[str, str] = {}
    replacements = 0

    for idx, source in enumerate(ordered_terms):
        target = entries[source]
        token = f"__LT_GLOSSARY_REPLACE_{idx:04d}__"
        pattern = _term_pattern(source)

        def _replace(_: re.Match[str]) -> str:
            nonlocal replacements
            placeholders[token] = target
            replacements += 1
            return token

        out = pattern.sub(_replace, out)

    for token, target in placeholders.items():
        out = out.replace(token, target)
    return out, replacements


def apply_glossary(text: str, glossary: Glossary | dict[str, str]) -> str:
    rendered, _ = apply_glossary_with_stats(text, glossary)
    return rendered
