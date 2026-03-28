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


def _term_pattern(term: str, ignore_case: bool = False) -> re.Pattern[str]:
    escaped = re.escape(term)
    flags = re.IGNORECASE if ignore_case else 0
    # Use word boundaries for terms containing word chars to avoid replacing inside other words.
    if re.search(r"\w", term):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=flags)
    return re.compile(escaped, flags=flags)


def _case_aware_target(source_match: str, target: str) -> str:
    if not source_match:
        return target
    if source_match.isupper():
        return target.upper()
    if source_match[0].isupper() and source_match[1:].islower():
        return target[:1].upper() + target[1:]
    return target


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
        pattern = _term_pattern(source, ignore_case=True)

        def _replace(match: re.Match[str]) -> str:
            nonlocal replacements
            placeholders[token] = _case_aware_target(match.group(0), target)
            replacements += 1
            return token

        out = pattern.sub(_replace, out)

    for token, target in placeholders.items():
        out = out.replace(token, target)
    return out, replacements


def apply_glossary(text: str, glossary: Glossary | dict[str, str]) -> str:
    rendered, _ = apply_glossary_with_stats(text, glossary)
    return rendered


def protect_glossary_terms_with_stats(text: str, glossary: Glossary | dict[str, str]) -> tuple[str, dict[str, str], int]:
    if isinstance(glossary, Glossary):
        entries = glossary.entries
    else:
        entries = glossary

    if not entries:
        return text, {}, 0

    ordered_terms = sorted(entries.keys(), key=lambda term: (-len(term), term))
    protected = text
    token_map: dict[str, str] = {}
    replacements = 0

    for idx, source in enumerate(ordered_terms):
        token = f"__LT_GLOSSARY_TERM_{idx:04d}__"
        target = entries[source]
        pattern = _term_pattern(source, ignore_case=True)

        def _replace(match: re.Match[str]) -> str:
            nonlocal replacements
            token_map[token] = _case_aware_target(match.group(0), target)
            replacements += 1
            return token

        protected = pattern.sub(_replace, protected)
    return protected, token_map, replacements


def restore_glossary_terms_with_stats(text: str, token_map: dict[str, str]) -> tuple[str, int]:
    if not token_map:
        return text, 0

    restored = text
    replacements = 0
    for token, target in token_map.items():
        variants = _glossary_placeholder_variants(token)
        for variant in variants:
            matches = variant.findall(restored)
            if not matches:
                continue
            replacements += len(matches)
            restored = variant.sub(target, restored)
    return restored, replacements


def normalize_restored_text(text: str) -> str:
    # Collapse repeated inline spacing introduced by placeholder restoration while
    # preserving explicit line breaks and paragraph boundaries.
    lines = text.splitlines(keepends=True)
    normalized_lines: list[str] = []

    for line in lines:
        if line.endswith("\r\n"):
            content, newline = line[:-2], "\r\n"
        elif line.endswith("\n"):
            content, newline = line[:-1], "\n"
        else:
            content, newline = line, ""

        content = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", content)
        normalized_lines.append(content + newline)

    return "".join(normalized_lines)


def _glossary_placeholder_variants(token: str) -> tuple[re.Pattern[str], ...]:
    normalized = token.strip("_")
    parts = [part for part in normalized.split("_") if part]
    if not parts:
        return (re.compile(re.escape(token)),)

    exact = re.compile(re.escape(token))
    # Some translators normalize internal placeholders by stripping underscores and
    # turning separators into spaces, e.g. "__LT_GLOSSARY_TERM_0000__" becomes
    # "LT GLOSSARY TERM 0000". Accept tightly scoped delimiter variations so the
    # restoration step can still recover the protected glossary term.
    normalized_pattern = re.compile(
        rf"(?<!\w)_*{r'[\W_]*'.join(re.escape(part) for part in parts)}_*(?!\w)",
        flags=re.IGNORECASE,
    )
    return exact, normalized_pattern
