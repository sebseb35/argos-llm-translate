from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from local_translator.config import LLMSettings
from local_translator.engines.llm_engine import LLMEngine
from local_translator.glossary.store import Glossary, apply_glossary_with_stats

LOGGER = logging.getLogger(__name__)
_PLACEHOLDER_PREFIX = "__LT_PROTECTED_"
_PLACEHOLDER_RE = re.compile(r"__LT_PROTECTED_(\d{4})__")
_GLOSSARY_PLACEHOLDER_PREFIX = "__LT_GLOSSARY_PROTECTED_"
_GLOSSARY_PLACEHOLDER_RE = re.compile(r"__LT_GLOSSARY_PROTECTED_(\d{4})__")

# Pragmatic set of technical tokens we should keep untouched during post-editing.
_PROTECTED_TOKEN_RE = re.compile(
    "|".join(
        [
            r"__LT_[A-Z0-9_]+__",  # internal pipeline placeholders
            r"```[\s\S]*?```",  # fenced code blocks
            r"`[^`\n]+`",  # inline code / commands
            r"https?://[^\s)\]>\"']+",  # URLs
            r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",  # emails
            r"\b(?:[A-Za-z]:\\\\|\\/)[^\s]+",  # Windows/Unix style paths
            r"\$\{[^}]+\}|\{\{[^}]+\}\}|%\([^)]+\)s",  # placeholders/templates
            r"\bv?\d+(?:\.\d+){1,3}(?:-[A-Za-z0-9]+)?\b",  # versions
            r"\b\d+(?:[.,]\d+)?%?\b",  # numeric values
            r"\b(?:[A-Za-z_]+[A-Za-z0-9_]*_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z_]+\d+[A-Za-z0-9_]*)\b",  # identifiers
            r"\b[A-Za-z][\w.-]*\s+(?:--?[\w-]+(?:[= ]\S+)\s*)+",  # CLI snippets
        ]
    ),
    flags=re.MULTILINE,
)
_ANY_PLACEHOLDER_RE = re.compile(r"__LT_[A-Z0-9_]+__")
_SEGMENT_BLOCK_RE = re.compile(r"\[SEGMENT_(\d+)\]\n(.*?)\n\[/SEGMENT_\1\]", flags=re.DOTALL)


@dataclass(slots=True)
class ProtectedText:
    text: str
    token_map: dict[str, str]


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    reason: str | None = None


@dataclass(slots=True)
class PostEditOutcome:
    text: str
    fallback_used: bool = False
    glossary_replacements: int = 0
    failure_reason: str | None = None


class TokenProtector:
    """Protect and restore technical tokens using placeholders."""

    def protect(self, text: str) -> ProtectedText:
        token_map: dict[str, str] = {}
        parts: list[str] = []
        cursor = 0

        for idx, match in enumerate(_PROTECTED_TOKEN_RE.finditer(text)):
            placeholder = f"{_PLACEHOLDER_PREFIX}{idx:04d}__"
            token_map[placeholder] = match.group(0)
            parts.append(text[cursor : match.start()])
            parts.append(placeholder)
            cursor = match.end()

        parts.append(text[cursor:])
        return ProtectedText(text="".join(parts), token_map=token_map)

    def restore(self, text: str, token_map: dict[str, str]) -> str:
        restored = text
        for placeholder, token in token_map.items():
            restored = restored.replace(placeholder, token)
        return restored


class GlossaryProtector:
    """Protect glossary-enforced target phrases so post-editing cannot rewrite them."""

    def protect(self, text: str, glossary: Glossary | dict[str, str]) -> ProtectedText:
        entries = glossary.entries if isinstance(glossary, Glossary) else glossary
        if not entries:
            return ProtectedText(text=text, token_map={})

        protected_text = text
        token_map: dict[str, str] = {}
        ordered_targets = sorted(set(entries.values()), key=lambda term: (-len(term), term))

        for idx, target_term in enumerate(ordered_targets):
            escaped = re.escape(target_term)
            if re.search(r"\w", target_term):
                pattern = re.compile(rf"(?<!\w){escaped}(?!\w)")
            else:
                pattern = re.compile(escaped)

            placeholder = f"{_GLOSSARY_PLACEHOLDER_PREFIX}{idx:04d}__"
            if pattern.search(protected_text):
                token_map[placeholder] = target_term
                protected_text = pattern.sub(placeholder, protected_text)

        return ProtectedText(text=protected_text, token_map=token_map)

    def restore(self, text: str, token_map: dict[str, str]) -> str:
        restored = text
        for placeholder, token in token_map.items():
            restored = restored.replace(placeholder, token)
        return restored


class PostEditValidator:
    def validate_protected_output(
        self,
        source_protected: str,
        translated_protected: str,
        candidate_protected: str,
        token_map: dict[str, str],
        max_expansion_ratio: float,
    ) -> ValidationResult:
        if translated_protected.strip() and not candidate_protected.strip():
            return ValidationResult(False, "empty_output")

        if translated_protected and len(candidate_protected) > int(len(translated_protected) * max_expansion_ratio):
            return ValidationResult(False, "length_expansion")

        expected_ids = sorted(_PLACEHOLDER_RE.findall(" ".join(token_map.keys())))
        actual_ids = sorted(_PLACEHOLDER_RE.findall(candidate_protected))
        if actual_ids != expected_ids:
            return ValidationResult(False, "placeholder_mismatch")

        glossary_expected = sorted(_GLOSSARY_PLACEHOLDER_RE.findall(" ".join(token_map.keys())))
        glossary_actual = sorted(_GLOSSARY_PLACEHOLDER_RE.findall(candidate_protected))
        if glossary_actual != glossary_expected:
            return ValidationResult(False, "glossary_placeholder_mismatch")

        if candidate_protected.count("\n") > source_protected.count("\n") + 3:
            return ValidationResult(False, "unexpected_reformat")

        return ValidationResult(True)


def _placeholder_ids(text: str, pattern: re.Pattern[str]) -> list[str]:
    return sorted(pattern.findall(text))


def _placeholder_variants(token: str) -> tuple[re.Pattern[str], ...]:
    normalized = token.strip("_")
    parts = [part for part in normalized.split("_") if part]
    if not parts:
        return (re.compile(re.escape(token)),)

    exact = re.compile(re.escape(token))
    normalized_pattern = re.compile(
        rf"(?<!\w)_*{r'[\W_]*'.join(re.escape(part) for part in parts)}_*(?!\w)",
        flags=re.IGNORECASE,
    )
    return exact, normalized_pattern


def _canonicalize_candidate_placeholders(candidate_protected: str, token_map: dict[str, str]) -> str:
    canonical = candidate_protected
    for placeholder in sorted(token_map.keys(), key=lambda item: (-len(item), item)):
        for variant in _placeholder_variants(placeholder):
            canonical = variant.sub(placeholder, canonical)
    return canonical


def _replace_once_outside_placeholders(text: str, needle: str, replacement: str) -> tuple[str, bool]:
    if not needle:
        return text, False

    def _needle_pattern(raw: str) -> re.Pattern[str]:
        escaped = re.escape(raw)
        # Accept whitespace normalization drift for multiword literals, e.g.
        # "acceptance testing" vs "acceptance   testing".
        escaped = escaped.replace(r"\ ", r"\s+")
        if re.search(r"\w", raw):
            return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
        return re.compile(escaped, flags=re.IGNORECASE)

    last = 0
    out_parts: list[str] = []
    replaced = False
    needle_re = _needle_pattern(needle)

    for match in _ANY_PLACEHOLDER_RE.finditer(text):
        segment = text[last : match.start()]
        if not replaced:
            segment, count = needle_re.subn(replacement, segment, count=1)
            replaced = count > 0
        out_parts.append(segment)
        out_parts.append(match.group(0))
        last = match.end()

    tail = text[last:]
    if not replaced:
        tail, count = needle_re.subn(replacement, tail, count=1)
        replaced = count > 0
    out_parts.append(tail)
    return "".join(out_parts), replaced


def _reinject_missing_placeholders(candidate_protected: str, token_map: dict[str, str]) -> str:
    """Reinsert expected placeholders when the LLM echoed the exact token literal.

    This keeps strict validation while tolerating a common behavior where the model
    outputs literal protected content (e.g. "2") instead of "__LT_PROTECTED_0000__".
    """
    restored = candidate_protected
    expected = set(_PLACEHOLDER_RE.findall(" ".join(token_map.keys()))) | set(
        _GLOSSARY_PLACEHOLDER_RE.findall(" ".join(token_map.keys()))
    )
    if not expected:
        return restored

    for placeholder, token in sorted(token_map.items(), key=lambda item: (-len(item[1]), item[0])):
        if placeholder in restored or not token:
            continue
        restored, _ = _replace_once_outside_placeholders(restored, token, placeholder)
    return restored


def _log_placeholder_diff(candidate_protected: str, token_map: dict[str, str]) -> None:
    expected_protected = _placeholder_ids(" ".join(token_map.keys()), _PLACEHOLDER_RE)
    actual_protected = _placeholder_ids(candidate_protected, _PLACEHOLDER_RE)
    expected_glossary = _placeholder_ids(" ".join(token_map.keys()), _GLOSSARY_PLACEHOLDER_RE)
    actual_glossary = _placeholder_ids(candidate_protected, _GLOSSARY_PLACEHOLDER_RE)

    expected_protected_set = set(expected_protected)
    actual_protected_set = set(actual_protected)
    expected_glossary_set = set(expected_glossary)
    actual_glossary_set = set(actual_glossary)

    LOGGER.debug(
        (
            "Placeholder diff | protected missing=%s extra=%s | glossary missing=%s extra=%s | "
            "candidate_preview=%r"
        ),
        sorted(expected_protected_set - actual_protected_set),
        sorted(actual_protected_set - expected_protected_set),
        sorted(expected_glossary_set - actual_glossary_set),
        sorted(actual_glossary_set - expected_glossary_set),
        candidate_protected[:300],
    )


def format_chunk_payload(
    segment_indices: list[int],
    source_segments: list[str],
    draft_segments: list[str],
) -> tuple[str, str]:
    source_parts: list[str] = []
    draft_parts: list[str] = []
    for rel_idx, seg_idx in enumerate(segment_indices):
        source_parts.append(f"[SEGMENT_{rel_idx}]\n{source_segments[seg_idx]}\n[/SEGMENT_{rel_idx}]")
        draft_parts.append(f"[SEGMENT_{rel_idx}]\n{draft_segments[seg_idx]}\n[/SEGMENT_{rel_idx}]")
    return "\n\n".join(source_parts), "\n\n".join(draft_parts)


def parse_chunk_output(candidate: str, expected_segments: int) -> tuple[list[str] | None, str | None]:
    blocks = list(_SEGMENT_BLOCK_RE.finditer(candidate))
    if len(blocks) != expected_segments:
        return None, "missing_segment"

    consumed: list[str] = []
    extracted: list[str] = []
    cursor = 0
    for expected_idx, block in enumerate(blocks):
        if int(block.group(1)) != expected_idx:
            return None, "malformed_markers"
        interstitial = candidate[cursor : block.start()]
        if interstitial.strip():
            return None, "malformed_markers"
        consumed.append(block.group(0))
        extracted.append(block.group(2).strip())
        cursor = block.end()
    if candidate[cursor:].strip():
        return None, "malformed_markers"
    return extracted, None


def apply_postedit_candidate(
    source_segment: str,
    translated_segment: str,
    candidate_protected: str,
    glossary: Glossary | dict[str, str],
    llm_settings: LLMSettings,
) -> PostEditOutcome:
    protector = TokenProtector()
    glossary_protector = GlossaryProtector()
    validator = PostEditValidator()

    source_protected = protector.protect(source_segment)
    translated_protected = protector.protect(translated_segment)
    glossary_protected = glossary_protector.protect(translated_protected.text, glossary)
    combined_token_map = {**translated_protected.token_map, **glossary_protected.token_map}
    canonical_candidate = _canonicalize_candidate_placeholders(candidate_protected, combined_token_map)
    reinjected_candidate = _reinject_missing_placeholders(canonical_candidate, combined_token_map)

    if llm_settings.strict_validation:
        validation = validator.validate_protected_output(
            source_protected=source_protected.text,
            translated_protected=glossary_protected.text,
            candidate_protected=reinjected_candidate,
            token_map=combined_token_map,
            max_expansion_ratio=llm_settings.max_expansion_ratio,
        )
        if not validation.is_valid:
            if validation.reason in {"placeholder_mismatch", "glossary_placeholder_mismatch"}:
                _log_placeholder_diff(reinjected_candidate, combined_token_map)
            if llm_settings.fallback_to_argos:
                return PostEditOutcome(
                    text=translated_segment,
                    fallback_used=True,
                    failure_reason=validation.reason,
                )
            return PostEditOutcome(text=reinjected_candidate, failure_reason=validation.reason)

    restored = protector.restore(reinjected_candidate, translated_protected.token_map)
    restored = glossary_protector.restore(restored, glossary_protected.token_map)
    if _PLACEHOLDER_RE.search(restored) or _GLOSSARY_PLACEHOLDER_RE.search(restored):
        if llm_settings.fallback_to_argos:
            return PostEditOutcome(
                text=translated_segment,
                fallback_used=True,
                failure_reason="unresolved_placeholders",
            )
        return PostEditOutcome(text=restored, failure_reason="unresolved_placeholders")

    rendered, replacements = apply_glossary_with_stats(restored, glossary)
    return PostEditOutcome(text=rendered, glossary_replacements=replacements)


def post_edit_segment_with_metrics(
    llm_engine: LLMEngine | None,
    source_segment: str,
    translated_segment: str,
    glossary: Glossary | dict[str, str],
    llm_settings: LLMSettings,
    mode: Literal["safe", "smart"] = "safe",
) -> PostEditOutcome:
    if llm_engine is None:
        return PostEditOutcome(text=translated_segment)

    protector = TokenProtector()
    glossary_protector = GlossaryProtector()
    source_protected = protector.protect(source_segment)
    translated_protected = protector.protect(translated_segment)
    glossary_protected = glossary_protector.protect(translated_protected.text, glossary)
    llm_started = time.perf_counter()

    try:
        candidate_protected = llm_engine.post_edit(
            source_protected.text,
            glossary_protected.text,
            glossary=glossary.entries if isinstance(glossary, Glossary) else glossary,
            mode=mode,
        )
    except Exception:
        LOGGER.warning("LLM post-edit failed; falling back to Argos output.", exc_info=True)
        return PostEditOutcome(text=translated_segment, fallback_used=True, failure_reason="llm_exception")
    llm_elapsed = time.perf_counter() - llm_started
    validate_started = time.perf_counter()
    outcome = apply_postedit_candidate(
        source_segment=source_segment,
        translated_segment=translated_segment,
        candidate_protected=candidate_protected,
        glossary=glossary,
        llm_settings=llm_settings,
    )
    validate_elapsed = time.perf_counter() - validate_started
    LOGGER.debug(
        "Post-edit timings | llm=%.3fs validate+restore=%.3fs fallback=%s reason=%s",
        llm_elapsed,
        validate_elapsed,
        outcome.fallback_used,
        outcome.failure_reason,
    )
    return outcome


def post_edit_segment(
    llm_engine: LLMEngine | None,
    source_segment: str,
    translated_segment: str,
    glossary: Glossary | dict[str, str],
    llm_settings: LLMSettings,
    mode: Literal["safe", "smart"] = "safe",
) -> str:
    return post_edit_segment_with_metrics(
        llm_engine=llm_engine,
        source_segment=source_segment,
        translated_segment=translated_segment,
        glossary=glossary,
        llm_settings=llm_settings,
        mode=mode,
    ).text
