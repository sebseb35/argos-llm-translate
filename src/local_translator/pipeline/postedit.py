from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

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
            r"```[\s\S]*?```",  # fenced code blocks
            r"`[^`\n]+`",  # inline code / commands
            r"https?://[^\s)\]>\"']+",  # URLs
            r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",  # emails
            r"\b(?:[A-Za-z]:\\\\|\\/)[^\s]+",  # Windows/Unix style paths
            r"\$\{[^}]+\}|\{\{[^}]+\}\}|%\([^)]+\)s",  # placeholders/templates
            r"\bv?\d+(?:\.\d+){1,3}(?:-[A-Za-z0-9]+)?\b",  # versions
            r"\b\d+(?:[.,]\d+)?%?\b",  # numeric values
            r"\b(?:[A-Z]{2,}[A-Z0-9_]*|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z_]*\d+[A-Za-z0-9_]*)\b",  # identifiers
            r"\b[A-Za-z][\w.-]*\s+(?:--?[\w-]+(?:[= ]\S+)\s*)+",  # CLI snippets
        ]
    ),
    flags=re.MULTILINE,
)


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
        if token in restored:
            restored = restored.replace(token, placeholder, 1)
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


def post_edit_segment_with_metrics(
    llm_engine: LLMEngine | None,
    source_segment: str,
    translated_segment: str,
    glossary: Glossary | dict[str, str],
    llm_settings: LLMSettings,
) -> PostEditOutcome:
    if llm_engine is None:
        return PostEditOutcome(text=translated_segment)

    protector = TokenProtector()
    glossary_protector = GlossaryProtector()
    validator = PostEditValidator()

    source_protected = protector.protect(source_segment)
    translated_protected = protector.protect(translated_segment)
    glossary_protected = glossary_protector.protect(translated_protected.text, glossary)
    llm_started = time.perf_counter()

    try:
        candidate_protected = llm_engine.post_edit(
            source_protected.text,
            glossary_protected.text,
            glossary=glossary.entries if isinstance(glossary, Glossary) else glossary,
        )
    except Exception:
        LOGGER.warning("LLM post-edit failed; falling back to Argos output.", exc_info=True)
        return PostEditOutcome(text=translated_segment, fallback_used=True)
    llm_elapsed = time.perf_counter() - llm_started
    combined_token_map = {**translated_protected.token_map, **glossary_protected.token_map}
    canonical_candidate = _canonicalize_candidate_placeholders(candidate_protected, combined_token_map)
    reinjected_candidate = _reinject_missing_placeholders(canonical_candidate, combined_token_map)
    if canonical_candidate != candidate_protected:
        LOGGER.debug(
            "Canonicalized placeholders before validation | raw=%r canonical=%r",
            candidate_protected[:300],
            canonical_candidate[:300],
        )
    if reinjected_candidate != canonical_candidate:
        LOGGER.debug(
            "Reinjected literal tokens as placeholders before validation | canonical=%r reinjected=%r",
            canonical_candidate[:300],
            reinjected_candidate[:300],
        )

    validation_started = time.perf_counter()
    validation_elapsed = 0.0
    if llm_settings.strict_validation:
        validation = validator.validate_protected_output(
            source_protected=source_protected.text,
            translated_protected=glossary_protected.text,
            candidate_protected=reinjected_candidate,
            token_map=combined_token_map,
            max_expansion_ratio=llm_settings.max_expansion_ratio,
        )
        if not validation.is_valid:
            LOGGER.warning("LLM post-edit rejected (%s).", validation.reason)
            if validation.reason in {"placeholder_mismatch", "glossary_placeholder_mismatch"}:
                _log_placeholder_diff(
                    reinjected_candidate,
                    combined_token_map,
                )
            if llm_settings.fallback_to_argos:
                return PostEditOutcome(text=translated_segment, fallback_used=True)
            return PostEditOutcome(text=reinjected_candidate)
        validation_elapsed = time.perf_counter() - validation_started

    restore_started = time.perf_counter()
    restored = protector.restore(reinjected_candidate, translated_protected.token_map)
    restored = glossary_protector.restore(restored, glossary_protected.token_map)
    if _PLACEHOLDER_RE.search(restored) or _GLOSSARY_PLACEHOLDER_RE.search(restored):
        LOGGER.warning("LLM output still contains unresolved placeholders; using Argos output.")
        if llm_settings.fallback_to_argos:
            return PostEditOutcome(text=translated_segment, fallback_used=True)
        return PostEditOutcome(text=restored)

    # Enforce glossary one more time after post-edit for deterministic behavior.
    rendered, replacements = apply_glossary_with_stats(restored, glossary)
    restore_elapsed = time.perf_counter() - restore_started
    LOGGER.debug(
        "Post-edit timings | llm=%.3fs validate=%.3fs restore=%.3fs",
        llm_elapsed,
        validation_elapsed,
        restore_elapsed,
    )
    return PostEditOutcome(text=rendered, glossary_replacements=replacements)


def post_edit_segment(
    llm_engine: LLMEngine | None,
    source_segment: str,
    translated_segment: str,
    glossary: Glossary | dict[str, str],
    llm_settings: LLMSettings,
) -> str:
    return post_edit_segment_with_metrics(
        llm_engine=llm_engine,
        source_segment=source_segment,
        translated_segment=translated_segment,
        glossary=glossary,
        llm_settings=llm_settings,
    ).text
