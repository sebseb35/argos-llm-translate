from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from local_translator.config import LLMSettings
from local_translator.engines.llm_engine import LLMEngine

LOGGER = logging.getLogger(__name__)
_PLACEHOLDER_PREFIX = "__LT_PROTECTED_"
_PLACEHOLDER_RE = re.compile(r"__LT_PROTECTED_(\d{4})__")

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

        expected = sorted(token_map.keys())
        actual = sorted(_PLACEHOLDER_RE.findall(candidate_protected))
        expected_ids = sorted(_PLACEHOLDER_RE.findall(" ".join(expected)))
        if actual != expected_ids:
            return ValidationResult(False, "placeholder_mismatch")

        if candidate_protected.count("\n") > source_protected.count("\n") + 3:
            return ValidationResult(False, "unexpected_reformat")

        return ValidationResult(True)


def post_edit_segment(
    llm_engine: LLMEngine | None,
    source_segment: str,
    translated_segment: str,
    glossary: dict[str, str],
    llm_settings: LLMSettings,
) -> str:
    if llm_engine is None:
        return translated_segment

    protector = TokenProtector()
    validator = PostEditValidator()

    source_protected = protector.protect(source_segment)
    translated_protected = protector.protect(translated_segment)

    try:
        candidate_protected = llm_engine.post_edit(
            source_protected.text,
            translated_protected.text,
            glossary=glossary,
        )
    except Exception:
        LOGGER.warning("LLM post-edit failed; falling back to Argos output.", exc_info=True)
        return translated_segment

    if llm_settings.strict_validation:
        validation = validator.validate_protected_output(
            source_protected=source_protected.text,
            translated_protected=translated_protected.text,
            candidate_protected=candidate_protected,
            token_map=translated_protected.token_map,
            max_expansion_ratio=llm_settings.max_expansion_ratio,
        )
        if not validation.is_valid:
            LOGGER.warning("LLM post-edit rejected (%s).", validation.reason)
            return translated_segment if llm_settings.fallback_to_argos else candidate_protected

    restored = protector.restore(candidate_protected, translated_protected.token_map)
    if _PLACEHOLDER_RE.search(restored):
        LOGGER.warning("LLM output still contains unresolved placeholders; using Argos output.")
        return translated_segment if llm_settings.fallback_to_argos else restored
    return restored
