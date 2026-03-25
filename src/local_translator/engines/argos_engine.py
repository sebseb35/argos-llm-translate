from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


class ArgosEngine:
    """Thin adapter around argostranslate package APIs."""

    def __init__(self, source_lang: str, target_lang: str) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._translator = None

    def _load_translator(self):
        if self._translator is not None:
            return self._translator

        try:
            from argostranslate import translate
        except Exception as exc:  # pragma: no cover - import/runtime dependency
            raise RuntimeError(
                "argostranslate is required. Install package and language model first."
            ) from exc

        languages = translate.get_installed_languages()
        src = next((lang for lang in languages if lang.code == self.source_lang), None)
        if src is None:
            raise RuntimeError(f"Source language not installed: {self.source_lang}")
        tgt = next((lang for lang in languages if lang.code == self.target_lang), None)
        if tgt is None:
            raise RuntimeError(f"Target language not installed: {self.target_lang}")

        self._translator = src.get_translation(tgt)
        return self._translator

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        translator = self._load_translator()
        return translator.translate(text)
