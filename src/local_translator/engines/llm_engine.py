from __future__ import annotations

import re


class LLMEngine:
    """Constrained local post-editor using llama-cpp-python if configured."""

    def __init__(self, model_path: str | None, temperature: float = 0.1, max_tokens: int = 256):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _load_model(self):
        if self._llm is not None:
            return self._llm
        if not self.model_path:
            raise RuntimeError("LLM model path is required for hybrid/llm modes.")
        try:
            from llama_cpp import Llama
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("llama-cpp-python is not installed.") from exc
        self._llm = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)
        return self._llm

    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        if not translated.strip() or not self.model_path:
            return translated

        llm = self._load_model()
        glossary_lines = "\n".join(f"- {k} => {v}" for k, v in (glossary or {}).items())
        prompt = (
            "You are a strict post-editor. Keep meaning identical. Do not add/remove facts. "
            "Preserve numbers, identifiers, paths, URLs, commands, code. Return only edited text.\n"
            f"Source:\n{source}\n\nDraft translation:\n{translated}\n\nGlossary:\n{glossary_lines}\n"
        )
        out = llm(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        candidate = out["choices"][0]["text"].strip()
        if not candidate:
            return translated

        # Basic guardrail: if too divergent in number tokens, fallback.
        if len(re.findall(r"\d+", candidate)) != len(re.findall(r"\d+", translated)):
            return translated
        return candidate
