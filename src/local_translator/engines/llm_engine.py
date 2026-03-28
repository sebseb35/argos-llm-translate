from __future__ import annotations


class LLMEngine:
    """Constrained local post-editor using llama-cpp-python if configured."""

    def __init__(self, model_path: str | None, temperature: float = 0.0, max_tokens: int = 256):
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
            "System: You are a deterministic translation post-editor.\n"
            "Rules:\n"
            "1) Keep meaning and scope identical to the draft.\n"
            "2) Do NOT add or remove information.\n"
            "3) Keep protected placeholders/tokens unchanged.\n"
            "4) Keep numbers, URLs, code, commands, identifiers unchanged.\n"
            "5) Output only the edited segment text, no commentary.\n\n"
            f"[SOURCE]\n{source}\n[/SOURCE]\n\n"
            f"[DRAFT_TRANSLATION]\n{translated}\n[/DRAFT_TRANSLATION]\n\n"
            f"[GLOSSARY]\n{glossary_lines}\n[/GLOSSARY]\n\n"
            "[OUTPUT]\n"
        )
        out = llm(
            prompt,
            temperature=min(self.temperature, 0.1),
            top_k=1,
            top_p=1.0,
            seed=0,
            max_tokens=self.max_tokens,
            stop=["[/OUTPUT]"],
        )
        return out["choices"][0]["text"].strip()
