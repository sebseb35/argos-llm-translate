from __future__ import annotations

import logging
import time

LOGGER = logging.getLogger(__name__)


class LLMEngine:
    """Constrained local post-editor using llama-cpp-python if configured."""

    def __init__(
        self,
        model_path: str | None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_batch: int = 256,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
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
        started = time.perf_counter()
        kwargs: dict[str, int | str | bool] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        self._llm = Llama(**kwargs)
        LOGGER.debug(
            "LLM model loaded in %.3fs | n_ctx=%d n_batch=%d n_threads=%s",
            time.perf_counter() - started,
            self.n_ctx,
            self.n_batch,
            self.n_threads,
        )
        return self._llm

    def _build_prompt(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        glossary_lines = "\n".join(f"- {k} => {v}" for k, v in (glossary or {}).items()) or "(none)"
        return (
            "System: You are a deterministic translation post-editor.\n"
            "Rules:\n"
            "1) Keep meaning and scope identical to the draft.\n"
            "2) Do NOT add or remove information.\n"
            "2b) Edit only [DRAFT_TRANSLATION]; do not rewrite [SOURCE].\n"
            "3) Preserve every placeholder matching __LT_[A-Z_0-9]+__ exactly, character-for-character.\n"
            "4) Keep numbers, URLs, code, commands, and identifiers unchanged.\n"
            "5) Return only the edited segment text, with no commentary, no quotes, and no prefix.\n"
            "6) If no edits are needed, return the draft unchanged.\n\n"
            f"[SOURCE]\n{source}\n[/SOURCE]\n\n"
            f"[DRAFT_TRANSLATION]\n{translated}\n[/DRAFT_TRANSLATION]\n\n"
            f"[GLOSSARY]\n{glossary_lines}\n[/GLOSSARY]\n\n"
            "[OUTPUT]\n"
        )

    def _max_tokens_budget(self, draft: str) -> int:
        # Limit short-segment latency while keeping headroom on longer drafts.
        heuristic_budget = int(len(draft) * 0.75) + 16
        return max(32, min(self.max_tokens, heuristic_budget))

    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        if not translated.strip() or not self.model_path:
            return translated

        llm = self._load_model()
        prompt = self._build_prompt(source, translated, glossary)
        max_tokens = self._max_tokens_budget(translated)
        LOGGER.debug("LLM prompt (%d chars): %s", len(prompt), prompt)
        started = time.perf_counter()
        out = llm(
            prompt,
            temperature=max(0.0, self.temperature),
            top_k=1,
            top_p=1.0,
            seed=0,
            max_tokens=max_tokens,
            stop=["[/OUTPUT]", "\n\n"],
        )
        candidate = out["choices"][0]["text"].strip()
        LOGGER.debug(
            "LLM generation completed in %.3fs | max_tokens=%d | output_chars=%d | output=%s",
            time.perf_counter() - started,
            max_tokens,
            len(candidate),
            candidate,
        )
        return candidate
