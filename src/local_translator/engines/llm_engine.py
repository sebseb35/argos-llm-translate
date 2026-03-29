from __future__ import annotations

import logging
import time
from typing import Literal

LOGGER = logging.getLogger(__name__)
PostEditMode = Literal["safe", "smart"]


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

    def _build_prompt(
        self,
        source: str,
        translated: str,
        glossary: dict[str, str] | None = None,
        mode: PostEditMode = "safe",
    ) -> str:
        glossary_lines = "\n".join(f"- {k} => {v}" for k, v in (glossary or {}).items()) or "(none)"
        mode_rules = {
            "safe": (
                "SAFE MODE:\n"
                "- Make minimal edits only (grammar, agreement, punctuation, light wording).\n"
                "- Prefer source-faithful phrasing over stylistic rewrites.\n"
            ),
            "smart": (
                "SMART MODE:\n"
                "- Improve fluency and readability while preserving exact meaning.\n"
                "- You may reorder clauses and tighten wording, but keep scope and facts unchanged.\n"
            ),
        }[mode]
        chunk_rule = ""
        if "[SEGMENT_" in translated:
            chunk_rule = (
                "8) Preserve all [SEGMENT_i]...[/SEGMENT_i] markers exactly.\n"
                "9) Return every segment block in the same order with no extra text.\n"
            )
        return (
            "System: You are a deterministic translation post-editor.\n"
            "Global rules:\n"
            "1) Keep meaning and scope identical to the draft.\n"
            "2) Do NOT add or remove information.\n"
            "3) Edit only [DRAFT_TRANSLATION]; do not rewrite [SOURCE].\n"
            "4) Preserve every placeholder matching __LT_[A-Z_0-9]+__ exactly.\n"
            "5) Keep numbers, URLs, code, commands, and identifiers unchanged.\n"
            "6) Keep glossary target terms exactly as written in [GLOSSARY].\n"
            "7) Return only the edited segment text, with no commentary.\n"
            f"{chunk_rule}"
            f"{mode_rules}\n"
            f"[SOURCE]\n{source}\n[/SOURCE]\n\n"
            f"[DRAFT_TRANSLATION]\n{translated}\n[/DRAFT_TRANSLATION]\n\n"
            f"[GLOSSARY]\n{glossary_lines}\n[/GLOSSARY]\n\n"
            "[OUTPUT]\n"
        )

    def _max_tokens_budget(self, draft: str, mode: PostEditMode) -> int:
        ratio = 0.55 if mode == "safe" else 0.85
        heuristic_budget = int(len(draft) * ratio) + 16
        return max(24, min(self.max_tokens, heuristic_budget))

    def post_edit(
        self,
        source: str,
        translated: str,
        glossary: dict[str, str] | None = None,
        mode: PostEditMode = "safe",
    ) -> str:
        if not translated.strip() or not self.model_path:
            return translated

        llm = self._load_model()
        prompt = self._build_prompt(source, translated, glossary, mode=mode)
        max_tokens = self._max_tokens_budget(translated, mode)
        LOGGER.debug("LLM prompt (%d chars, mode=%s)", len(prompt), mode)
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
            "LLM generation completed in %.3fs | mode=%s | max_tokens=%d | output_chars=%d",
            time.perf_counter() - started,
            mode,
            max_tokens,
            len(candidate),
        )
        return candidate
