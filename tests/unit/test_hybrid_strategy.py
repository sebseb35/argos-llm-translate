from local_translator.config import LLMSettings
from local_translator.pipeline.hybrid_strategy import build_llm_chunks, decide_llm_postedit


def test_decide_llm_skips_short_segments_by_default():
    decision = decide_llm_postedit("Bonjour", "Hello", {}, LLMSettings())
    assert not decision.use_llm
    assert decision.reason == "short_segment"


def test_decide_llm_selects_smart_for_long_multisentence_text():
    translated = "This is the first sentence. This is the second sentence with more context and detail."
    decision = decide_llm_postedit("src", translated, {}, LLMSettings(skip_short_characters=5, smart_min_chars=40))
    assert decision.use_llm
    assert decision.mode == "smart"


def test_build_llm_chunks_groups_contiguous_segments():
    chunks = build_llm_chunks(["a" * 30, "b" * 30, "c" * 80], target_chars=60, max_chars=100)
    assert [(c.start_index, c.end_index) for c in chunks] == [(0, 1), (2, 2)]
