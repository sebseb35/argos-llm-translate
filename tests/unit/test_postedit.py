from local_translator.config import LLMSettings, RuntimeConfig
from local_translator.models.types import EngineMode
from local_translator.pipeline.postedit import TokenProtector, post_edit_segment
from local_translator.pipeline.translator import TranslationPipeline


class FakeLLM:
    def __init__(self, response: str = "", error: Exception | None = None):
        self.response = response
        self.error = error

    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        if self.error:
            raise self.error
        return self.response


class DummyArgos:
    def translate(self, text: str) -> str:
        return f"ARGOS:{text}"


def test_token_protection_handles_technical_tokens():
    protector = TokenProtector()
    original = (
        "Visit https://example.com/v1 and run `python app.py --port 8080` for build v2.3.4 "
        "with user_id=42 and ${ENV}."
    )
    protected = protector.protect(original)
    restored = protector.restore(protected.text, protected.token_map)

    assert restored == original
    assert "__LT_PROTECTED_" in protected.text


def test_post_edit_falls_back_on_llm_exception():
    draft = "Translated with URL https://example.com and value 12"
    result = post_edit_segment(
        llm_engine=FakeLLM(error=RuntimeError("boom")),
        source_segment="source",
        translated_segment=draft,
        glossary={},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True),
    )
    assert result == draft


def test_post_edit_falls_back_on_empty_output():
    draft = "Result 99"
    result = post_edit_segment(
        llm_engine=FakeLLM(response=""),
        source_segment="source",
        translated_segment=draft,
        glossary={},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True),
    )
    assert result == draft


def test_post_edit_rejects_placeholder_tampering():
    draft = "Keep https://example.com and v1.2.3"
    result = post_edit_segment(
        llm_engine=FakeLLM(response="Tampered __LT_PROTECTED_9999__"),
        source_segment="source",
        translated_segment=draft,
        glossary={},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True),
    )
    assert result == draft


def test_post_edit_rejects_length_explosion():
    draft = "short draft"
    result = post_edit_segment(
        llm_engine=FakeLLM(response="x" * 1000),
        source_segment="source",
        translated_segment=draft,
        glossary={},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True, max_expansion_ratio=1.2),
    )
    assert result == draft


def test_post_edit_accepts_valid_refinement_and_restores_tokens():
    draft = "Install from https://example.com for version v1.2.3 and value 15"
    protected_draft = TokenProtector().protect(draft).text
    candidate = protected_draft.replace("Install", "Please install")
    result = post_edit_segment(
        llm_engine=FakeLLM(response=candidate),
        source_segment="source",
        translated_segment=draft,
        glossary={},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True),
    )
    assert "https://example.com" in result
    assert "v1.2.3" in result
    assert "15" in result
    assert "Please install" in result


def test_strict_validation_is_configurable():
    draft = "Keep https://example.com"
    response = "Expanded explanation. " * 20
    strict_result = post_edit_segment(
        llm_engine=FakeLLM(response=response),
        source_segment="source",
        translated_segment=draft,
        glossary={},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True, max_expansion_ratio=1.2),
    )
    relaxed_result = post_edit_segment(
        llm_engine=FakeLLM(response=response),
        source_segment="source",
        translated_segment=draft,
        glossary={},
        llm_settings=LLMSettings(strict_validation=False, fallback_to_argos=True, max_expansion_ratio=1.2),
    )
    assert strict_result == draft
    assert relaxed_result == response


def test_pipeline_hybrid_can_disable_llm_post_editing(monkeypatch):
    cfg = RuntimeConfig(
        source_lang="fr",
        target_lang="en",
        engine_mode=EngineMode.HYBRID,
        llm=LLMSettings(enabled=False),
    )
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", DummyArgos())

    result = pipeline.translate_text("bonjour")
    assert result.text == "ARGOS:bonjour"


def test_post_edit_rejects_glossary_tampering_in_strict_mode():
    draft = "Use production deployment for this release"
    result = post_edit_segment(
        llm_engine=FakeLLM(response="Use rollout for this release"),
        source_segment="Utiliser mise en production pour cette version",
        translated_segment=draft,
        glossary={"mise en production": "production deployment"},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True),
    )
    assert result == draft


def test_post_edit_restores_glossary_placeholders_when_valid():
    draft = "Use production deployment and password"
    candidate = "Please use __LT_GLOSSARY_PROTECTED_0001__ and __LT_GLOSSARY_PROTECTED_0000__"
    result = post_edit_segment(
        llm_engine=FakeLLM(response=candidate),
        source_segment="source",
        translated_segment=draft,
        glossary={"mise en production": "production deployment", "mot de passe": "password"},
        llm_settings=LLMSettings(strict_validation=True, fallback_to_argos=True),
    )
    assert "production deployment" in result
    assert "password" in result
