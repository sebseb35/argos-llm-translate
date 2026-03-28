from local_translator.config import RuntimeConfig
from local_translator.models.types import EngineMode
from local_translator.pipeline.translator import TranslationPipeline


class DummyArgos:
    def translate(self, text: str) -> str:
        return f"T:{text}"


class DummyLLM:
    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        # Tries to rewrite glossary terms; strict placeholder validation should reject this.
        return translated.replace("password", "credential")


class FailingLLM:
    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        raise RuntimeError("post-edit failed")


def test_pipeline_argos_mode(monkeypatch):
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.ARGOS)
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", DummyArgos())

    result = pipeline.translate_text("bonjour")
    assert result.text.startswith("T:")


def test_pipeline_hybrid_glossary_terms_survive_post_editing(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  mot de passe: password
""".strip(),
        encoding="utf-8",
    )

    cfg = RuntimeConfig(
        source_lang="fr",
        target_lang="en",
        engine_mode=EngineMode.HYBRID,
        glossary_path=glossary_path,
    )
    cfg.llm.enabled = True
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", DummyArgos())
    monkeypatch.setattr(pipeline, "llm", DummyLLM())

    result = pipeline.translate_text("mot de passe")
    assert "password" in result.text
    assert "credential" not in result.text


def test_pipeline_report_tracks_fallbacks_and_glossary_replacements(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  mot de passe: password
""".strip(),
        encoding="utf-8",
    )

    cfg = RuntimeConfig(
        source_lang="fr",
        target_lang="en",
        engine_mode=EngineMode.HYBRID,
        glossary_path=glossary_path,
    )
    cfg.llm.enabled = True
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", DummyArgos())
    monkeypatch.setattr(pipeline, "llm", FailingLLM())

    result = pipeline.translate_text("mot de passe")
    assert result.report.segment_count == 1
    assert result.report.fallback_count == 1
    assert result.report.glossary_replacements == 1
    assert len(result.report.errors) == 0
