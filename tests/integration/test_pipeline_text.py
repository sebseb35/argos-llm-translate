from local_translator.config import RuntimeConfig
from local_translator.models.types import EngineMode
from local_translator.pipeline.translator import TranslationPipeline


class DummyArgos:
    def translate(self, text: str) -> str:
        return f"T:{text}"


class FrenchToEnglishArgos:
    def translate(self, text: str) -> str:
        return (
            text.replace("La ", "The ")
            .replace(" la ", " the ")
            .replace(" du ", " for ")
            .replace(" démarre", " starts")
            .replace(" après", " after")
            .replace(" validation", " validation")
            .replace("recette", "recipe")
            .replace("lot", "batch")
        )


class DummyLLM:
    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        # Tries to rewrite glossary terms; strict placeholder validation should reject this.
        return translated.replace("password", "credential")


class RewritingLLM:
    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        return translated.replace("acceptance testing", "recipe").replace("work package", "batch")


class FailingLLM:
    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        raise RuntimeError("post-edit failed")


class RolloutArgos:
    def translate(self, text: str) -> str:
        return text.replace("mise en production", "rollout")


class PlaceholderNormalizingArgos:
    def translate(self, text: str) -> str:
        return (
            text.replace("La ", "The ")
            .replace(" du ", "   of   ")
            .replace(" démarre", " starts")
            .replace(" après", " after")
            .replace(" validation", " validation")
            .replace("__LT_GLOSSARY_TERM_0000__", "  LT GLOSSARY TERM 0000  ")
            .replace("__LT_GLOSSARY_TERM_0001__", "  LT GLOSSARY TERM 0001  ")
        )


class PassthroughLLM:
    def post_edit(self, source: str, translated: str, glossary: dict[str, str] | None = None) -> str:
        return translated


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
    assert result.report.glossary_replacements == 2
    assert len(result.report.errors) == 0


def test_pipeline_argos_post_translation_glossary_enforcement(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  recette: acceptance testing
  lot: work package
""".strip(),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.ARGOS, glossary_path=glossary_path)
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", FrenchToEnglishArgos())

    result = pipeline.translate_text("La recette du lot 2 démarre après validation.")
    assert result.text == "The acceptance testing for work package 2 starts after validation."


def test_pipeline_hybrid_llm_success_still_enforces_glossary(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  recette: acceptance testing
  lot: work package
""".strip(),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.HYBRID, glossary_path=glossary_path)
    cfg.llm.enabled = True
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", FrenchToEnglishArgos())
    monkeypatch.setattr(pipeline, "llm", RewritingLLM())

    result = pipeline.translate_text("La recette du lot 2 démarre après validation.")
    assert "acceptance testing" in result.text
    assert "work package" in result.text
    assert "recipe" not in result.text
    assert "batch" not in result.text


def test_pipeline_hybrid_fallback_still_enforces_glossary(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  recette: acceptance testing
  lot: work package
""".strip(),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.HYBRID, glossary_path=glossary_path)
    cfg.llm.enabled = True
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", FrenchToEnglishArgos())
    monkeypatch.setattr(pipeline, "llm", FailingLLM())

    result = pipeline.translate_text("La recette du lot 2 démarre après validation.")
    assert result.report.fallback_count == 1
    assert "acceptance testing" in result.text
    assert "work package" in result.text


def test_pipeline_argos_restores_normalized_glossary_placeholders(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  recette: acceptance testing
  lot: work package
""".strip(),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.ARGOS, glossary_path=glossary_path)
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", PlaceholderNormalizingArgos())

    result = pipeline.translate_text("La recette du lot 2 démarre après validation.")
    assert "acceptance testing" in result.text
    assert "work package" in result.text
    assert "LT GLOSSARY TERM" not in result.text
    assert "  " not in result.text
    assert "recipe" not in result.text
    assert "batch" not in result.text


def test_pipeline_hybrid_restores_normalized_glossary_placeholders(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  recette: acceptance testing
  lot: work package
""".strip(),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.HYBRID, glossary_path=glossary_path)
    cfg.llm.enabled = True
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", PlaceholderNormalizingArgos())
    monkeypatch.setattr(pipeline, "llm", PassthroughLLM())

    result = pipeline.translate_text("La recette du lot 2 démarre après validation.")
    assert "acceptance testing" in result.text
    assert "work package" in result.text
    assert "LT GLOSSARY TERM" not in result.text
    assert "  " not in result.text
    assert "recipe" not in result.text
    assert "batch" not in result.text


def test_pipeline_hybrid_fallback_restores_normalized_glossary_placeholders(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  recette: acceptance testing
  lot: work package
""".strip(),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.HYBRID, glossary_path=glossary_path)
    cfg.llm.enabled = True
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", PlaceholderNormalizingArgos())
    monkeypatch.setattr(pipeline, "llm", FailingLLM())

    result = pipeline.translate_text("La recette du lot 2 démarre après validation.")
    assert result.report.fallback_count == 1
    assert "acceptance testing" in result.text
    assert "work package" in result.text
    assert "LT GLOSSARY TERM" not in result.text
    assert "  " not in result.text
    assert "recipe" not in result.text
    assert "batch" not in result.text


def test_pipeline_enforces_multiword_and_repeated_glossary_terms(monkeypatch, tmp_path):
    glossary_path = tmp_path / "fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  mise en production: production deployment
""".strip(),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.ARGOS, glossary_path=glossary_path)
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", RolloutArgos())

    result = pipeline.translate_text("mise en production puis mise en production")
    assert result.text == "production deployment puis production deployment"
