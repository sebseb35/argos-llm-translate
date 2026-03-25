from local_translator.config import RuntimeConfig
from local_translator.models.types import EngineMode
from local_translator.pipeline.translator import TranslationPipeline


class DummyArgos:
    def translate(self, text: str) -> str:
        return f"T:{text}"


def test_pipeline_argos_mode(monkeypatch):
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.ARGOS)
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", DummyArgos())

    result = pipeline.translate_text("bonjour")
    assert result.text.startswith("T:")
