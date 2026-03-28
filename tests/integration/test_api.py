from pathlib import Path

from typer.testing import CliRunner

from local_translator.api import APIValidationError, preview_file, translate_file, translate_text
from local_translator.cli import app
from local_translator.models.types import TranslationReport
from local_translator.pipeline.translator import TranslationResult


runner = CliRunner()


class FakePipeline:
    def __init__(self, config):
        self.config = config

    def translate_text(self, content: str) -> TranslationResult:
        return TranslationResult(
            text=f"API:{content}",
            report=TranslationReport(
                segment_count=1,
                translated_count=1,
                skipped_count=0,
                elapsed_seconds=0.01,
                fallback_count=0,
                glossary_replacements=0,
                errors=[],
            ),
        )


class FakeExtraction:
    def __init__(self, segments: list[str]):
        self.segments = segments


class FakeExtractor:
    suffixes = (".txt",)

    def extract(self, input_path: Path) -> FakeExtraction:
        content = input_path.read_text(encoding="utf-8")
        return FakeExtraction([content])

    def reconstruct(self, extracted: FakeExtraction, translated_segments: list[str], output_path: Path) -> None:
        output_path.write_text("\n".join(translated_segments), encoding="utf-8")


def test_translate_text_api_returns_structured_output(monkeypatch):
    monkeypatch.setattr("local_translator.api.TranslationPipeline", FakePipeline)

    result = translate_text("bonjour", source_lang="fr", target_lang="en")

    assert result.translated_text == "API:bonjour"
    assert result.report.segment_count == 1
    assert result.warnings == []


def test_translate_file_api_writes_output_and_returns_report(tmp_path, monkeypatch):
    monkeypatch.setattr("local_translator.api.TranslationPipeline", FakePipeline)
    monkeypatch.setattr("local_translator.api.get_extractor", lambda _: FakeExtractor())

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text("bonjour", encoding="utf-8")

    result = translate_file(
        input_path,
        output_path,
        source_lang="fr",
        target_lang="en",
    )

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "API:bonjour"
    assert result.output_path == output_path
    assert result.report.segment_count == 1
    assert result.translated_segments == ["API:bonjour"]


def test_preview_file_api_returns_segment_count(tmp_path, monkeypatch):
    monkeypatch.setattr("local_translator.api.get_extractor", lambda _: FakeExtractor())

    input_path = tmp_path / "input.txt"
    input_path.write_text("one segment", encoding="utf-8")

    preview = preview_file(input_path)

    assert preview.input_path == input_path
    assert preview.segment_count == 1


def test_translate_text_api_matches_cli_output(monkeypatch):
    monkeypatch.setattr("local_translator.api.TranslationPipeline", FakePipeline)

    api_result = translate_text("bonjour", source_lang="fr", target_lang="en")
    cli_result = runner.invoke(
        app,
        ["text", "--from", "fr", "--to", "en", "--engine", "argos", "--content", "bonjour"],
    )

    assert cli_result.exit_code == 0
    assert api_result.translated_text in cli_result.output


def test_translate_text_rejects_empty_content():
    try:
        translate_text("   ", source_lang="fr", target_lang="en")
    except APIValidationError as exc:
        assert "content cannot be empty" in str(exc)
    else:  # pragma: no cover - explicit guard
        raise AssertionError("Expected APIValidationError")
