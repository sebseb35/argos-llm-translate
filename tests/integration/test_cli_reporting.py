import json

from typer.testing import CliRunner

from local_translator.cli import app
from local_translator.models.types import TranslationReport
from local_translator.pipeline.translator import TranslationResult


runner = CliRunner()


class FakePipeline:
    def __init__(self, config):
        self.config = config

    def translate_text(self, content: str) -> TranslationResult:
        return TranslationResult(
            text=f"T:{content}",
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


def test_text_report_printed_with_flag(monkeypatch):
    monkeypatch.setattr("local_translator.cli.TranslationPipeline", FakePipeline)
    result = runner.invoke(
        app,
        [
            "text",
            "--from",
            "fr",
            "--to",
            "en",
            "--engine",
            "argos",
            "--content",
            "bonjour",
            "--report",
        ],
    )

    assert result.exit_code == 0
    assert "Execution report" in result.output
    assert "Segments processed:" in result.output
    assert "Fallbacks:" in result.output


def test_text_report_json_export(tmp_path, monkeypatch):
    monkeypatch.setattr("local_translator.cli.TranslationPipeline", FakePipeline)
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "text",
            "--from",
            "fr",
            "--to",
            "en",
            "--engine",
            "argos",
            "--content",
            "mot de passe",
            "--report",
            "--report-json",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["segment_count"] == 1
    assert payload["error_count"] == 0
    assert "fallback_count" in payload
    assert "glossary_replacements" in payload
