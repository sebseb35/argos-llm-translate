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
    monkeypatch.setattr("local_translator.api.TranslationPipeline", FakePipeline)
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
    monkeypatch.setattr("local_translator.api.TranslationPipeline", FakePipeline)
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
    assert "llm_calls" in payload
    assert "llm_skipped" in payload
    assert "validation_failures" in payload
    assert "routing_trace" in payload


def test_report_summary_command(tmp_path):
    report_a = tmp_path / "a.json"
    report_b = tmp_path / "b.json"
    report_a.write_text(
        json.dumps(
            {
                "segment_count": 10,
                "llm_skipped": 7,
                "llm_safe_segments": 2,
                "llm_smart_segments": 1,
                "llm_chunks_built": 1,
                "avg_chunk_size": 3.0,
                "chunk_fallbacks": 0,
                "segment_fallbacks": 1,
                "llm_calls_saved_by_chunking": 2,
                "validation_failures": {"placeholder_mismatch": 1},
            }
        ),
        encoding="utf-8",
    )
    report_b.write_text(
        json.dumps(
            {
                "segment_count": 10,
                "llm_skipped": 6,
                "llm_safe_segments": 3,
                "llm_smart_segments": 1,
                "llm_chunks_built": 2,
                "avg_chunk_size": 2.0,
                "chunk_fallbacks": 1,
                "segment_fallbacks": 1,
                "llm_calls_saved_by_chunking": 1,
                "validation_failures": {"placeholder_mismatch": 2},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["report-summary", str(report_a), str(report_b)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["segments"] == 20
    assert payload["llm_calls_saved_by_chunking"] == 3
    assert payload["validation_failures"]["placeholder_mismatch"] == 3
