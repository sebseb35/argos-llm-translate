from typer.testing import CliRunner

from local_translator.cli import app


runner = CliRunner()


def test_root_help_includes_examples_and_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Translate local documents and plain text" in result.output
    assert "Examples:" in result.output
    assert "translate" in result.output
    assert "preview" in result.output


def test_text_rejects_invalid_language_code():
    result = runner.invoke(
        app,
        [
            "text",
            "--from",
            "de",
            "--to",
            "en",
            "--content",
            "bonjour",
        ],
    )

    assert result.exit_code != 0
    assert "Unsupported language code 'de'" in result.output
    assert "Traceback" not in result.output


def test_text_requires_llm_model_for_llm_engine():
    result = runner.invoke(
        app,
        [
            "text",
            "--from",
            "fr",
            "--to",
            "en",
            "--engine",
            "llm",
            "--content",
            "bonjour",
        ],
    )

    assert result.exit_code != 0
    assert "--llm-model is required" in result.output


def test_translate_rejects_unsupported_input_format(tmp_path):
    input_file = tmp_path / "sample.rtf"
    input_file.write_text("hello", encoding="utf-8")
    output_file = tmp_path / "out.rtf"

    result = runner.invoke(
        app,
        [
            "translate",
            str(input_file),
            "--from",
            "fr",
            "--to",
            "en",
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code != 0
    assert "Unsupported input format '.rtf'" in result.output
    assert "Supported formats:" in result.output


def test_translate_rejects_missing_glossary_file(tmp_path):
    input_file = tmp_path / "sample.txt"
    input_file.write_text("hello", encoding="utf-8")
    output_file = tmp_path / "out.txt"
    glossary = tmp_path / "missing.json"

    result = runner.invoke(
        app,
        [
            "translate",
            str(input_file),
            "--from",
            "fr",
            "--to",
            "en",
            "--output",
            str(output_file),
            "--glossary",
            str(glossary),
        ],
    )

    assert result.exit_code != 0
    assert "Glossary file not found" in result.output


def test_preview_rejects_missing_file(tmp_path):
    input_file = tmp_path / "missing.txt"

    result = runner.invoke(app, ["preview", str(input_file)])

    assert result.exit_code != 0
    assert "Input file not found" in result.output
