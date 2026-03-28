from __future__ import annotations

from pathlib import Path

import typer

from local_translator.config import LLMSettings, RuntimeConfig
from local_translator.gui.stub import run_gui_stub
from local_translator.logging_utils import setup_logging
from local_translator.models.types import EngineMode, TranslationReport
from local_translator.pipeline.translator import TranslationPipeline
from local_translator.reporting import format_report, write_report_json
from local_translator.reconstructors.registry import build_extractors, get_extractor

SUPPORTED_LANGS = ("fr", "en")
SUPPORTED_ENGINES = tuple(mode.value for mode in EngineMode)
SUPPORTED_SUFFIXES = tuple(
    sorted({suffix for extractor in build_extractors() for suffix in extractor.suffixes})
)

LANG_HELP = (
    "Language code. Supported values: "
    + ", ".join(SUPPORTED_LANGS)
    + ". Example: --from fr --to en"
)
ENGINE_HELP = (
    "Translation engine mode. Use 'argos' (default), 'hybrid', or 'llm'. "
    "When using 'llm' or 'hybrid', pass --llm-model."
)

app = typer.Typer(
    help=(
        "Translate local documents and plain text with optional glossary and reporting.\n\n"
        "Examples:\n"
        "  local-translator translate input.md --from fr --to en --output output.md\n"
        "  local-translator text --from en --to fr --content 'hello world' --engine argos\n"
        "  local-translator preview input.docx"
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _merge_reports(reports: list[TranslationReport]) -> TranslationReport:
    return TranslationReport(
        segment_count=sum(item.segment_count for item in reports),
        translated_count=sum(item.translated_count for item in reports),
        skipped_count=sum(item.skipped_count for item in reports),
        elapsed_seconds=sum(item.elapsed_seconds for item in reports),
        fallback_count=sum(item.fallback_count for item in reports),
        glossary_replacements=sum(item.glossary_replacements for item in reports),
        errors=[error for item in reports for error in item.errors],
    )


def _validate_language(value: str, option_name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_LANGS:
        supported = ", ".join(SUPPORTED_LANGS)
        raise typer.BadParameter(
            f"Unsupported language code '{value}'. Supported values: {supported}."
        )
    return normalized


def _validate_engine(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_ENGINES:
        supported = ", ".join(SUPPORTED_ENGINES)
        raise typer.BadParameter(
            f"Unsupported engine '{value}'. Choose one of: {supported}."
        )
    return normalized


def _validate_glossary_path(glossary: Path | None) -> Path | None:
    if glossary is None:
        return None

    if not glossary.exists():
        raise typer.BadParameter(f"Glossary file not found: {glossary}")
    if glossary.is_dir():
        raise typer.BadParameter(f"Glossary path must be a file, not a directory: {glossary}")

    supported_glossary_suffixes = {".json", ".yaml", ".yml"}
    suffix = glossary.suffix.lower()
    if suffix not in supported_glossary_suffixes:
        allowed = ", ".join(sorted(supported_glossary_suffixes))
        raise typer.BadParameter(
            f"Unsupported glossary format '{suffix or '<none>'}'. Supported formats: {allowed}."
        )

    return glossary


def _validate_input_file(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise typer.BadParameter(
            f"Unsupported input format '{suffix or '<none>'}'. Supported formats: {supported}."
        )


def _validate_common_options(
    source_lang: str,
    target_lang: str,
    engine: str,
    glossary: Path | None,
    llm_model: Path | None,
) -> tuple[str, str, str, Path | None]:
    source_lang = _validate_language(source_lang, "from")
    target_lang = _validate_language(target_lang, "to")
    if source_lang == target_lang:
        raise typer.BadParameter(
            "Source and target languages must be different. Example: --from fr --to en."
        )

    engine = _validate_engine(engine)
    glossary = _validate_glossary_path(glossary)

    if engine in {EngineMode.HYBRID.value, EngineMode.LLM.value} and llm_model is None:
        raise typer.BadParameter(
            "--llm-model is required when --engine is 'llm' or 'hybrid'."
        )
    if llm_model is not None and not llm_model.exists():
        raise typer.BadParameter(f"LLM model file not found: {llm_model}")

    return source_lang, target_lang, engine, glossary


@app.command(
    help=(
        "Translate a document file and write a translated file to --output.\n\n"
        "Supported input formats: " + ", ".join(SUPPORTED_SUFFIXES)
    )
)
def translate(
    input_path: Path = typer.Argument(..., help="Path to the source file to translate."),
    source_lang: str = typer.Option(..., "--from", help=LANG_HELP),
    target_lang: str = typer.Option(..., "--to", help=LANG_HELP),
    output: Path = typer.Option(..., "--output", help="Path where translated file is written."),
    engine: str = typer.Option("argos", "--engine", help=ENGINE_HELP),
    glossary: Path | None = typer.Option(
        None,
        "--glossary",
        help="Optional glossary file (.json/.yaml/.yml) for term overrides.",
    ),
    llm_model: Path | None = typer.Option(
        None,
        "--llm-model",
        help="Path to local GGUF model (required for --engine llm/hybrid).",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logs."),
    report: bool = typer.Option(False, "--report", help="Print execution report summary."),
    report_json: Path | None = typer.Option(
        None,
        "--report-json",
        help="Write execution report as JSON to this path.",
    ),
):
    setup_logging(verbose)

    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {input_path}")
    if input_path.is_dir():
        raise typer.BadParameter(f"Input path must be a file, not a directory: {input_path}")
    _validate_input_file(input_path)

    source_lang, target_lang, engine, glossary = _validate_common_options(
        source_lang,
        target_lang,
        engine,
        glossary,
        llm_model,
    )

    llm_cfg = LLMSettings(enabled=engine in {"hybrid", "llm"}, model_path=llm_model)
    cfg = RuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        engine_mode=EngineMode(engine),
        glossary_path=glossary,
        llm=llm_cfg,
        report=report,
    )

    try:
        pipeline = TranslationPipeline(cfg)
        extractor = get_extractor(input_path)
        extracted = extractor.extract(input_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    translated: list[str] = []
    reports: list[TranslationReport] = []
    for seg in extracted.segments:
        result = pipeline.translate_text(seg)
        translated.append(result.text)
        reports.append(result.report)
    extractor.reconstruct(extracted, translated, output)
    typer.echo(f"Translated file written to {output}")
    if report and reports:
        aggregated = _merge_reports(reports)
        typer.echo(format_report(aggregated))
        if report_json:
            write_report_json(aggregated, report_json)
            typer.echo(f"Report JSON written to {report_json}")


@app.command(help="Translate inline text content from one language to another.")
def text(
    source_lang: str = typer.Option(..., "--from", help=LANG_HELP),
    target_lang: str = typer.Option(..., "--to", help=LANG_HELP),
    engine: str = typer.Option("argos", "--engine", help=ENGINE_HELP),
    content: str = typer.Option(..., "--content", help="Text content to translate."),
    glossary: Path | None = typer.Option(
        None,
        "--glossary",
        help="Optional glossary file (.json/.yaml/.yml) for term overrides.",
    ),
    llm_model: Path | None = typer.Option(
        None,
        "--llm-model",
        help="Path to local GGUF model (required for --engine llm/hybrid).",
    ),
    report: bool = typer.Option(False, "--report", help="Print execution report summary."),
    report_json: Path | None = typer.Option(
        None,
        "--report-json",
        help="Write execution report as JSON to this path.",
    ),
):
    source_lang, target_lang, engine, glossary = _validate_common_options(
        source_lang,
        target_lang,
        engine,
        glossary,
        llm_model,
    )

    if not content.strip():
        raise typer.BadParameter("--content cannot be empty.")

    llm_cfg = LLMSettings(enabled=engine in {"hybrid", "llm"}, model_path=llm_model)
    cfg = RuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        engine_mode=EngineMode(engine),
        glossary_path=glossary,
        llm=llm_cfg,
        report=report,
    )
    try:
        pipeline = TranslationPipeline(cfg)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    result = pipeline.translate_text(content)
    typer.echo(result.text)
    if report:
        typer.echo(format_report(result.report))
        if report_json:
            write_report_json(result.report, report_json)
            typer.echo(f"Report JSON written to {report_json}")


@app.command(help="Inspect a file and print how many translatable segments were detected.")
def preview(
    input_path: Path = typer.Argument(..., help="Path to input file to inspect."),
):
    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {input_path}")
    if input_path.is_dir():
        raise typer.BadParameter(f"Input path must be a file, not a directory: {input_path}")
    _validate_input_file(input_path)

    try:
        extractor = get_extractor(input_path)
        extracted = extractor.extract(input_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"{input_path.name}: {len(extracted.segments)} segments detected")


@app.command(help="Open the GUI placeholder (stub).")
def gui():
    run_gui_stub()


if __name__ == "__main__":
    app()
