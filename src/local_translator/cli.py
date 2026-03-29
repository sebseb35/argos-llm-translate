from __future__ import annotations

from pathlib import Path

import typer

from local_translator.api import (
    APIValidationError,
    SUPPORTED_ENGINES,
    SUPPORTED_LANGS,
    SUPPORTED_SUFFIXES,
    preview_file,
    translate_file,
    translate_text,
)
from local_translator.gui.stub import run_gui_stub
from local_translator.logging_utils import setup_logging
from local_translator.reporting import format_report

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
    llm_n_ctx: int = typer.Option(
        1024,
        "--llm-n-ctx",
        help="LLM context window size. Lower values reduce memory usage.",
    ),
    llm_n_batch: int = typer.Option(
        64,
        "--llm-n-batch",
        help="LLM batch size. Lower values reduce memory pressure.",
    ),
    llm_n_threads: int = typer.Option(
        2,
        "--llm-n-threads",
        help="LLM CPU threads. Lower values reduce CPU contention and memory pressure.",
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
    try:
        result = translate_file(
            input_path,
            output,
            source_lang=source_lang,
            target_lang=target_lang,
            engine=engine,
            glossary=glossary,
            llm_model=llm_model,
            llm_n_ctx=llm_n_ctx,
            llm_n_batch=llm_n_batch,
            llm_n_threads=llm_n_threads,
            report=report,
            report_json=report_json,
        )
    except (APIValidationError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Translated file written to {result.output_path}")
    if report:
        typer.echo(format_report(result.report))
        if report_json:
            typer.echo(f"Report JSON written to {report_json}")
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}")


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
    llm_n_ctx: int = typer.Option(
        1024,
        "--llm-n-ctx",
        help="LLM context window size. Lower values reduce memory usage.",
    ),
    llm_n_batch: int = typer.Option(
        64,
        "--llm-n-batch",
        help="LLM batch size. Lower values reduce memory pressure.",
    ),
    llm_n_threads: int = typer.Option(
        2,
        "--llm-n-threads",
        help="LLM CPU threads. Lower values reduce CPU contention and memory pressure.",
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
    try:
        result = translate_text(
            content,
            source_lang=source_lang,
            target_lang=target_lang,
            engine=engine,
            glossary=glossary,
            llm_model=llm_model,
            llm_n_ctx=llm_n_ctx,
            llm_n_batch=llm_n_batch,
            llm_n_threads=llm_n_threads,
            report=report,
            report_json=report_json,
        )
    except (APIValidationError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(result.translated_text)
    if report:
        typer.echo(format_report(result.report))
        if report_json:
            typer.echo(f"Report JSON written to {report_json}")
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}")


@app.command(help="Inspect a file and print how many translatable segments were detected.")
def preview(
    input_path: Path = typer.Argument(..., help="Path to input file to inspect."),
):
    try:
        result = preview_file(input_path)
    except (APIValidationError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"{result.input_path.name}: {result.segment_count} segments detected")


@app.command(help="Open the GUI placeholder (stub).")
def gui():
    run_gui_stub()


if __name__ == "__main__":
    app()
