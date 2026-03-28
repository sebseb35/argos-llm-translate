from __future__ import annotations

from pathlib import Path

import typer

from local_translator.config import LLMSettings, RuntimeConfig
from local_translator.gui.stub import run_gui_stub
from local_translator.logging_utils import setup_logging
from local_translator.models.types import TranslationReport
from local_translator.pipeline.translator import TranslationPipeline
from local_translator.reporting import format_report, write_report_json
from local_translator.reconstructors.registry import get_extractor

app = typer.Typer(help="Offline local translator for sensitive documents")


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


@app.command()
def translate(
    input_path: Path = typer.Argument(..., exists=True),
    source_lang: str = typer.Option(..., "--from"),
    target_lang: str = typer.Option(..., "--to"),
    output: Path = typer.Option(..., "--output"),
    engine: str = typer.Option("argos", "--engine"),
    glossary: Path | None = typer.Option(None, "--glossary"),
    llm_model: Path | None = typer.Option(None, "--llm-model"),
    verbose: bool = typer.Option(False, "--verbose"),
    report: bool = typer.Option(False, "--report"),
    report_json: Path | None = typer.Option(None, "--report-json"),
):
    setup_logging(verbose)
    llm_cfg = LLMSettings(enabled=engine in {"hybrid", "llm"}, model_path=llm_model)
    cfg = RuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        engine_mode=engine,
        glossary_path=glossary,
        llm=llm_cfg,
        report=report,
    )
    pipeline = TranslationPipeline(cfg)
    extractor = get_extractor(input_path)
    extracted = extractor.extract(input_path)
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


@app.command()
def text(
    source_lang: str = typer.Option(..., "--from"),
    target_lang: str = typer.Option(..., "--to"),
    engine: str = typer.Option("argos", "--engine"),
    content: str = typer.Option(..., "--content"),
    glossary: Path | None = typer.Option(None, "--glossary"),
    llm_model: Path | None = typer.Option(None, "--llm-model"),
    report: bool = typer.Option(False, "--report"),
    report_json: Path | None = typer.Option(None, "--report-json"),
):
    llm_cfg = LLMSettings(enabled=engine in {"hybrid", "llm"}, model_path=llm_model)
    cfg = RuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        engine_mode=engine,
        glossary_path=glossary,
        llm=llm_cfg,
        report=report,
    )
    pipeline = TranslationPipeline(cfg)
    result = pipeline.translate_text(content)
    typer.echo(result.text)
    if report:
        typer.echo(format_report(result.report))
        if report_json:
            write_report_json(result.report, report_json)
            typer.echo(f"Report JSON written to {report_json}")


@app.command()
def preview(
    input_path: Path = typer.Argument(..., exists=True),
):
    extractor = get_extractor(input_path)
    extracted = extractor.extract(input_path)
    typer.echo(f"{input_path.name}: {len(extracted.segments)} segments detected")


@app.command()
def gui():
    run_gui_stub()


if __name__ == "__main__":
    app()
