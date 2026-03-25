from __future__ import annotations

from pathlib import Path

import typer

from local_translator.config import LLMSettings, RuntimeConfig
from local_translator.gui.stub import run_gui_stub
from local_translator.logging_utils import setup_logging
from local_translator.pipeline.translator import TranslationPipeline
from local_translator.reconstructors.registry import get_extractor

app = typer.Typer(help="Offline local translator for sensitive documents")


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
):
    setup_logging(verbose)
    llm_cfg = LLMSettings(enabled=engine in {"hybrid", "llm"}, model_path=llm_model)
    cfg = RuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        engine_mode=engine,
        glossary_path=glossary,
        llm=llm_cfg,
        report=True,
    )
    pipeline = TranslationPipeline(cfg)
    extractor = get_extractor(input_path)
    extracted = extractor.extract(input_path)
    translated = [pipeline.translate_text(seg).text for seg in extracted.segments]
    extractor.reconstruct(extracted, translated, output)
    typer.echo(f"Translated file written to {output}")


@app.command()
def text(
    source_lang: str = typer.Option(..., "--from"),
    target_lang: str = typer.Option(..., "--to"),
    engine: str = typer.Option("argos", "--engine"),
    content: str = typer.Option(..., "--content"),
    llm_model: Path | None = typer.Option(None, "--llm-model"),
):
    llm_cfg = LLMSettings(enabled=engine in {"hybrid", "llm"}, model_path=llm_model)
    cfg = RuntimeConfig(source_lang=source_lang, target_lang=target_lang, engine_mode=engine, llm=llm_cfg)
    pipeline = TranslationPipeline(cfg)
    typer.echo(pipeline.translate_text(content).text)


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
