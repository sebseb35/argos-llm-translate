from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from local_translator.config import LLMSettings, RuntimeConfig
from local_translator.models.types import EngineMode, TranslationReport
from local_translator.pipeline.translator import TranslationPipeline
from local_translator.reporting import write_report_json
from local_translator.reconstructors.registry import build_extractors, get_extractor

SUPPORTED_LANGS = ("fr", "en")
SUPPORTED_ENGINES = tuple(mode.value for mode in EngineMode)
SUPPORTED_SUFFIXES = tuple(
    sorted({suffix for extractor in build_extractors() for suffix in extractor.suffixes})
)


class APIValidationError(ValueError):
    """Raised when API input parameters are invalid."""


@dataclass(slots=True)
class TextTranslationOutput:
    translated_text: str
    report: TranslationReport
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FileTranslationOutput:
    output_path: Path
    translated_segments: list[str]
    report: TranslationReport
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PreviewOutput:
    input_path: Path
    segment_count: int


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
        raise APIValidationError(
            f"Unsupported language code '{value}' for {option_name}. Supported values: {supported}."
        )
    return normalized


def _validate_engine(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_ENGINES:
        supported = ", ".join(SUPPORTED_ENGINES)
        raise APIValidationError(f"Unsupported engine '{value}'. Choose one of: {supported}.")
    return normalized


def _validate_glossary_path(glossary: Path | None) -> Path | None:
    if glossary is None:
        return None

    if not glossary.exists():
        raise APIValidationError(f"Glossary file not found: {glossary}")
    if glossary.is_dir():
        raise APIValidationError(f"Glossary path must be a file, not a directory: {glossary}")

    supported_glossary_suffixes = {".json", ".yaml", ".yml"}
    suffix = glossary.suffix.lower()
    if suffix not in supported_glossary_suffixes:
        allowed = ", ".join(sorted(supported_glossary_suffixes))
        raise APIValidationError(
            f"Unsupported glossary format '{suffix or '<none>'}'. Supported formats: {allowed}."
        )

    return glossary


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
        raise APIValidationError(
            "Source and target languages must be different. Example: --from fr --to en."
        )

    engine = _validate_engine(engine)
    glossary = _validate_glossary_path(glossary)

    if engine in {EngineMode.HYBRID.value, EngineMode.LLM.value} and llm_model is None:
        raise APIValidationError("--llm-model is required when engine is 'llm' or 'hybrid'.")
    if llm_model is not None and not llm_model.exists():
        raise APIValidationError(f"LLM model file not found: {llm_model}")

    return source_lang, target_lang, engine, glossary


def _validate_input_file(path: Path) -> None:
    if not path.exists():
        raise APIValidationError(f"Input file not found: {path}")
    if path.is_dir():
        raise APIValidationError(f"Input path must be a file, not a directory: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise APIValidationError(
            f"Unsupported input format '{suffix or '<none>'}'. Supported formats: {supported}."
        )


def _build_runtime_config(
    source_lang: str,
    target_lang: str,
    engine: str,
    glossary: Path | None,
    llm_model: Path | None,
    report: bool,
) -> RuntimeConfig:
    source_lang, target_lang, engine, glossary = _validate_common_options(
        source_lang,
        target_lang,
        engine,
        glossary,
        llm_model,
    )
    llm_cfg = LLMSettings(enabled=engine in {"hybrid", "llm"}, model_path=llm_model)
    return RuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        engine_mode=EngineMode(engine),
        glossary_path=glossary,
        llm=llm_cfg,
        report=report,
    )


def translate_text(
    content: str,
    *,
    source_lang: str,
    target_lang: str,
    engine: str = "argos",
    glossary: Path | None = None,
    llm_model: Path | None = None,
    report: bool = False,
    report_json: Path | None = None,
) -> TextTranslationOutput:
    if not content.strip():
        raise APIValidationError("content cannot be empty.")

    warnings: list[str] = []
    if report_json and not report:
        warnings.append("report_json was provided without report=True; JSON export skipped.")

    cfg = _build_runtime_config(source_lang, target_lang, engine, glossary, llm_model, report)
    pipeline = TranslationPipeline(cfg)
    result = pipeline.translate_text(content)

    if report and report_json:
        write_report_json(result.report, report_json)

    return TextTranslationOutput(translated_text=result.text, report=result.report, warnings=warnings)


def translate_file(
    input_path: Path,
    output_path: Path,
    *,
    source_lang: str,
    target_lang: str,
    engine: str = "argos",
    glossary: Path | None = None,
    llm_model: Path | None = None,
    report: bool = False,
    report_json: Path | None = None,
) -> FileTranslationOutput:
    _validate_input_file(input_path)
    warnings: list[str] = []
    if report_json and not report:
        warnings.append("report_json was provided without report=True; JSON export skipped.")

    cfg = _build_runtime_config(source_lang, target_lang, engine, glossary, llm_model, report)
    pipeline = TranslationPipeline(cfg)
    extractor = get_extractor(input_path)
    extracted = extractor.extract(input_path)

    translated: list[str] = []
    reports: list[TranslationReport] = []
    for seg in extracted.segments:
        result = pipeline.translate_text(seg)
        translated.append(result.text)
        reports.append(result.report)

    extractor.reconstruct(extracted, translated, output_path)
    merged_report = _merge_reports(reports) if reports else TranslationReport(0, 0, 0, 0.0)

    if report and report_json:
        write_report_json(merged_report, report_json)

    return FileTranslationOutput(
        output_path=output_path,
        translated_segments=translated,
        report=merged_report,
        warnings=warnings,
    )


def preview_file(input_path: Path) -> PreviewOutput:
    _validate_input_file(input_path)
    extractor = get_extractor(input_path)
    extracted = extractor.extract(input_path)
    return PreviewOutput(input_path=input_path, segment_count=len(extracted.segments))
