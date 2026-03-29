from local_translator.models.types import TranslationReport
from local_translator.reporting import format_report, report_to_dict


def test_format_report_contains_human_readable_metrics():
    report = TranslationReport(
        segment_count=120,
        translated_count=120,
        skipped_count=0,
        elapsed_seconds=2.345,
        fallback_count=3,
        glossary_replacements=8,
        errors=[],
    )

    output = format_report(report)
    assert "Segments processed: 120" in output
    assert "Fallbacks: 3" in output
    assert "Errors: 0" in output
    assert "Glossary replacements: 8" in output
    assert "LLM calls: 0" in output
    assert "LLM skipped: 0" in output
    assert "Time: 2.345s" in output


def test_report_to_dict_contains_expected_keys():
    report = TranslationReport(
        segment_count=2,
        translated_count=1,
        skipped_count=1,
        elapsed_seconds=0.5,
        fallback_count=1,
        glossary_replacements=4,
        errors=["oops"],
    )

    payload = report_to_dict(report)
    assert payload == {
        "segment_count": 2,
        "translated_count": 1,
        "skipped_count": 1,
        "elapsed_seconds": 0.5,
        "error_count": 1,
        "errors": ["oops"],
        "fallback_count": 1,
        "glossary_replacements": 4,
        "llm_calls": 0,
        "llm_skipped": 0,
    }
