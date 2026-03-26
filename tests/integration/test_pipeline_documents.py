from __future__ import annotations

from pathlib import Path

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader
from pptx import Presentation

from local_translator.config import RuntimeConfig
from local_translator.models.types import EngineMode
from local_translator.pipeline.translator import TranslationPipeline
from local_translator.reconstructors.registry import get_extractor
from tests.integration.document_fixture_builder import (
    create_docx_fixture,
    create_pdf_fixture,
    create_pptx_fixture,
    create_xlsx_fixture,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
TRANSLATED_TAG = " [TRANSLATED]"


class FakeTranslator:
    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        return f"{text}{TRANSLATED_TAG}"


def prepare_fixture_path(tmp_path: Path, file_name: str) -> Path:
    if file_name in {"sample.txt", "sample.md"}:
        return FIXTURES_DIR / file_name

    generated_dir = tmp_path / "generated-fixtures"
    generated_dir.mkdir(exist_ok=True)
    fixture_path = generated_dir / file_name

    if file_name == "sample.docx":
        create_docx_fixture(fixture_path)
    elif file_name == "sample.pptx":
        create_pptx_fixture(fixture_path)
    elif file_name == "sample.xlsx":
        create_xlsx_fixture(fixture_path)
    elif file_name == "sample.pdf":
        create_pdf_fixture(fixture_path)
    else:  # pragma: no cover - guard for future updates
        raise ValueError(f"Unsupported fixture request: {file_name}")
    return fixture_path


def run_document_pipeline(input_path: Path, output_path: Path, monkeypatch) -> Path:
    cfg = RuntimeConfig(source_lang="fr", target_lang="en", engine_mode=EngineMode.ARGOS)
    pipeline = TranslationPipeline(cfg)
    monkeypatch.setattr(pipeline, "argos", FakeTranslator())

    extractor = get_extractor(input_path)
    extracted = extractor.extract(input_path)
    translated_segments = [pipeline.translate_text(segment).text for segment in extracted.segments]
    extractor.reconstruct(extracted, translated_segments, output_path)
    return output_path


def test_txt_pipeline_with_fixture(tmp_path, monkeypatch):
    input_path = prepare_fixture_path(tmp_path, "sample.txt")
    output_path = tmp_path / "sample.translated.txt"

    run_document_pipeline(input_path, output_path, monkeypatch)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Quarterly Summary" in content
    assert TRANSLATED_TAG in content


def test_md_pipeline_preserves_markdown_structure(tmp_path, monkeypatch):
    input_path = prepare_fixture_path(tmp_path, "sample.md")
    output_path = tmp_path / "sample.translated.md"

    run_document_pipeline(input_path, output_path, monkeypatch)

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("# ") for line in lines)
    assert any(line.startswith("- ") for line in lines)
    assert any(TRANSLATED_TAG in line for line in lines if line.strip())


def test_docx_pipeline_with_fixture(tmp_path, monkeypatch):
    input_path = prepare_fixture_path(tmp_path, "sample.docx")
    output_path = tmp_path / "sample.translated.docx"

    run_document_pipeline(input_path, output_path, monkeypatch)

    assert output_path.exists()
    translated_doc = Document(str(output_path))

    paragraphs = translated_doc.paragraphs
    paragraph_texts = [paragraph.text for paragraph in paragraphs]

    expected_heading = f"Project Brief{TRANSLATED_TAG}"
    expected_intro = (
        f"This document describes migration milestones for the billing service.{TRANSLATED_TAG}"
    )

    assert paragraphs[0].style.name.startswith("Heading")
    assert paragraph_texts[0] == expected_heading
    assert paragraph_texts[1] == expected_intro

    assert "Key term:" in paragraph_texts[2]
    assert "Service Level Objective" in paragraph_texts[2]
    assert "incident response policy" in paragraph_texts[2]

    key_phrases_in_order = [expected_heading, expected_intro, paragraph_texts[2]]
    ordered_indexes = [paragraph_texts.index(phrase) for phrase in key_phrases_in_order]
    assert ordered_indexes == sorted(ordered_indexes)

    formatted_paragraph = paragraphs[2]
    assert formatted_paragraph.runs[1].bold is True
    assert formatted_paragraph.runs[3].italic is True
    assert all(TRANSLATED_TAG in run.text for run in formatted_paragraph.runs if run.text.strip())

    assert len(translated_doc.tables) == 1
    table = translated_doc.tables[0]
    table_text = "\n".join(cell.text for row in table.rows for cell in row.cells)
    assert "Owner" in table_text
    assert "Status" in table_text
    assert "Platform Team" in table_text
    assert "On Track" in table_text

    expected_cells = [
        [f"Owner{TRANSLATED_TAG}", f"Status{TRANSLATED_TAG}"],
        [f"Platform Team{TRANSLATED_TAG}", f"On Track{TRANSLATED_TAG}"],
    ]
    actual_cells = [[cell.text for cell in row.cells] for row in table.rows]
    assert actual_cells == expected_cells


def test_pptx_pipeline_with_fixture(tmp_path, monkeypatch):
    input_path = prepare_fixture_path(tmp_path, "sample.pptx")
    output_path = tmp_path / "sample.translated.pptx"

    run_document_pipeline(input_path, output_path, monkeypatch)

    assert output_path.exists()
    translated_presentation = Presentation(str(output_path))
    assert len(translated_presentation.slides) == 1

    texts = []
    for shape in translated_presentation.slides[0].shapes:
        if hasattr(shape, "text"):
            texts.append(shape.text)

    assert any("Roadmap Update" in text for text in texts)
    assert any(TRANSLATED_TAG in text for text in texts)


def test_xlsx_pipeline_preserves_formulas_and_numbers(tmp_path, monkeypatch):
    input_path = prepare_fixture_path(tmp_path, "sample.xlsx")
    output_path = tmp_path / "sample.translated.xlsx"

    run_document_pipeline(input_path, output_path, monkeypatch)

    assert output_path.exists()
    workbook = load_workbook(str(output_path))
    overview = workbook["Overview"]

    assert overview["B2"].value == 125000
    assert overview["B4"].value == "=B2*1.1"
    assert TRANSLATED_TAG in overview["A1"].value
    assert TRANSLATED_TAG in overview["B3"].value

    risks = workbook["Risks"]
    assert TRANSLATED_TAG in risks["A2"].value


def test_pdf_text_pipeline_writes_translated_text_output(tmp_path, monkeypatch):
    input_path = prepare_fixture_path(tmp_path, "sample.pdf")
    output_path = tmp_path / "sample.translated.txt"

    reader = PdfReader(str(input_path))
    extracted = reader.pages[0].extract_text()
    assert "Text Native PDF Sample" in extracted

    run_document_pipeline(input_path, output_path, monkeypatch)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Text Native PDF Sample" in content
    assert TRANSLATED_TAG in content
