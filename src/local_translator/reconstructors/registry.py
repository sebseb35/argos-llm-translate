from __future__ import annotations

from pathlib import Path

from local_translator.extractors.base import BaseExtractor
from local_translator.extractors.docx import DocxExtractor
from local_translator.extractors.md import MarkdownExtractor
from local_translator.extractors.pdf_text import PdfTextExtractor
from local_translator.extractors.pptx import PptxExtractor
from local_translator.extractors.txt import TxtExtractor
from local_translator.extractors.xlsx import XlsxExtractor


def build_extractors() -> list[BaseExtractor]:
    return [
        TxtExtractor(),
        MarkdownExtractor(),
        DocxExtractor(),
        PptxExtractor(),
        XlsxExtractor(),
        PdfTextExtractor(),
    ]


def get_extractor(file_path: Path) -> BaseExtractor:
    suffix = file_path.suffix.lower()
    for extractor in build_extractors():
        if suffix in extractor.suffixes:
            return extractor
    raise ValueError(f"Unsupported format: {suffix}")
