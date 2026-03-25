from __future__ import annotations

from pathlib import Path

from docx import Document

from local_translator.extractors.base import BaseExtractor, ExtractedDocument


class DocxExtractor(BaseExtractor):
    suffixes = (".docx",)

    def extract(self, file_path: Path) -> ExtractedDocument:
        doc = Document(str(file_path))
        segments = [p.text for p in doc.paragraphs]
        return ExtractedDocument(file_path=file_path, segments=segments)

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        doc = Document(str(extracted.file_path))
        for p, text in zip(doc.paragraphs, translated_segments):
            p.text = text
        doc.save(str(output_path))
