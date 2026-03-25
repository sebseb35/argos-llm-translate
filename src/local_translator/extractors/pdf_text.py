from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from local_translator.extractors.base import BaseExtractor, ExtractedDocument


class PdfTextExtractor(BaseExtractor):
    suffixes = (".pdf",)

    def extract(self, file_path: Path) -> ExtractedDocument:
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return ExtractedDocument(file_path=file_path, segments=pages)

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        # Best effort V1: write translated content as plain text sidecar.
        output_path.write_text("\n\n---\n\n".join(translated_segments), encoding="utf-8")
