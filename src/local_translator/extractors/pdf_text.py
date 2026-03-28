from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from local_translator.extractors.base import BaseExtractor, ExtractedDocument

_MIN_MEANINGFUL_TEXT_CHARS = 20


class PdfTextExtractor(BaseExtractor):
    suffixes = (".pdf",)

    def extract(self, file_path: Path) -> ExtractedDocument:
        try:
            reader = PdfReader(str(file_path))
        except PdfReadError as exc:
            raise ValueError(f"Invalid PDF file: {file_path.name}") from exc

        pages = [self._normalize_page_text(page.extract_text() or "") for page in reader.pages]
        joined_text = "\n".join(pages).strip()
        meaningful_chars = sum(1 for char in joined_text if char.isalnum())

        if meaningful_chars < _MIN_MEANINGFUL_TEXT_CHARS:
            raise ValueError(
                "Unsupported PDF: no meaningful extractable text was found. "
                "Only text-native PDFs are supported; scanned/image PDFs are not supported."
            )

        metadata = {
            "output_strategy": "translated_text_sidecar",
            "output_format": "txt",
            "min_meaningful_text_chars": _MIN_MEANINGFUL_TEXT_CHARS,
        }
        return ExtractedDocument(file_path=file_path, segments=pages, metadata=metadata)

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        if output_path.suffix.lower() != ".txt":
            raise ValueError(
                "PDF translations are text sidecar outputs only. "
                "Use a .txt output path (example: input.translated.txt)."
            )

        non_empty_blocks = [segment.strip() for segment in translated_segments if segment.strip()]
        output_path.write_text("\n\n".join(non_empty_blocks) + "\n", encoding="utf-8")

    @staticmethod
    def _normalize_page_text(text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        normalized = "\n".join(line for line in lines if line.strip())
        return normalized.strip()
