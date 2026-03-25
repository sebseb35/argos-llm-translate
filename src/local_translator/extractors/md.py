from __future__ import annotations

from pathlib import Path

from local_translator.extractors.base import BaseExtractor, ExtractedDocument


class MarkdownExtractor(BaseExtractor):
    suffixes = (".md",)

    def extract(self, file_path: Path) -> ExtractedDocument:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        return ExtractedDocument(file_path=file_path, segments=lines)

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        output_path.write_text("\n".join(translated_segments) + "\n", encoding="utf-8")
