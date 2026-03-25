from __future__ import annotations

from pathlib import Path

from local_translator.extractors.base import BaseExtractor, ExtractedDocument


class TxtExtractor(BaseExtractor):
    suffixes = (".txt",)

    def extract(self, file_path: Path) -> ExtractedDocument:
        return ExtractedDocument(file_path=file_path, segments=[file_path.read_text(encoding="utf-8")])

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        output_path.write_text("\n".join(translated_segments), encoding="utf-8")
