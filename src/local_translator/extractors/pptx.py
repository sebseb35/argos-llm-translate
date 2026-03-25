from __future__ import annotations

from pathlib import Path

from pptx import Presentation

from local_translator.extractors.base import BaseExtractor, ExtractedDocument


class PptxExtractor(BaseExtractor):
    suffixes = (".pptx",)

    def extract(self, file_path: Path) -> ExtractedDocument:
        prs = Presentation(str(file_path))
        segments: list[str] = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    segments.append(shape.text)
        return ExtractedDocument(file_path=file_path, segments=segments)

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        prs = Presentation(str(extracted.file_path))
        idx = 0
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and idx < len(translated_segments):
                    shape.text = translated_segments[idx]
                    idx += 1
        prs.save(str(output_path))
