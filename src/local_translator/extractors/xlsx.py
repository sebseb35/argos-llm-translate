from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from local_translator.extractors.base import BaseExtractor, ExtractedDocument


class XlsxExtractor(BaseExtractor):
    suffixes = (".xlsx",)

    def extract(self, file_path: Path) -> ExtractedDocument:
        wb = load_workbook(str(file_path))
        segments: list[str] = []
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str) and cell.value.strip() and not cell.data_type == "f":
                        segments.append(cell.value)
        return ExtractedDocument(file_path=file_path, segments=segments)

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        wb = load_workbook(str(extracted.file_path))
        idx = 0
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str) and cell.value.strip() and not cell.data_type == "f":
                        if idx < len(translated_segments):
                            cell.value = translated_segments[idx]
                        idx += 1
        wb.save(str(output_path))
