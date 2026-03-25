from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExtractedDocument:
    file_path: Path
    segments: list[str]


class BaseExtractor:
    suffixes: tuple[str, ...] = ()

    def extract(self, file_path: Path) -> ExtractedDocument:
        raise NotImplementedError

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        raise NotImplementedError
