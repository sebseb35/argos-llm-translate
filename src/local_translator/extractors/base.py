from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExtractedDocument:
    file_path: Path
    segments: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseExtractor:
    suffixes: tuple[str, ...] = ()

    def extract(self, file_path: Path) -> ExtractedDocument:
        raise NotImplementedError

    def reconstruct(self, extracted: ExtractedDocument, translated_segments: list[str], output_path: Path) -> None:
        raise NotImplementedError
