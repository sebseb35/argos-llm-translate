"""Core package for local-translator."""

from local_translator.api import (
    APIValidationError,
    FileTranslationOutput,
    PreviewOutput,
    TextTranslationOutput,
    preview_file,
    translate_file,
    translate_text,
)

__all__ = [
    "__version__",
    "APIValidationError",
    "TextTranslationOutput",
    "FileTranslationOutput",
    "PreviewOutput",
    "translate_text",
    "translate_file",
    "preview_file",
]
__version__ = "0.1.0"
