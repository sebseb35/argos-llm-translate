from __future__ import annotations

import json
from pathlib import Path


def load_glossary(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise ValueError("PyYAML required for YAML glossary files") from exc
        data = yaml.safe_load(payload) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(payload)
    elif path.suffix.lower() == ".toml":
        import tomllib

        data = tomllib.loads(payload)
    else:
        raise ValueError(f"Unsupported glossary format: {path}")
    if not isinstance(data, dict):
        raise ValueError("Glossary must be a key-value object")
    return {str(k): str(v) for k, v in data.items()}


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    out = text
    for source, target in glossary.items():
        out = out.replace(source, target)
    return out
