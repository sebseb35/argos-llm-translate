# License notes and compliance strategy

## Summary

This project is designed as a **standalone translator product** that uses `argostranslate` as a translation engine dependency, without merging repositories.

## Why side-by-side and not merge

- `argostranslate` is used as an engine API dependency (translation runtime).
- `argos-translate-gui` and `argos-translate-files` are separate projects.
- `argos-translate-files` is AGPL-3.0; we do **not** copy code from it.
- We build our own extraction/reconstruction pipeline for `.md` and `.xlsx` and other formats in this repository.

## What we use from Argos

- Python package dependency: `argostranslate` runtime APIs for local model-based translation.
- User-managed installation of language packages.

## What we do not use

- No source code copy/paste from `argos-translate-files`.
- No repo merge with Argos repositories.

## AGPL implication reminder

If code is copied from AGPL projects, AGPL obligations could apply to the resulting derivative work. To avoid this, this project keeps a clean-room implementation for file format handling.

## Recommended repository license

- **MIT** for this repository (code and docs authored here).
- Keep third-party dependencies under their own licenses.
- Distribute clear notices for models and language packs.

## Model and package licensing caveats

- Argos translation models may include their own terms.
- Local LLM model files (`.gguf`) have model-specific licenses.
- Enterprise deployments should validate model licenses before internal redistribution.
