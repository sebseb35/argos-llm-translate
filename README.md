# local-translator

Open-source **offline** translation pipeline for sensitive professional content (FR↔EN), built for CPU-only environments.

## Product vision

`local-translator` is not a simple Argos wrapper. It provides:

- a modular document translation pipeline (extract → segment → translate → optional post-edit → reconstruct),
- enterprise-oriented local processing (no cloud required),
- support for `.txt`, `.md`, `.docx`, `.pptx`, `.xlsx`, and text-native `.pdf` (best effort),
- optional constrained LLM post-editing via `llama-cpp-python`,
- architecture designed so a future GUI uses the same core pipeline.

## Candidate names considered

1. `local-translator` / CLI `local-translator`
2. `secure-doc-translate` / CLI `secure-translate`
3. `argonova-translate` / CLI `argonova`

**Selected:** `local-translator` (clear, neutral, easy to adopt).

## Why side-by-side with Argos (and no merge)

- We depend on `argostranslate` as a translation **engine**.
- We keep this repository independent from `argos-translate`, `argos-translate-gui`, and `argos-translate-files`.
- We do not copy source code from `argos-translate-files` (AGPL-3.0).
- We implement our own format extraction/reconstruction pipeline, especially for `.md` and `.xlsx`.

See: `docs/license-notes.md`.

## Architecture

```text
src/local_translator/
  cli.py
  config.py
  logging_utils.py
  models/types.py
  pipeline/
    chunker.py
    postedit.py
    translator.py
  engines/
    argos_engine.py
    llm_engine.py
  extractors/
    base.py
    txt.py
    md.py
    docx.py
    pptx.py
    xlsx.py
    pdf_text.py
  reconstructors/registry.py
  glossary/store.py
  gui/stub.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# optional LLM support
pip install -e .[llm]
```

## Quickstart

```bash
# text mode
local-translator text --from fr --to en --engine argos --content "Bonjour le monde"

# file mode
local-translator translate input.docx --from fr --to en --output output.docx --engine argos

# hybrid mode (Argos + local post-edit)
local-translator translate input.md --from en --to fr --output out.md --engine hybrid --llm-model ./models/model.gguf

# preview extracted segment count
local-translator preview input.pdf

# GUI reserved for V2
local-translator gui
```

## V1 supported formats

- `.txt`: full support
- `.md`: line-based best effort preserving markdown structure
- `.docx`: paragraph-focused support
- `.pptx`: text in shapes/placeholders (best effort)
- `.xlsx`: textual cells only; formulas, numbers, non-text cells preserved
- `.pdf`: text-native extraction only; translated output is a `.translated.txt` sidecar (no PDF reconstruction)



## PDF translation strategy

PDF support is intentionally narrow and explicit:

- **Supported input:** PDFs with native selectable text.
- **Unsupported input:** scanned/image-only PDFs and OCR-based workflows.
- **Pipeline:** extract text blocks -> segment/chunk -> translate -> optional post-edit -> write sidecar text.
- **Output:** translated text sidecar file (recommended naming: `input.translated.txt`).
- **Non-goal:** no layout-preserving or page-faithful translated PDF reconstruction.

When extraction yields no meaningful text (for example blank or image-like PDFs), translation fails with an explicit error instead of producing low-quality output.

## Known limits

- PDF strategy is text-native only: scanned/image PDFs and OCR workflows are unsupported.
- Complex layouts may not round-trip perfectly (`.pptx`, `.pdf`, complex `.docx`).
- Very large documents may require improved segmentation strategy in future versions.

## Confidentiality and security

- Designed for offline operation.
- No mandatory cloud API.
- Sensitive content can remain inside the company network.

## License notes

This repository is MIT-licensed. See `LICENSE` and `docs/license-notes.md` for third-party licensing and side-by-side strategy details.

## GUI roadmap (V2)

Planned features:

- drag-and-drop file translation,
- source/target language selectors,
- engine mode selection (`argos`, `hybrid`),
- text area translation like web translators,
- glossary and run history,
- live processing logs.
