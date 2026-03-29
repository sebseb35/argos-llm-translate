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

# text mode with glossary
local-translator text --from fr --to en --engine argos --glossary examples/glossary.example.yaml --content "mot de passe oublié"

# file mode
local-translator translate input.docx --from fr --to en --output output.docx --engine argos

# file mode with glossary
local-translator translate input.docx --from fr --to en --output output.docx --engine hybrid --llm-model ./models/model.gguf --glossary examples/glossary.example.yaml

# hybrid mode (Argos + local post-edit)
local-translator translate input.md --from en --to fr --output out.md --engine hybrid --llm-model ./models/model.gguf

# preview extracted segment count
local-translator preview input.pdf

# text mode with execution report (+ optional JSON export)
local-translator text --from fr --to en --engine argos --content "Bonjour" --report --report-json run-report.json

# GUI reserved for V2
local-translator gui
```

## Python API (stable entry points)

The internal translation API is exposed for GUI and automation use via:

- `local_translator.translate_text(...)`
- `local_translator.translate_file(...)`
- `local_translator.preview_file(...)`

These functions validate inputs, execute translation, and return structured dataclass outputs (translated text/path, report metrics, warnings). The CLI is intentionally a thin wrapper over these API functions.

Example (text):

```python
from local_translator import translate_text

result = translate_text(
    "Bonjour le monde",
    source_lang="fr",
    target_lang="en",
    engine="argos",
    report=True,
)

print(result.translated_text)
print(result.report.segment_count)
```

Example (file):

```python
from pathlib import Path
from local_translator import translate_file

result = translate_file(
    input_path=Path("input.docx"),
    output_path=Path("output.docx"),
    source_lang="fr",
    target_lang="en",
    engine="argos",
    report=True,
)

print(result.output_path)
print(result.report.fallback_count)
```

## Execution reporting

Use `--report` on `text` and `translate` commands to print a structured execution summary:

- segments processed,
- fallback events (LLM -> Argos),
- error count,
- glossary replacement count,
- elapsed processing time.

Use `--report-json <path>` together with `--report` to export the same metrics as JSON for auditing.

Hybrid reports now include routing/chunk tuning metrics:

- skip/safe/smart segment counts,
- routing reasons per segment (for example `short_plain_segment`, `technical_token_density`, `multi_sentence_prose`),
- chunk planning traces (`segment_indices`, merge/boundary reasons, placeholder/char counts),
- validation failure counters (including placeholder mismatch counters),
- average LLM latency per segment/chunk and estimated calls saved by chunking.

You can summarize multiple JSON reports with:

```bash
local-translator report-summary run-a.json run-b.json run-c.json
```

This gives a lightweight feedback loop for threshold tuning on long documents.

### Heuristic tuning knobs (centralized in `LLMSettings`)

Initial conservative defaults:

- `skip_short_characters=48`
- `skip_high_placeholder_ratio=0.12`
- `routing_technical_token_threshold=1`
- `routing_safe_placeholder_count=2`
- `routing_multi_sentence_threshold=2`
- `smart_min_chars=160`
- `chunk_max_placeholders_per_segment=1`
- `chunk_max_segments=4`
- `chunk_max_chars=560`

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



## Glossary support (YAML/JSON)

Glossary usage is optional and enabled per run with `--glossary`. Supported formats:

- `.yaml` / `.yml`
- `.json`

Schema:

```yaml
source_language: fr      # optional but recommended
target_language: en      # optional but recommended
entries:
  "recette": "acceptance testing"
  "lot": "work package"
  "maîtrise d'oeuvre": "project management"
```

Behavior and precedence:

- glossary terms are applied deterministically using longest source match first,
- replacements avoid changing substrings inside larger words (for word-like terms),
- glossary is enforced before and after optional LLM post-editing,
- in strict hybrid validation, glossary-protected terms are preserved and candidate edits that remove them are rejected.

Validation and errors:

- invalid structures fail with a clear `GlossaryError`,
- empty source/target terms are rejected,
- when `source_language`/`target_language` are provided, they must match the CLI `--from`/`--to` values.

Current limitations:

- matching is case-sensitive,
- duplicate keys in YAML/JSON files are parser-defined and may overwrite earlier values,
- this is exact string matching only (no fuzzy/semantic terminology memory).

## Known limits

- PDF strategy is text-native only: scanned/image PDFs and OCR workflows are unsupported.
- Complex layouts may not round-trip perfectly (`.pptx`, `.pdf`, complex `.docx`).
- Very large documents may require improved segmentation strategy in future versions.
- Minimum CPU requirements: SSE4.1 required (Argos / CTranslate2 dependency)

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
