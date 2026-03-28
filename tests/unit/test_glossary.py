import json

import pytest

from local_translator.glossary.store import (
    GlossaryError,
    apply_glossary,
    load_glossary,
    protect_glossary_terms_with_stats,
    restore_glossary_terms_with_stats,
)


def test_load_glossary_yaml_schema(tmp_path):
    glossary_path = tmp_path / "glossary.fr-en.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  recette: acceptance testing
  lot: work package
  "maîtrise d'oeuvre": project management
""".strip(),
        encoding="utf-8",
    )

    glossary = load_glossary(glossary_path, source_lang="fr", target_lang="en")
    assert glossary.source_language == "fr"
    assert glossary.target_language == "en"
    assert glossary.entries["recette"] == "acceptance testing"


def test_load_glossary_json_schema(tmp_path):
    glossary_path = tmp_path / "glossary.fr-en.json"
    glossary_path.write_text(
        json.dumps(
            {
                "source_language": "fr",
                "target_language": "en",
                "entries": {
                    "mise en production": "production deployment",
                    "mot de passe": "password",
                },
            }
        ),
        encoding="utf-8",
    )

    glossary = load_glossary(glossary_path)
    assert glossary.entries["mot de passe"] == "password"


def test_load_glossary_invalid_file_fails_clearly(tmp_path):
    glossary_path = tmp_path / "invalid.yaml"
    glossary_path.write_text("entries:\n  valid: ok\n  '': nope", encoding="utf-8")

    with pytest.raises(GlossaryError, match="empty source"):
        load_glossary(glossary_path)


def test_load_glossary_language_mismatch_fails(tmp_path):
    glossary_path = tmp_path / "glossary.yaml"
    glossary_path.write_text(
        """
source_language: fr
target_language: en
entries:
  lot: work package
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(GlossaryError, match="source language mismatch"):
        load_glossary(glossary_path, source_lang="en", target_lang="fr")


def test_apply_glossary_overrides_translation():
    out = apply_glossary("please keep mot de passe as term", {"mot de passe": "password"})
    assert out == "please keep password as term"


def test_apply_glossary_prefers_longer_terms_first():
    out = apply_glossary(
        "maîtrise d'oeuvre et maîtrise",
        {"maîtrise": "mastery", "maîtrise d'oeuvre": "project management"},
    )
    assert out == "project management et mastery"


def test_apply_glossary_avoids_partial_word_corruption():
    out = apply_glossary("catalog cat concatenation", {"cat": "dog"})
    assert out == "catalog dog concatenation"


def test_apply_glossary_replaces_repeated_terms():
    out = apply_glossary("mot de passe, mot de passe", {"mot de passe": "password"})
    assert out == "password, password"


def test_apply_glossary_preserves_simple_title_case():
    out = apply_glossary("Recette validée", {"recette": "acceptance testing"})
    assert out == "Acceptance testing validée"


def test_glossary_protect_and_restore_round_trip():
    protected, token_map, replacements = protect_glossary_terms_with_stats(
        "La recette du lot 2, puis recette finale.",
        {"recette": "acceptance testing", "lot": "work package"},
    )
    assert replacements == 3
    assert "__LT_GLOSSARY_TERM_" in protected

    restored, restored_replacements = restore_glossary_terms_with_stats(protected, token_map)
    assert restored_replacements == 3
    assert restored == "La acceptance testing du work package 2, puis acceptance testing finale."


def test_glossary_restore_handles_normalized_placeholder_variants():
    restored, replacements = restore_glossary_terms_with_stats(
        "The LT GLOSSARY TERM 0000 of LT GLOSSARY TERM 0001 2 starts after validation.",
        {
            "__LT_GLOSSARY_TERM_0000__": "acceptance testing",
            "__LT_GLOSSARY_TERM_0001__": "work package",
        },
    )

    assert replacements == 2
    assert restored == "The acceptance testing of work package 2 starts after validation."
    assert "LT GLOSSARY TERM" not in restored
