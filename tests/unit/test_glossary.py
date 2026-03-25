from local_translator.glossary.store import apply_glossary


def test_apply_glossary_replace():
    out = apply_glossary("mot de passe oublié", {"mot de passe": "password"})
    assert "password" in out
