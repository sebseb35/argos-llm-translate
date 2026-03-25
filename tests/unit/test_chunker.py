from local_translator.pipeline.chunker import segment_text


def test_segment_text_basic():
    text = "Bonjour. Comment ça va ? Très bien!"
    segments = segment_text(text)
    assert len(segments) == 3
