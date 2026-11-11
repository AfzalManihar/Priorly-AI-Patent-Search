from app.corpus import split_text


def test_split_text_respects_max_chars_for_long_sentence():
    text = " ".join(["patentclaim"] * 1000)
    chunks = split_text(text, max_chars=1200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1200 for chunk in chunks)
    assert " ".join(chunks) == text


def test_split_text_preserves_long_single_token():
    text = "A" * 3000
    chunks = split_text(text, max_chars=1200)

    assert all(len(chunk) <= 1200 for chunk in chunks)
    assert "".join(chunks) == text
