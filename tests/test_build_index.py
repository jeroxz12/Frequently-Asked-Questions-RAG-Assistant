import pytest

from src.build_index import (
    DocumentLoadError,
    chunk_document,
    count_tokens,
    load_document,
    split_into_blocks,
    split_long_text,
)
from src.config import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS

MINI_DOCUMENT = """NEXO HR — FAQ
Intro del documento.

## Cuenta y acceso

P: ¿Cómo activo mi cuenta?
R: Sigue estos pasos:
1. Abre el correo.
2. Define una contraseña.

P: ¿Hay 2FA?
R: Sí.

## Nómina

P: ¿Cuándo cobro?
R: El último día hábil.
"""


def test_load_document_returns_text(tmp_path):
    document = tmp_path / "faq.txt"
    document.write_text("P: ¿Hola?\nR: Chau.", encoding="utf-8")

    assert load_document(document) == "P: ¿Hola?\nR: Chau."


def test_load_document_raises_error_if_document_not_found(tmp_path):
    document = tmp_path / "faq.txt"

    with pytest.raises(DocumentLoadError, match="Document not found at"):
        load_document(document)


def test_load_document_raises_error_if_document_has_invalid_utf8_encoding(tmp_path):
    document = tmp_path / "faq.txt"
    document.write_text("P: ¿Hola?\nR: Chau.", encoding="latin-1")

    with pytest.raises(DocumentLoadError, match="is not valid UTF-8."):
        load_document(document)


def test_load_document_raises_error_if_document_is_empty(tmp_path):
    document = tmp_path / "faq.txt"
    document.write_text("  \n\n\t  ", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="is empty."):
        load_document(document)


def test_split_into_blocks_keeps_intro_and_each_qa_pair_with_its_section():
    assert split_into_blocks(MINI_DOCUMENT) == [
        ("Introducción", "NEXO HR — FAQ\nIntro del documento."),
        (
            "Cuenta y acceso",
            "P: ¿Cómo activo mi cuenta?\nR: Sigue estos pasos:\n"
            "1. Abre el correo.\n2. Define una contraseña.",
        ),
        ("Cuenta y acceso", "P: ¿Hay 2FA?\nR: Sí."),
        ("Nómina", "P: ¿Cuándo cobro?\nR: El último día hábil."),
    ]


def test_split_long_text_keeps_short_text_whole():
    assert split_long_text("Una oración corta.", max_tokens=100) == ["Una oración corta."]


def test_split_long_text_splits_by_sentences_under_the_limit():
    text = " ".join(f"Esta es la oración número {n}." for n in range(20))

    pieces = split_long_text(text, max_tokens=30)

    assert len(pieces) > 1
    assert all(count_tokens(piece) <= 30 for piece in pieces)
    assert " ".join(pieces) == text


@pytest.fixture(scope="module")
def real_chunks():
    return chunk_document(load_document())


def test_real_document_produces_at_least_20_chunks(real_chunks):
    assert len(real_chunks) >= 20


def test_real_document_chunks_are_within_token_limits(real_chunks):
    for chunk in real_chunks:
        assert MIN_CHUNK_TOKENS <= chunk.token_count <= MAX_CHUNK_TOKENS, chunk.chunk_id


def test_real_document_chunk_ids_are_sequential(real_chunks):
    assert [chunk.chunk_id for chunk in real_chunks] == list(range(len(real_chunks)))


def test_real_document_is_fully_covered_by_chunks(real_chunks):
    all_chunk_text = "\n".join(chunk.text for chunk in real_chunks)
    for line in load_document().splitlines():
        content = line.removeprefix("## ").strip()
        assert content in all_chunk_text, line
