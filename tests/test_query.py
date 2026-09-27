from types import SimpleNamespace

import numpy as np
import pytest
from openai import OpenAIError

from src.query import GenerationError, answer_question, build_context, generate_answer
from src.schemas import Chunk, GeneratedAnswer, QueryResult, RetrievedChunk
from src.vector_store import IndexNotFoundError, save_index


class FakeClient:
    """Embeds every question as [1, 0] and returns a fixed answer from the LLM."""

    def __init__(self, llm_error: Exception | None = None):
        self.llm_error = llm_error
        self.llm_calls = []
        self.embeddings = SimpleNamespace(create=self.create_embeddings)
        self.responses = SimpleNamespace(parse=self.parse)

    def create_embeddings(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])

    def parse(self, **kwargs):
        self.llm_calls.append(kwargs)
        if self.llm_error:
            raise self.llm_error
        return SimpleNamespace(output_parsed=GeneratedAnswer(answer="Tienes 15 días."))


def make_retrieved(chunk_id: int, section: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, section=section, similarity=0.6, text=text)


@pytest.fixture
def index_dir(tmp_path):
    texts = ["Vacaciones: 15 días.", "Vacaciones: pedir con anticipación.", "Soporte: 8 a 20."]
    chunks = [
        Chunk(chunk_id=i, section="Sección", text=text, token_count=5)
        for i, text in enumerate(texts)
    ]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.4], [0.0, 1.0]], dtype=np.float32)
    save_index(chunks, embeddings, tmp_path)
    return tmp_path


def test_build_context_numbers_chunks_and_includes_section():
    context = build_context(
        [make_retrieved(3, "Vacaciones", "Texto A"), make_retrieved(8, "Nómina", "Texto B")]
    )

    assert context == (
        "[Fragmento 1 | Sección: Vacaciones]\nTexto A\n\n[Fragmento 2 | Sección: Nómina]\nTexto B"
    )


def test_generate_answer_sends_question_and_context_to_llm():
    client = FakeClient()

    answer = generate_answer("¿Cuántos días?", "CONTEXTO", client)

    assert answer == "Tienes 15 días."
    sent_input = client.llm_calls[0]["input"]
    assert "CONTEXTO" in sent_input and "¿Cuántos días?" in sent_input
    assert client.llm_calls[0]["text_format"] is GeneratedAnswer


def test_generate_answer_wraps_api_errors():
    with pytest.raises(GenerationError, match="timeout"):
        generate_answer("¿?", "ctx", FakeClient(llm_error=OpenAIError("timeout")))


def test_answer_question_returns_exactly_three_keys(index_dir):
    result = answer_question("¿Cuántos días de vacaciones tengo?", FakeClient(), index_dir)

    assert isinstance(result, QueryResult)
    assert set(result.model_dump()) == {"user_question", "system_answer", "chunks_related"}
    assert result.system_answer == "Tienes 15 días."


def test_answer_question_returns_most_similar_chunks_first(index_dir):
    result = answer_question("¿Cuántos días de vacaciones tengo?", FakeClient(), index_dir)

    assert [chunk.chunk_id for chunk in result.chunks_related] == [0, 1]
    assert 2 <= len(result.chunks_related) <= 5


def test_answer_question_rejects_empty_question(index_dir):
    with pytest.raises(ValueError, match="empty"):
        answer_question("   ", FakeClient(), index_dir)


def test_answer_question_raises_if_index_missing(tmp_path):
    with pytest.raises(IndexNotFoundError):
        answer_question("¿Hola?", FakeClient(), tmp_path / "missing")


def test_query_result_rejects_extra_keys():
    with pytest.raises(ValueError):
        QueryResult(
            user_question="q",
            system_answer="a",
            chunks_related=[make_retrieved(0, "s", "t"), make_retrieved(1, "s", "t")],
            score=8,
        )


def test_query_result_requires_between_2_and_5_chunks():
    with pytest.raises(ValueError):
        QueryResult(user_question="q", system_answer="a", chunks_related=[make_retrieved(0, "s", "t")])
