from fastapi.testclient import TestClient

import src.api as api
from src.query import GenerationError
from src.schemas import QueryResult, RetrievedChunk
from src.vector_store import IndexNotFoundError

client = TestClient(api.app)


def raise_missing_index(*args, **kwargs):
    raise IndexNotFoundError("No index found. Run `python main.py build` first.")


def fake_answer_question(question: str) -> QueryResult:
    chunk = RetrievedChunk(chunk_id=10, section="Vacaciones", similarity=0.68, text="14 días.")
    return QueryResult(user_question=question, system_answer="Tienes 14 días.", chunks_related=[chunk, chunk])


def test_health_returns_200_with_chunk_count(monkeypatch):
    monkeypatch.setattr(api, "load_index", lambda: (["chunk"] * 34, None))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "chunks_loaded": 34}


def test_health_returns_503_when_index_is_missing(monkeypatch):
    monkeypatch.setattr(api, "load_index", raise_missing_index)

    response = client.get("/health")

    assert response.status_code == 503
    assert "python main.py build" in response.json()["detail"]


def test_ask_returns_query_result_with_three_keys(monkeypatch):
    monkeypatch.setattr(api, "answer_question", fake_answer_question)

    response = client.post("/ask", json={"question": "¿Cuántos días de vacaciones tengo?"})

    assert response.status_code == 200
    assert set(response.json()) == {"user_question", "system_answer", "chunks_related"}


def test_ask_returns_400_for_empty_question():
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400


def test_ask_returns_422_when_body_is_invalid():
    response = client.post("/ask", json={"pregunta": "hola"})

    assert response.status_code == 422


def test_ask_returns_503_when_index_is_missing(monkeypatch):
    monkeypatch.setattr(api, "answer_question", raise_missing_index)

    response = client.post("/ask", json={"question": "¿Hola?"})

    assert response.status_code == 503


def test_ask_returns_502_when_openai_fails(monkeypatch):
    def raise_generation_error(question):
        raise GenerationError("LLM request failed: timeout")

    monkeypatch.setattr(api, "answer_question", raise_generation_error)

    response = client.post("/ask", json={"question": "¿Hola?"})

    assert response.status_code == 502
