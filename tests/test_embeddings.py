from types import SimpleNamespace

import numpy as np
import pytest
from openai import OpenAIError

from src.config import ConfigurationError
from src.embeddings import EmbeddingError, embed_query, generate_embeddings


class FakeEmbeddingsAPI:
    """Returns [i, 1.0] for the i-th input, in reverse order, like an API that doesn't guarantee order."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        if self.error:
            raise self.error
        data = [SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))


def make_client(error: Exception | None = None):
    return SimpleNamespace(embeddings=FakeEmbeddingsAPI(error))


def test_generate_embeddings_returns_one_row_per_text_in_order():
    client = make_client()

    embeddings = generate_embeddings(["a", "b", "c"], client)

    assert embeddings.shape == (3, 2)
    assert embeddings.dtype == np.float32
    assert embeddings[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_generate_embeddings_sends_all_texts_in_one_request():
    client = make_client()

    generate_embeddings(["a", "b"], client)

    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == ["a", "b"]


def test_generate_embeddings_wraps_api_errors():
    client = make_client(error=OpenAIError("rate limit"))

    with pytest.raises(EmbeddingError, match="rate limit"):
        generate_embeddings(["a"], client)


def test_generate_embeddings_rejects_empty_input():
    with pytest.raises(ValueError):
        generate_embeddings([], make_client())


def test_generate_embeddings_without_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        generate_embeddings(["a"])


def test_embed_query_returns_a_single_vector():
    vector = embed_query("¿Cuántos días de vacaciones tengo?", make_client())

    assert vector.shape == (2,)
