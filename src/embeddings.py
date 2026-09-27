import numpy as np
from openai import OpenAI, OpenAIError

from src.config import get_embedding_model, get_openai_api_key


class EmbeddingError(Exception):
    """Raised when the embeddings API call fails."""


def create_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def generate_embeddings(texts: list[str], client: OpenAI | None = None) -> np.ndarray:
    """Return a (len(texts), dimensions) float32 matrix, one row per text, in the same order."""
    if not texts:
        raise ValueError("generate_embeddings needs at least one text.")
    client = client or create_client()
    try:
        response = client.embeddings.create(model=get_embedding_model(), input=texts)
    except OpenAIError as error:
        raise EmbeddingError(f"Embeddings API request failed: {error}") from error
    ordered = sorted(response.data, key=lambda item: item.index)
    return np.array([item.embedding for item in ordered], dtype=np.float32)


def embed_query(question: str, client: OpenAI | None = None) -> np.ndarray:
    """Return the embedding vector of a single question."""
    return generate_embeddings([question], client)[0]
