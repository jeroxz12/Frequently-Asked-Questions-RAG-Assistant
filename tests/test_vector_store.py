import numpy as np
import pytest

from src.schemas import Chunk
from src.vector_store import (
    IndexNotFoundError,
    cosine_similarity,
    load_index,
    save_index,
    search_similar_chunks,
)

TOY_EMBEDDINGS = np.array(
    [
        [1.0, 0.0],   # chunk 0: same direction as the query
        [0.8, 0.6],   # chunk 1: close
        [0.0, 1.0],   # chunk 2: orthogonal
        [-1.0, 0.0],  # chunk 3: opposite
        [0.6, 0.8],   # chunk 4: somewhat close
        [0.9, 0.1],   # chunk 5: very close, not normalized
    ]
)
QUERY = np.array([2.0, 0.0])


def make_chunks(count: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=i, section="Vacaciones", text=f"Texto {i} con ñ y tildes: días", token_count=10)
        for i in range(count)
    ]


def test_save_and_load_index_round_trip(tmp_path):
    chunks = make_chunks(3)
    embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)

    save_index(chunks, embeddings, tmp_path)
    loaded_chunks, loaded_embeddings = load_index(tmp_path)

    assert loaded_chunks == chunks
    np.testing.assert_array_equal(loaded_embeddings, embeddings)


def test_save_index_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="2 chunks but 3 embeddings"):
        save_index(make_chunks(2), np.zeros((3, 2)), tmp_path)


def test_load_index_raises_if_index_was_not_built(tmp_path):
    with pytest.raises(IndexNotFoundError, match="Run `python main.py build`"):
        load_index(tmp_path / "missing")


def test_cosine_similarity_depends_on_direction_not_length():
    similarities = cosine_similarity(QUERY, TOY_EMBEDDINGS)

    np.testing.assert_allclose(similarities, [1.0, 0.8, 0.0, -1.0, 0.6, 0.9939], atol=1e-4)


def search_ids(**kwargs) -> list[int]:
    results = search_similar_chunks(QUERY, make_chunks(len(TOY_EMBEDDINGS)), TOY_EMBEDDINGS, **kwargs)
    return [result.chunk_id for result in results]


def test_search_returns_most_similar_first_limited_to_top_k():
    assert search_ids(top_k=3, threshold=-1.0) == [0, 5, 1]


def test_search_drops_chunks_below_threshold():
    assert search_ids(top_k=5, threshold=0.7) == [0, 5, 1]


def test_search_returns_at_least_min_results_even_below_threshold():
    assert search_ids(top_k=5, min_results=2, threshold=0.999) == [0, 5]


def test_search_results_include_similarity_and_chunk_data():
    results = search_similar_chunks(QUERY, make_chunks(6), TOY_EMBEDDINGS, top_k=2, threshold=0.0)

    assert results[0].similarity == 1.0
    assert results[0].section == "Vacaciones"
    assert results[0].text == "Texto 0 con ñ y tildes: días"
