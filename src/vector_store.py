import json
from pathlib import Path

import numpy as np

from src.config import INDEX_DIR, MIN_RESULTS, SIMILARITY_THRESHOLD, TOP_K
from src.schemas import Chunk, RetrievedChunk

CHUNKS_FILENAME = "chunks.json"
EMBEDDINGS_FILENAME = "embeddings.npy"


class IndexNotFoundError(Exception):
    """Raised when the index has not been built or is incomplete."""


def save_index(chunks: list[Chunk], embeddings: np.ndarray, index_dir: Path = INDEX_DIR) -> None:
    """Save chunks as JSON and embeddings as a NumPy matrix; row i belongs to chunk i."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings.")
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    chunk_dicts = [chunk.model_dump() for chunk in chunks]
    (index_dir / CHUNKS_FILENAME).write_text(
        json.dumps(chunk_dicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(index_dir / EMBEDDINGS_FILENAME, embeddings)


def load_index(index_dir: Path = INDEX_DIR) -> tuple[list[Chunk], np.ndarray]:
    """Load the chunks and embeddings saved by save_index."""
    chunks_path = Path(index_dir) / CHUNKS_FILENAME
    embeddings_path = Path(index_dir) / EMBEDDINGS_FILENAME
    if not chunks_path.exists() or not embeddings_path.exists():
        raise IndexNotFoundError(
            f"No index found in {index_dir}. Run `python main.py build` first."
        )
    chunk_dicts = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk.model_validate(item) for item in chunk_dicts]
    embeddings = np.load(embeddings_path)
    if len(chunks) != len(embeddings):
        raise IndexNotFoundError(
            f"Index in {index_dir} is inconsistent. Run `python main.py build` again."
        )
    return chunks, embeddings


def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """cos(q, e_i) = (q · e_i) / (||q|| * ||e_i||) for every row e_i of embeddings."""
    dot_products = embeddings @ query_embedding
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    return dot_products / norms


def search_similar_chunks(
    query_embedding: np.ndarray,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    min_results: int = MIN_RESULTS,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[RetrievedChunk]:
    """Exact k-NN: rank all chunks by cosine similarity, keep the top_k above threshold, never fewer than min_results."""
    similarities = cosine_similarity(query_embedding, embeddings)
    ranked = np.argsort(similarities)[::-1][:top_k]
    selected = [i for i in ranked if similarities[i] >= threshold]
    if len(selected) < min_results:
        selected = list(ranked[:min_results])
    return [
        RetrievedChunk(
            chunk_id=chunks[i].chunk_id,
            section=chunks[i].section,
            similarity=round(float(similarities[i]), 4),
            text=chunks[i].text,
        )
        for i in selected
    ]
