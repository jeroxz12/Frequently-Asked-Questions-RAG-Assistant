import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import tiktoken

from src.config import (
    DOCUMENT_PATH,
    INDEX_DIR,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    get_embedding_model,
)
from src.embeddings import generate_embeddings
from src.schemas import Chunk
from src.vector_store import save_index

INTRO_SECTION = "Introducción"
SECTION_PREFIX = "## "
QUESTION_PREFIX = "P: "
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class DocumentLoadError(Exception):
    """Raised when the document cannot be loaded or is empty."""


def load_document(path: Path = DOCUMENT_PATH) -> str:
    """Read the source document as UTF-8 and return its text."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise DocumentLoadError(
            f"Document not found at {path}. Check the path or DOCUMENT_PATH in src/config.py."
        ) from error
    except UnicodeDecodeError as error:
        raise DocumentLoadError(
            f"Document at {path} is not valid UTF-8. Re-save it with UTF-8 encoding."
        ) from error
    except OSError as error:
        raise DocumentLoadError(f"Could not read document at {path}: {error}") from error

    if not text.strip():
        raise DocumentLoadError(f"Document at {path} is empty.")
    return text


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(get_embedding_model())


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text))


def close_block(section: str, lines: list[str], blocks: list[tuple[str, str]]) -> None:
    """Append the accumulated lines as a block (if not blank) and reset them."""
    block_text = "\n".join(lines).strip()
    if block_text:
        blocks.append((section, block_text))
    lines.clear()


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Return (section, block_text) pairs: the intro first, then one block per Q/A pair."""
    blocks: list[tuple[str, str]] = []
    lines: list[str] = []
    section = INTRO_SECTION
    for line in text.splitlines():
        if line.startswith(SECTION_PREFIX):
            close_block(section, lines, blocks)
            section = line.removeprefix(SECTION_PREFIX).strip()
            continue
        if line.startswith(QUESTION_PREFIX):
            close_block(section, lines, blocks)
        lines.append(line)
    close_block(section, lines, blocks)
    return blocks


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split text by sentences into pieces of at most max_tokens (a single longer sentence is kept whole)."""
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_BOUNDARY.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(current)
            candidate = sentence
        current = candidate
    pieces.append(current)
    return pieces


def chunk_document(text: str) -> list[Chunk]:
    """Build one chunk per block, prefixed with its section title, splitting blocks over the max."""
    chunks: list[Chunk] = []
    for section, block_text in split_into_blocks(text):
        max_body_tokens = MAX_CHUNK_TOKENS - count_tokens(f"{section}\n")
        for piece in split_long_text(block_text, max_body_tokens):
            chunk_text = f"{section}\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    section=section,
                    text=chunk_text,
                    token_count=count_tokens(chunk_text),
                )
            )
    return chunks


def print_chunk_summary(chunks: list[Chunk]) -> None:
    token_counts = [chunk.token_count for chunk in chunks]
    out_of_range = [
        chunk.chunk_id
        for chunk in chunks
        if not MIN_CHUNK_TOKENS <= chunk.token_count <= MAX_CHUNK_TOKENS
    ]
    print(f"Chunks: {len(chunks)}")
    print(
        f"Tokens per chunk: min {min(token_counts)}, max {max(token_counts)}, "
        f"avg {sum(token_counts) / len(token_counts):.0f}"
    )
    print(f"Out of range [{MIN_CHUNK_TOKENS}-{MAX_CHUNK_TOKENS}]: {out_of_range or 'none'}")


def print_embedding_summary(embeddings: np.ndarray) -> None:
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embeddings: {embeddings.shape[0]} vectors x {embeddings.shape[1]} dimensions")
    print(f"Vector norms: min {norms.min():.4f}, max {norms.max():.4f}")


def build_index() -> None:
    """Indexing pipeline: load -> chunk -> embed -> save."""
    chunks = chunk_document(load_document())
    print_chunk_summary(chunks)
    embeddings = generate_embeddings([chunk.text for chunk in chunks])
    print_embedding_summary(embeddings)
    save_index(chunks, embeddings)
    print(f"Index saved to {INDEX_DIR}")


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
