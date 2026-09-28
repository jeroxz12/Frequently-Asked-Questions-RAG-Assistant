from fastapi import FastAPI, HTTPException

from src.config import ConfigurationError
from src.embeddings import EmbeddingError
from src.query import GenerationError, answer_question
from src.schemas import AskRequest, QueryResult
from src.vector_store import IndexNotFoundError, load_index

app = FastAPI(title="FAQ RAG Assistant", version="1.0.0")


@app.get("/health")
def health_check() -> dict:
    """Report whether the index is built and how many chunks it has."""
    try:
        chunks, _ = load_index()
    except IndexNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ready", "chunks_loaded": len(chunks)}


@app.post("/ask", response_model=QueryResult)
def ask(request: AskRequest) -> QueryResult:
    """Answer a question with the same RAG pipeline as the CLI."""
    try:
        return answer_question(request.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except IndexNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (EmbeddingError, GenerationError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ConfigurationError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
