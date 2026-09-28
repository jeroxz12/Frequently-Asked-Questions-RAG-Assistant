from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A fragment of the source document, created once at indexing time."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: int = Field(ge=0)
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)


class RetrievedChunk(BaseModel):
    """A chunk returned by the search, with its similarity to one specific question."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: int = Field(ge=0)
    section: str
    similarity: float
    text: str


class AskRequest(BaseModel):
    """Body of POST /ask."""

    model_config = ConfigDict(extra="forbid")

    question: str


class GeneratedAnswer(BaseModel):
    """Structured Output the LLM must return."""

    model_config = ConfigDict(extra="forbid")

    answer: str


class QueryResult(BaseModel):
    """Final output of a query: exactly these three keys."""

    model_config = ConfigDict(extra="forbid")

    user_question: str = Field(min_length=1)
    system_answer: str = Field(min_length=1)
    chunks_related: list[RetrievedChunk] = Field(min_length=2, max_length=5)


class JudgeOutput(BaseModel):
    """Structured Output the evaluator LLM must return: one grade per dimension."""

    model_config = ConfigDict(extra="forbid")

    relevance: int
    faithfulness: int
    completeness: int
    reason: str


class Evaluation(BaseModel):
    """Evaluator result, stored apart from QueryResult to keep its three keys."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=50)
