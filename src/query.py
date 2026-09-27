import sys
from pathlib import Path

from openai import OpenAI, OpenAIError

from src.config import INDEX_DIR, get_openai_model
from src.embeddings import create_client, embed_query
from src.schemas import GeneratedAnswer, QueryResult, RetrievedChunk
from src.vector_store import load_index, search_similar_chunks

NO_ANSWER = "No encontré esa información en la documentación de Nexo HR."

SYSTEM_PROMPT = f"""Eres el asistente de soporte de Nexo HR, una plataforma SaaS de recursos humanos.
Respondes preguntas usando EXCLUSIVAMENTE los fragmentos de documentación que recibes como contexto.

Reglas:
- Usa solo información que aparezca en el contexto. No inventes datos, plazos, precios ni pasos.
- Si el contexto no alcanza para responder, responde exactamente: "{NO_ANSWER}"
- Si la respuesta tiene pasos, enuméralos en orden.
- Responde en español, de forma clara y breve."""


class GenerationError(Exception):
    """Raised when the LLM call fails."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Join the retrieved chunks into one numbered context block for the prompt."""
    return "\n\n".join(
        f"[Fragmento {position} | Sección: {chunk.section}]\n{chunk.text}"
        for position, chunk in enumerate(chunks, start=1)
    )


def generate_answer(question: str, context: str, client: OpenAI) -> str:
    """Ask the LLM to answer the question grounded only on the given context."""
    try:
        response = client.responses.parse(
            model=get_openai_model(),
            instructions=SYSTEM_PROMPT,
            input=f"Contexto:\n{context}\n\nPregunta: {question}",
            text_format=GeneratedAnswer,
            temperature=0,
        )
    except OpenAIError as error:
        raise GenerationError(f"LLM request failed: {error}") from error
    if response.output_parsed is None:
        raise GenerationError("LLM returned no parsable answer.")
    return response.output_parsed.answer


def answer_question(
    question: str, client: OpenAI | None = None, index_dir: Path = INDEX_DIR
) -> QueryResult:
    """Query pipeline: embed question -> search chunks -> build context -> generate answer."""
    question = question.strip()
    if not question:
        raise ValueError("The question cannot be empty.")
    chunks, embeddings = load_index(index_dir)
    client = client or create_client()
    related = search_similar_chunks(embed_query(question, client), chunks, embeddings)
    answer = generate_answer(question, build_context(related), client)
    return QueryResult(user_question=question, system_answer=answer, chunks_related=related)


def main() -> None:
    result = answer_question(" ".join(sys.argv[1:]))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
