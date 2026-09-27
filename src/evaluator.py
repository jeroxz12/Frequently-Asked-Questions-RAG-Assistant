from openai import OpenAI, OpenAIError

from src.config import get_openai_model
from src.embeddings import create_client
from src.query import NO_ANSWER, build_context
from src.schemas import Evaluation, JudgeOutput, QueryResult

MAX_RELEVANCE = 3
MAX_FAITHFULNESS = 4
MAX_COMPLETENESS = 3

EVALUATOR_PROMPT = f"""Eres un evaluador estricto de un asistente RAG de soporte.
Recibes una pregunta, los fragmentos recuperados (contexto) y la respuesta del sistema.
Califica tres dimensiones con números enteros:

- relevance (0-{MAX_RELEVANCE}): ¿los fragmentos recuperados sirven para responder la pregunta?
- faithfulness (0-{MAX_FAITHFULNESS}): ¿cada afirmación de la respuesta está respaldada por el contexto?
  Resta por cada dato inventado (plazos, precios, pasos) que no aparezca en los fragmentos.
- completeness (0-{MAX_COMPLETENESS}): ¿la respuesta cubre todo lo que pregunta el usuario con la información disponible?

Si el contexto no contiene la información y la respuesta dice "{NO_ANSWER}",
eso es el comportamiento correcto: faithfulness y completeness van al máximo, y relevance refleja los fragmentos.

En reason explica en español, en 1 a 3 oraciones, qué justifica cada nota."""


class EvaluationError(Exception):
    """Raised when the evaluator LLM call fails."""


def build_evaluation_input(result: QueryResult) -> str:
    return (
        f"Pregunta: {result.user_question}\n\n"
        f"Contexto:\n{build_context(result.chunks_related)}\n\n"
        f"Respuesta del sistema: {result.system_answer}"
    )


def request_judgement(result: QueryResult, client: OpenAI) -> JudgeOutput:
    try:
        response = client.responses.parse(
            model=get_openai_model(),
            instructions=EVALUATOR_PROMPT,
            input=build_evaluation_input(result),
            text_format=JudgeOutput,
            temperature=0,
        )
    except OpenAIError as error:
        raise EvaluationError(f"Evaluator request failed: {error}") from error
    if response.output_parsed is None:
        raise EvaluationError("Evaluator returned no parsable judgement.")
    return response.output_parsed


def clamp(value: int, maximum: int) -> int:
    return max(0, min(value, maximum))


def to_evaluation(judgement: JudgeOutput) -> Evaluation:
    """Sum the per-dimension grades into a 0-10 score and prefix the breakdown to the reason."""
    relevance = clamp(judgement.relevance, MAX_RELEVANCE)
    faithfulness = clamp(judgement.faithfulness, MAX_FAITHFULNESS)
    completeness = clamp(judgement.completeness, MAX_COMPLETENESS)
    score = relevance + faithfulness + completeness
    breakdown = (
        f"Puntaje {score}: relevancia {relevance}/{MAX_RELEVANCE}, "
        f"fidelidad {faithfulness}/{MAX_FAITHFULNESS}, completitud {completeness}/{MAX_COMPLETENESS}."
    )
    return Evaluation(score=score, reason=f"{breakdown} {judgement.reason.strip()}")


def evaluate_answer(result: QueryResult, client: OpenAI | None = None) -> Evaluation:
    """LLM-as-judge: grade retrieval relevance, faithfulness to the context and completeness."""
    judgement = request_judgement(result, client or create_client())
    return to_evaluation(judgement)
