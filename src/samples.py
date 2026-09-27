import json
from pathlib import Path

from src.config import OUTPUTS_DIR
from src.embeddings import create_client
from src.evaluator import evaluate_answer
from src.query import answer_question
from src.schemas import Evaluation, QueryResult, RetrievedChunk

SAMPLES_PATH = OUTPUTS_DIR / "sample_queries.json"
EVALUATIONS_PATH = OUTPUTS_DIR / "sample_evaluations.json"

# (question, keywords): a retrieved chunk counts as relevant if it contains any keyword.
# An empty keyword list marks an out-of-scope question, excluded from the relevance check.
SAMPLE_QUESTIONS: list[tuple[str, list[str]]] = [
    ("¿Cuántos días de vacaciones tengo?", ["vacaciones"]),
    ("¿Cómo procesa un administrador la nómina?", ["nómina", "sueldo"]),
    ("¿Puedo activar la verificación en dos pasos?", ["verificación en dos pasos"]),
    ("Me olvidé de fichar, ¿cómo lo corrijo?", ["fichaje", "fichar"]),
    ("¿Cuánto cuesta el plan Business?", ["plan"]),
    ("¿Cuál es la capital de Francia?", []),
]


def is_relevant(chunk: RetrievedChunk, keywords: list[str]) -> bool:
    text = chunk.text.lower()
    return any(keyword.lower() in text for keyword in keywords)


def count_relevant(result: QueryResult, keywords: list[str]) -> int:
    return sum(is_relevant(chunk, keywords) for chunk in result.chunks_related)


def save_results(results: list[QueryResult], path: Path = SAMPLES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_relevance_report(results: list[QueryResult]) -> None:
    relevant_total = retrieved_total = 0
    for result, (_, keywords) in zip(results, SAMPLE_QUESTIONS):
        if not keywords:
            print(f"- {result.user_question} -> out of scope: {result.system_answer}")
            continue
        relevant = count_relevant(result, keywords)
        retrieved = len(result.chunks_related)
        relevant_total += relevant
        retrieved_total += retrieved
        print(f"- {result.user_question} -> {relevant}/{retrieved} relevant chunks")
    print(f"Overall relevance: {relevant_total}/{retrieved_total} = {relevant_total / retrieved_total:.0%}")


def save_evaluations(
    results: list[QueryResult], evaluations: list[Evaluation], path: Path = EVALUATIONS_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"user_question": result.user_question, "evaluation": evaluation.model_dump()}
        for result, evaluation in zip(results, evaluations)
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_samples() -> None:
    """Answer and evaluate every sample question, save both files and print the reports."""
    client = create_client()
    results = [answer_question(question, client) for question, _ in SAMPLE_QUESTIONS]
    save_results(results)
    print(f"Saved {len(results)} examples to {SAMPLES_PATH}")
    print_relevance_report(results)
    evaluations = [evaluate_answer(result, client) for result in results]
    save_evaluations(results, evaluations)
    print(f"Saved evaluations to {EVALUATIONS_PATH}")
    for evaluation in evaluations:
        print(f"- {evaluation.reason}")
