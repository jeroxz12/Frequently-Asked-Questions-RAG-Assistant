from types import SimpleNamespace

import pytest
from openai import OpenAIError

from src.evaluator import (
    EvaluationError,
    build_evaluation_input,
    evaluate_answer,
    to_evaluation,
)
from src.schemas import Evaluation, JudgeOutput, QueryResult, RetrievedChunk


def make_result() -> QueryResult:
    chunks = [
        RetrievedChunk(chunk_id=10, section="Vacaciones", similarity=0.68, text="14 días hasta 5 años."),
        RetrievedChunk(chunk_id=11, section="Vacaciones", similarity=0.58, text="Pedir con 15 días."),
    ]
    return QueryResult(
        user_question="¿Cuántos días de vacaciones tengo?",
        system_answer="Tienes 14 días hasta 5 años de antigüedad.",
        chunks_related=chunks,
    )


def make_client(judgement: JudgeOutput | None = None, error: Exception | None = None):
    def parse(**kwargs):
        if error:
            raise error
        return SimpleNamespace(output_parsed=judgement)

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


def test_evaluation_input_includes_question_context_and_answer():
    text = build_evaluation_input(make_result())

    assert "¿Cuántos días de vacaciones tengo?" in text
    assert "[Fragmento 1 | Sección: Vacaciones]" in text
    assert "Tienes 14 días" in text


def test_to_evaluation_sums_dimensions_and_explains_breakdown():
    judgement = JudgeOutput(relevance=3, faithfulness=4, completeness=2, reason="Falta el saldo.")

    evaluation = to_evaluation(judgement)

    assert evaluation.score == 9
    assert evaluation.reason.startswith("Puntaje 9: relevancia 3/3, fidelidad 4/4, completitud 2/3.")
    assert len(evaluation.reason) >= 50


def test_to_evaluation_clamps_out_of_range_grades():
    judgement = JudgeOutput(relevance=7, faithfulness=-2, completeness=3, reason="x")

    assert to_evaluation(judgement).score == 6


def test_evaluate_answer_returns_valid_evaluation():
    judgement = JudgeOutput(relevance=3, faithfulness=4, completeness=3, reason="Todo respaldado.")

    evaluation = evaluate_answer(make_result(), make_client(judgement))

    assert isinstance(evaluation, Evaluation)
    assert 0 <= evaluation.score <= 10


def test_evaluate_answer_wraps_api_errors():
    with pytest.raises(EvaluationError, match="quota"):
        evaluate_answer(make_result(), make_client(error=OpenAIError("quota")))


def test_evaluation_rejects_short_reason_and_out_of_range_score():
    with pytest.raises(ValueError):
        Evaluation(score=8, reason="corta")
    with pytest.raises(ValueError):
        Evaluation(score=11, reason="x" * 60)
