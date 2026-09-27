import json

from src.samples import (
    SAMPLE_QUESTIONS,
    count_relevant,
    is_relevant,
    save_evaluations,
    save_results,
)
from src.schemas import Evaluation, QueryResult, RetrievedChunk


def make_chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=0, section="s", similarity=0.5, text=text)


def make_result() -> QueryResult:
    return QueryResult(
        user_question="¿Cuántos días de vacaciones tengo?",
        system_answer="14 días.",
        chunks_related=[make_chunk("Vacaciones y licencias\n..."), make_chunk("Nómina\n...")],
    )


def test_is_relevant_matches_keywords_case_insensitively():
    assert is_relevant(make_chunk("Vacaciones y licencias"), ["vacaciones"])
    assert not is_relevant(make_chunk("Nómina y recibos"), ["vacaciones"])


def test_count_relevant_counts_matching_chunks():
    assert count_relevant(make_result(), ["vacaciones"]) == 1


def test_save_results_writes_a_list_of_three_key_objects(tmp_path):
    path = tmp_path / "outputs" / "sample_queries.json"

    save_results([make_result()], path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved[0]) == {"user_question", "system_answer", "chunks_related"}
    assert saved[0]["user_question"] == "¿Cuántos días de vacaciones tengo?"


def test_save_evaluations_keeps_them_apart_from_query_results(tmp_path):
    path = tmp_path / "sample_evaluations.json"
    evaluation = Evaluation(score=9, reason="Puntaje 9: relevancia 3/3, fidelidad 4/4, completitud 2/3.")

    save_evaluations([make_result()], [evaluation], path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == [
        {"user_question": "¿Cuántos días de vacaciones tengo?", "evaluation": evaluation.model_dump()}
    ]


def test_there_are_at_least_three_in_scope_samples():
    assert sum(1 for _, keywords in SAMPLE_QUESTIONS if keywords) >= 3
