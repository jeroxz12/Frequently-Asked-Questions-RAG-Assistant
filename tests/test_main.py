import pytest

import main
from src.vector_store import IndexNotFoundError


def test_ask_with_empty_question_prints_error_and_returns_1(capsys):
    assert main.main(["ask", "   "]) == 1
    assert "Error: The question cannot be empty." in capsys.readouterr().err


def test_ask_without_index_prints_friendly_error(monkeypatch, capsys):
    def raise_missing_index(question):
        raise IndexNotFoundError("No index found. Run `python main.py build` first.")

    monkeypatch.setattr(main, "answer_question", raise_missing_index)

    assert main.main(["ask", "¿Hola?"]) == 1
    assert "python main.py build" in capsys.readouterr().err


def test_missing_command_exits_with_usage_error():
    with pytest.raises(SystemExit):
        main.main([])
