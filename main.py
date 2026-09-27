import argparse
import sys

from src.build_index import DocumentLoadError, build_index
from src.config import ConfigurationError
from src.embeddings import EmbeddingError
from src.evaluator import EvaluationError, evaluate_answer
from src.query import GenerationError, answer_question
from src.samples import run_samples
from src.vector_store import IndexNotFoundError

KNOWN_ERRORS = (
    ConfigurationError,
    DocumentLoadError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IndexNotFoundError,
    ValueError,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAQ RAG Assistant for Nexo HR.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="Chunk the document, embed it and save the index.")
    ask = commands.add_parser("ask", help="Answer a question and print the JSON result.")
    ask.add_argument("question", help='Question in quotes, e.g. "¿Cuántos días de vacaciones tengo?"')
    ask.add_argument("--evaluate", action="store_true", help="Also print the evaluator's score.")
    commands.add_parser("samples", help="Run the sample questions into outputs/sample_queries.json.")
    return parser.parse_args(argv)


def run_command(args: argparse.Namespace) -> None:
    if args.command == "build":
        build_index()
    elif args.command == "ask":
        result = answer_question(args.question)
        print(result.model_dump_json(indent=2))
        if args.evaluate:
            print(evaluate_answer(result).model_dump_json(indent=2))
    elif args.command == "samples":
        run_samples()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_command(args)
    except KNOWN_ERRORS as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
