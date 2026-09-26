import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DOCUMENT_PATH = PROJECT_ROOT / "data" / "faq_document.txt"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

MIN_CHUNK_TOKENS = 50
MAX_CHUNK_TOKENS = 500

TOP_K = 5
MIN_RESULTS = 2


class ConfigurationError(Exception):
    """Raised when required environment configuration is missing."""


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. Run `export OPENAI_API_KEY=...` "
            "or copy .env.example to .env and add your key."
        )
    return api_key


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
