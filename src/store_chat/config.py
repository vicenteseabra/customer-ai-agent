"""Typed application settings for Store Chat, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEFAULT_DATABASE_PATH = Path("data/store_chat.sqlite3")
DEFAULT_SOURCE_DATA_DIR = Path("data/source")
DEFAULT_VECTOR_STORE_PATH = Path("data/vector_store")
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "llama3.1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Policy PDF chunking: fixed-size character windows with overlap, so no
# sentence meaning is lost at a chunk boundary.
DEFAULT_POLICY_CHUNK_SIZE = 500
DEFAULT_POLICY_CHUNK_OVERLAP = 50

# Chroma reports cosine *distance* (0 = identical, up to 2 = opposite; two
# unrelated texts land close to 1, i.e. an orthogonal embedding). A retrieved
# chunk is only considered relevant, and returned, when its distance is at or
# below this threshold; otherwise `retrieve` returns an empty list rather
# than a fabricated answer.
DEFAULT_POLICY_RELEVANCE_DISTANCE_THRESHOLD = 0.95


@dataclass(frozen=True)
class Settings:
    """Local runtime configuration for Store Chat."""

    database_path: Path
    source_data_dir: Path
    vector_store_path: Path
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    policy_chunk_size: int
    policy_chunk_overlap: int
    policy_relevance_distance_threshold: float


def load_settings() -> Settings:
    """Build Settings from environment variables, falling back to local defaults.

    Reads STORE_CHAT_DB_PATH, STORE_CHAT_SOURCE_DIR, STORE_CHAT_VECTOR_PATH,
    OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBEDDING_MODEL,
    STORE_CHAT_POLICY_CHUNK_SIZE, STORE_CHAT_POLICY_CHUNK_OVERLAP, and
    STORE_CHAT_POLICY_RELEVANCE_DISTANCE_THRESHOLD.

    Loads a `.env` file from the current working directory (or a parent of
    it) first, if one exists, without overriding variables already set in
    the real environment.
    """
    load_dotenv(find_dotenv(usecwd=True))
    return Settings(
        database_path=Path(
            os.environ.get("STORE_CHAT_DB_PATH", str(DEFAULT_DATABASE_PATH))
        ),
        source_data_dir=Path(
            os.environ.get("STORE_CHAT_SOURCE_DIR", str(DEFAULT_SOURCE_DATA_DIR))
        ),
        vector_store_path=Path(
            os.environ.get("STORE_CHAT_VECTOR_PATH", str(DEFAULT_VECTOR_STORE_PATH))
        ),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        chat_model=os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        embedding_model=os.environ.get(
            "OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
        policy_chunk_size=int(
            os.environ.get(
                "STORE_CHAT_POLICY_CHUNK_SIZE", str(DEFAULT_POLICY_CHUNK_SIZE)
            )
        ),
        policy_chunk_overlap=int(
            os.environ.get(
                "STORE_CHAT_POLICY_CHUNK_OVERLAP", str(DEFAULT_POLICY_CHUNK_OVERLAP)
            )
        ),
        policy_relevance_distance_threshold=float(
            os.environ.get(
                "STORE_CHAT_POLICY_RELEVANCE_DISTANCE_THRESHOLD",
                str(DEFAULT_POLICY_RELEVANCE_DISTANCE_THRESHOLD),
            )
        ),
    )
