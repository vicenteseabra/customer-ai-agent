"""Typed application settings for Store Chat, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATABASE_PATH = Path("data/store_chat.sqlite3")
DEFAULT_SOURCE_DATA_DIR = Path("data/source")
DEFAULT_VECTOR_STORE_PATH = Path("data/vector_store")
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "llama3.1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class Settings:
    """Local runtime configuration for Store Chat."""

    database_path: Path
    source_data_dir: Path
    vector_store_path: Path
    ollama_base_url: str
    chat_model: str
    embedding_model: str


def load_settings() -> Settings:
    """Build Settings from environment variables, falling back to local defaults.

    Reads STORE_CHAT_DB_PATH, STORE_CHAT_SOURCE_DIR, STORE_CHAT_VECTOR_PATH,
    OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, and OLLAMA_EMBEDDING_MODEL.
    """
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
    )
