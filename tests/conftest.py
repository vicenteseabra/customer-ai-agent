"""Shared pytest fixtures for Store Chat tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import pytest

from store_chat.db import get_connection, initialize_database


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Provide a SQLite database file path inside a pytest temporary directory."""
    return tmp_path / "store_chat_test.sqlite3"


@pytest.fixture
def connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Provide an initialized SQLite connection backed by a temporary database file."""
    conn = get_connection(database_path)
    initialize_database(conn)
    try:
        yield conn
    finally:
        conn.close()
