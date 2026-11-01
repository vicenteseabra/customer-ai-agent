"""SQLite connection management and schema initialization for conversation persistence.

This module owns only the `conversations` and `messages` tables used to persist
chat history. The six business-data tables imported from the source CSVs are
created separately by the importer (see `store_chat.importer`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_CONVERSATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    customer_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
);
"""


def get_connection(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection to `database_path` with foreign keys enforced.

    Creates the parent directory if it does not exist yet. Rows are returned
    as `sqlite3.Row` objects so callers can access columns by name.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the `conversations` and `messages` tables if they do not exist.

    Safe to call multiple times: existing tables and their data are left
    untouched.
    """
    connection.executescript(_CONVERSATION_SCHEMA)
    connection.commit()
