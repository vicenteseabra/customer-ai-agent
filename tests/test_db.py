"""Tests for SQLite connection setup and conversation schema initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from store_chat.db import get_connection, initialize_database


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row["name"] for row in rows}


def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def test_initialize_database_creates_conversation_tables(
    connection: sqlite3.Connection,
) -> None:
    tables = _table_names(connection)
    assert "conversations" in tables
    assert "messages" in tables

    conversation_columns = _column_names(connection, "conversations")
    assert {"id", "customer_id", "created_at", "updated_at"} <= conversation_columns

    message_columns = _column_names(connection, "messages")
    assert {"id", "conversation_id", "role", "content", "created_at"} <= message_columns

    foreign_keys = connection.execute("PRAGMA foreign_key_list(messages)").fetchall()
    assert any(fk["table"] == "conversations" for fk in foreign_keys)

    (foreign_keys_enabled,) = connection.execute("PRAGMA foreign_keys").fetchone()
    assert foreign_keys_enabled == 1


def test_initialize_database_is_idempotent(database_path: Path) -> None:
    connection = get_connection(database_path)
    initialize_database(connection)

    connection.execute(
        "INSERT INTO conversations (id, customer_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("conv-1", None, "2026-11-01T00:00:00", "2026-11-01T00:00:00"),
    )
    connection.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("msg-1", "conv-1", "user", "Hello", "2026-11-01T00:00:00"),
    )
    connection.commit()

    # Re-running initialization must not drop existing tables or data.
    initialize_database(connection)

    messages = connection.execute("SELECT * FROM messages").fetchall()
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello"

    connection.close()
