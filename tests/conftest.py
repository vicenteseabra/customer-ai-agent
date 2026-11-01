"""Shared pytest fixtures for Store Chat tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import pytest

from store_chat.db import get_connection, initialize_database
from store_chat.importer import import_csv_directory

# Minimal, hand-written fixture CSVs covering the six business tables and
# their relationships, used by both importer and repository tests. Two
# customers each have exactly one order, which lets isolation tests assert
# that one customer's data never leaks into another's results.
FIXTURE_CSV_CONTENTS: dict[str, str] = {
    "categories.csv": (
        "category_id,name,description\n"
        "1,Guitars,Electric and acoustic guitars\n"
        "2,Basses,Electric and acoustic basses\n"
    ),
    "customers.csv": (
        "customer_id,name,phone,email,city\n"
        "1,Alice Smith,555-0001,alice@example.com,Springfield\n"
        "2,Bob Jones,555-0002,bob@example.com,Shelbyville\n"
    ),
    "products.csv": (
        "product_id,price_brl,name,category_id,description,stock_quantity,status,specs,created_at\n"
        '101,999.9,Stratocaster,1,"Classic electric guitar",5,active,"{""color"":""red""}",2021-01-01\n'
        '102,499.9,Precision Bass,2,"Classic electric bass",0,discontinued,"{}",2020-01-01\n'
    ),
    "orders.csv": (
        "order_id,customer_id,order_date,status,total_brl,payment_method,tracking_code,estimated_delivery,notes\n"
        "1,1,2026-01-01,delivered,999.9,pix,BR111,2026-01-10,\n"
        "2,2,2026-02-01,shipped,499.9,credit_card,BR222,2026-02-10,Gift wrap\n"
    ),
    "order_items.csv": (
        "order_id,quantity,product_id\n"
        "1,1,101\n"
        "2,1,102\n"
    ),
    "promotions.csv": (
        "promotion_id,product_id,discount_percent,description,is_active\n"
        "1,101,10,Spring sale,1\n"
    ),
}


def write_fixture_csvs(directory: Path, overrides: dict[str, str | None] | None = None) -> Path:
    """Write the standard fixture CSVs into `directory`, applying `overrides`.

    `overrides` maps a filename to replacement content; mapping a filename to
    `None` omits that file entirely (used to simulate a missing source file).
    """
    overrides = overrides or {}
    for filename, content in FIXTURE_CSV_CONTENTS.items():
        if filename in overrides:
            replacement = overrides[filename]
            if replacement is None:
                continue
            (directory / filename).write_text(replacement, encoding="utf-8")
        else:
            (directory / filename).write_text(content, encoding="utf-8")
    return directory


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


@pytest.fixture
def fixture_source_dir(tmp_path: Path) -> Path:
    """Provide a directory containing the standard valid fixture CSVs."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    return write_fixture_csvs(source_dir)


@pytest.fixture
def imported_connection(
    connection: sqlite3.Connection, fixture_source_dir: Path
) -> sqlite3.Connection:
    """Provide a connection with the standard fixture CSVs already imported."""
    import_csv_directory(connection, fixture_source_dir)
    return connection
