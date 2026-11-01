"""Tests for atomic CSV import of the store business data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from store_chat.importer import CsvImportError, ImportReport, import_csv_directory
from tests.conftest import write_fixture_csvs


def _all_table_row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    tables = ("categories", "customers", "products", "orders", "order_items", "promotions")
    counts = {}
    for table in tables:
        exists = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        counts[table] = (
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] if exists else 0
        )
    return counts


def test_import_csv_directory_imports_all_six_tables(
    connection: sqlite3.Connection, fixture_source_dir: Path
) -> None:
    report = import_csv_directory(connection, fixture_source_dir)

    assert isinstance(report, ImportReport)
    assert report.imported_rows == {
        "categories": 2,
        "customers": 2,
        "products": 2,
        "orders": 2,
        "order_items": 2,
        "promotions": 1,
    }

    counts = _all_table_row_counts(connection)
    assert counts == report.imported_rows

    # Relationships are preserved: order 1 belongs to customer 1 and contains
    # product 101.
    order_row = connection.execute(
        "SELECT customer_id FROM orders WHERE order_id = 1"
    ).fetchone()
    assert order_row["customer_id"] == 1

    item_row = connection.execute(
        "SELECT product_id FROM order_items WHERE order_id = 1"
    ).fetchone()
    assert item_row["product_id"] == 101

    product_row = connection.execute(
        "SELECT category_id FROM products WHERE product_id = 101"
    ).fetchone()
    assert product_row["category_id"] == 1


def test_import_csv_directory_rejects_missing_file(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_fixture_csvs(source_dir, overrides={"promotions.csv": None})

    with pytest.raises(CsvImportError) as excinfo:
        import_csv_directory(connection, source_dir)

    assert any("promotions" in error for error in excinfo.value.errors)
    assert _all_table_row_counts(connection) == {
        "categories": 0,
        "customers": 0,
        "products": 0,
        "orders": 0,
        "order_items": 0,
        "promotions": 0,
    }


def test_import_csv_directory_rejects_missing_required_column(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_fixture_csvs(
        source_dir,
        overrides={
            "customers.csv": (
                "customer_id,name,phone,city\n"
                "1,Alice Smith,555-0001,Springfield\n"
            )
        },
    )

    with pytest.raises(CsvImportError) as excinfo:
        import_csv_directory(connection, source_dir)

    assert any(
        "customers" in error and "email" in error for error in excinfo.value.errors
    )
    assert _all_table_row_counts(connection) == {
        "categories": 0,
        "customers": 0,
        "products": 0,
        "orders": 0,
        "order_items": 0,
        "promotions": 0,
    }


def test_import_csv_directory_rolls_back_invalid_import(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_fixture_csvs(
        source_dir,
        overrides={
            # order_items references product_id 999, which does not exist.
            "order_items.csv": ("order_id,quantity,product_id\n" "1,1,999\n" "2,1,102\n")
        },
    )

    with pytest.raises(CsvImportError) as excinfo:
        import_csv_directory(connection, source_dir)

    assert any(
        "order_items" in error and "999" in error for error in excinfo.value.errors
    )
    # No partial rows left behind in ANY table, including the valid ones
    # (categories, customers, products, orders) that were processed before
    # the invalid order_items row was encountered.
    assert _all_table_row_counts(connection) == {
        "categories": 0,
        "customers": 0,
        "products": 0,
        "orders": 0,
        "order_items": 0,
        "promotions": 0,
    }


def test_import_csv_directory_rejects_duplicate_primary_key(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_fixture_csvs(
        source_dir,
        overrides={
            # category_id 1 appears twice.
            "categories.csv": (
                "category_id,name,description\n"
                "1,Guitars,Electric and acoustic guitars\n"
                "2,Basses,Electric and acoustic basses\n"
                "1,Guitars Duplicate,Should not import\n"
            )
        },
    )

    with pytest.raises(CsvImportError) as excinfo:
        import_csv_directory(connection, source_dir)

    # A clean CsvImportError naming the table/row, not a raw sqlite3
    # IntegrityError leaking out of the importer.
    assert any(
        "categories" in error and "duplicate" in error and "1" in error
        for error in excinfo.value.errors
    )
    assert _all_table_row_counts(connection) == {
        "categories": 0,
        "customers": 0,
        "products": 0,
        "orders": 0,
        "order_items": 0,
        "promotions": 0,
    }


def test_import_csv_directory_is_idempotent(
    connection: sqlite3.Connection, fixture_source_dir: Path
) -> None:
    import_csv_directory(connection, fixture_source_dir)
    report = import_csv_directory(connection, fixture_source_dir)

    assert report.imported_rows["customers"] == 2
    counts = _all_table_row_counts(connection)
    assert counts["customers"] == 2


def test_find_matching_order_requires_order_and_email_on_same_order(
    imported_connection: sqlite3.Connection,
) -> None:
    from store_chat.repositories import StoreRepository

    repository = StoreRepository(imported_connection)

    # Order 1 belongs to Alice; order 2 belongs to Bob. Mixing a real order
    # number with a real-but-wrong email must not match.
    assert repository.find_matching_order("1", "bob@example.com") is None
    assert repository.find_matching_order("2", "alice@example.com") is None

    # Only the correct order+email pairing matches.
    match = repository.find_matching_order("1", "alice@example.com")
    assert match is not None
    assert match.order_id == 1
    assert match.customer_id == 1
