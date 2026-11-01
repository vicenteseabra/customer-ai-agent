"""Atomic CSV import of the musical-instrument-store business data.

Reads the six source CSVs (categories, customers, products, orders,
order_items, promotions) from a source directory and loads them into SQLite
business tables inside a single transaction. Any invalid input — a missing
file, a missing required column, or a row referencing a relationship that
does not exist — rolls back the entire import and raises `CsvImportError`
naming the offending table/row.

The real source files are named like
``desafio_tecnico_ai_eng - categories.csv`` (a fixed prefix plus the table
name), so files are located by suffix match (``*<table>.csv``) rather than by
exact name. This also lets tests use plain ``categories.csv``-style fixture
names.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Table import order matters: each table is only inserted after the tables
# it references via foreign key, so referential checks can be done against
# already-known id sets.
_TABLE_ORDER = ("categories", "customers", "products", "orders", "order_items", "promotions")

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "categories": ("category_id", "name", "description"),
    "customers": ("customer_id", "name", "phone", "email", "city"),
    "products": (
        "product_id",
        "price_brl",
        "name",
        "category_id",
        "description",
        "stock_quantity",
        "status",
        "specs",
        "created_at",
    ),
    "orders": (
        "order_id",
        "customer_id",
        "order_date",
        "status",
        "total_brl",
        "payment_method",
        "tracking_code",
        "estimated_delivery",
        "notes",
    ),
    "order_items": ("order_id", "quantity", "product_id"),
    "promotions": ("promotion_id", "product_id", "discount_percent", "description", "is_active"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT,
    email TEXT NOT NULL,
    city TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    price_brl REAL NOT NULL,
    name TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories (category_id),
    description TEXT,
    stock_quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    specs TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    order_date TEXT,
    status TEXT,
    total_brl REAL,
    payment_method TEXT,
    tracking_code TEXT,
    estimated_delivery TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id INTEGER NOT NULL REFERENCES orders (order_id),
    quantity INTEGER NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products (product_id)
);

CREATE TABLE IF NOT EXISTS promotions (
    promotion_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products (product_id),
    discount_percent REAL NOT NULL,
    description TEXT,
    is_active INTEGER NOT NULL
);
"""

# Tables are cleared in reverse dependency order before a fresh import so
# `import_csv_directory` is safe to call more than once against the same
# database.
_CLEAR_ORDER = tuple(reversed(_TABLE_ORDER))


class CsvImportError(Exception):
    """Raised when the source CSVs cannot be imported as-is.

    `errors` lists every problem found (missing files, missing columns,
    invalid relationships) so the caller can see everything wrong at once.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class ImportReport:
    """Summary of a successful import: number of rows imported per table."""

    imported_rows: dict[str, int]


def _find_source_file(source_dir: Path, table: str) -> Path | None:
    matches = sorted(source_dir.glob(f"*{table}.csv"))
    return matches[0] if matches else None


def _read_rows(path: Path, table: str, errors: list[str]) -> list[dict[str, str]] | None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing_columns = [
            column for column in _REQUIRED_COLUMNS[table] if column not in header
        ]
        if missing_columns:
            errors.append(
                f"{table}: missing required column(s) {missing_columns} "
                f"(found columns: {header})"
            )
            return None
        return list(reader)


def _parse_int(value: str, table: str, row_number: int, column: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{table} row {row_number}: invalid integer for '{column}': {value!r}")
        return None


def _parse_float(value: str, table: str, row_number: int, column: str, errors: list[str]) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{table} row {row_number}: invalid number for '{column}': {value!r}")
        return None


def import_csv_directory(connection: sqlite3.Connection, source_dir: Path) -> ImportReport:
    """Atomically import the six business CSVs from `source_dir` into `connection`.

    Creates the business tables if they do not exist, clears any existing
    rows, and inserts rows parsed from the source CSVs. On any problem
    (missing file, missing required column, or a row referencing an id that
    does not exist) the entire operation is rolled back and `CsvImportError`
    is raised listing every problem found; no partial data is left behind.
    """
    errors: list[str] = []
    paths: dict[str, Path] = {}
    for table in _TABLE_ORDER:
        path = _find_source_file(source_dir, table)
        if path is None:
            errors.append(f"{table}: missing source CSV file (expected '*{table}.csv' in {source_dir})")
        else:
            paths[table] = path

    if errors:
        raise CsvImportError(errors)

    rows_by_table: dict[str, list[dict[str, str]]] = {}
    for table in _TABLE_ORDER:
        rows = _read_rows(paths[table], table, errors)
        if rows is not None:
            rows_by_table[table] = rows

    if errors:
        raise CsvImportError(errors)

    connection.execute("BEGIN")
    try:
        connection.executescript(_SCHEMA)
        for table in _CLEAR_ORDER:
            connection.execute(f"DELETE FROM {table}")

        imported_rows: dict[str, int] = {}
        known_ids: dict[str, set[int]] = {}

        # categories
        category_ids: set[int] = set()
        for i, row in enumerate(rows_by_table["categories"], start=1):
            category_id = _parse_int(row["category_id"], "categories", i, "category_id", errors)
            if category_id is None:
                continue
            category_ids.add(category_id)
            connection.execute(
                "INSERT INTO categories (category_id, name, description) VALUES (?, ?, ?)",
                (category_id, row["name"], row["description"]),
            )
        imported_rows["categories"] = len(rows_by_table["categories"])
        known_ids["categories"] = category_ids

        # customers
        customer_ids: set[int] = set()
        for i, row in enumerate(rows_by_table["customers"], start=1):
            customer_id = _parse_int(row["customer_id"], "customers", i, "customer_id", errors)
            if customer_id is None:
                continue
            customer_ids.add(customer_id)
            connection.execute(
                "INSERT INTO customers (customer_id, name, phone, email, city) "
                "VALUES (?, ?, ?, ?, ?)",
                (customer_id, row["name"], row["phone"], row["email"], row["city"]),
            )
        imported_rows["customers"] = len(rows_by_table["customers"])
        known_ids["customers"] = customer_ids

        # products (reference categories)
        product_ids: set[int] = set()
        for i, row in enumerate(rows_by_table["products"], start=1):
            product_id = _parse_int(row["product_id"], "products", i, "product_id", errors)
            category_id = _parse_int(row["category_id"], "products", i, "category_id", errors)
            price_brl = _parse_float(row["price_brl"], "products", i, "price_brl", errors)
            stock_quantity = _parse_int(
                row["stock_quantity"], "products", i, "stock_quantity", errors
            )
            if None in (product_id, category_id, price_brl, stock_quantity):
                continue
            if category_id not in known_ids["categories"]:
                errors.append(
                    f"products row {i}: category_id {category_id} does not exist in categories"
                )
                continue
            product_ids.add(product_id)
            connection.execute(
                "INSERT INTO products "
                "(product_id, price_brl, name, category_id, description, "
                "stock_quantity, status, specs, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    product_id,
                    price_brl,
                    row["name"],
                    category_id,
                    row["description"],
                    stock_quantity,
                    row["status"],
                    row["specs"],
                    row["created_at"],
                ),
            )
        imported_rows["products"] = len(rows_by_table["products"])
        known_ids["products"] = product_ids

        # orders (reference customers)
        order_ids: set[int] = set()
        for i, row in enumerate(rows_by_table["orders"], start=1):
            order_id = _parse_int(row["order_id"], "orders", i, "order_id", errors)
            customer_id = _parse_int(row["customer_id"], "orders", i, "customer_id", errors)
            total_brl = _parse_float(row["total_brl"], "orders", i, "total_brl", errors)
            if None in (order_id, customer_id, total_brl):
                continue
            if customer_id not in known_ids["customers"]:
                errors.append(
                    f"orders row {i}: customer_id {customer_id} does not exist in customers"
                )
                continue
            order_ids.add(order_id)
            connection.execute(
                "INSERT INTO orders "
                "(order_id, customer_id, order_date, status, total_brl, "
                "payment_method, tracking_code, estimated_delivery, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order_id,
                    customer_id,
                    row["order_date"],
                    row["status"],
                    total_brl,
                    row["payment_method"],
                    row["tracking_code"],
                    row["estimated_delivery"],
                    row["notes"],
                ),
            )
        imported_rows["orders"] = len(rows_by_table["orders"])
        known_ids["orders"] = order_ids

        # order_items (reference orders and products)
        for i, row in enumerate(rows_by_table["order_items"], start=1):
            order_id = _parse_int(row["order_id"], "order_items", i, "order_id", errors)
            quantity = _parse_int(row["quantity"], "order_items", i, "quantity", errors)
            product_id = _parse_int(row["product_id"], "order_items", i, "product_id", errors)
            if None in (order_id, quantity, product_id):
                continue
            if order_id not in known_ids["orders"]:
                errors.append(f"order_items row {i}: order_id {order_id} does not exist in orders")
                continue
            if product_id not in known_ids["products"]:
                errors.append(
                    f"order_items row {i}: product_id {product_id} does not exist in products"
                )
                continue
            connection.execute(
                "INSERT INTO order_items (order_id, quantity, product_id) VALUES (?, ?, ?)",
                (order_id, quantity, product_id),
            )
        imported_rows["order_items"] = len(rows_by_table["order_items"])

        # promotions (reference products)
        for i, row in enumerate(rows_by_table["promotions"], start=1):
            promotion_id = _parse_int(row["promotion_id"], "promotions", i, "promotion_id", errors)
            product_id = _parse_int(row["product_id"], "promotions", i, "product_id", errors)
            discount_percent = _parse_float(
                row["discount_percent"], "promotions", i, "discount_percent", errors
            )
            is_active = _parse_int(row["is_active"], "promotions", i, "is_active", errors)
            if None in (promotion_id, product_id, discount_percent, is_active):
                continue
            if product_id not in known_ids["products"]:
                errors.append(
                    f"promotions row {i}: product_id {product_id} does not exist in products"
                )
                continue
            connection.execute(
                "INSERT INTO promotions "
                "(promotion_id, product_id, discount_percent, description, is_active) "
                "VALUES (?, ?, ?, ?, ?)",
                (promotion_id, product_id, discount_percent, row["description"], is_active),
            )
        imported_rows["promotions"] = len(rows_by_table["promotions"])

        if errors:
            raise CsvImportError(errors)

        connection.commit()
        return ImportReport(imported_rows=imported_rows)
    except Exception:
        connection.rollback()
        if errors:
            raise CsvImportError(errors) from None
        raise
