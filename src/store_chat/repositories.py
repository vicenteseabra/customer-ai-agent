"""Read-only repository over the imported store business data.

All queries are parameterized SQL and all methods are read-only: this module
never mutates the database. `find_matching_order` and
`list_orders_for_customer` are the privacy-sensitive entry points customer
verification and the chat agent build on, so they are written to never leak
data across customers (see their docstrings for the exact guarantee).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ProductSummary:
    """A product as sourced from products.csv (joined with its category name)."""

    product_id: int
    name: str
    description: str
    price_brl: float
    category: str
    stock_quantity: int
    status: str


@dataclass(frozen=True)
class CategorySummary:
    """A product category as sourced from categories.csv."""

    category_id: int
    name: str
    description: str


@dataclass(frozen=True)
class PromotionSummary:
    """A promotion as sourced from promotions.csv (joined with its product's name)."""

    promotion_id: int
    product_id: int
    product_name: str
    discount_percent: float
    description: str
    is_active: bool


@dataclass(frozen=True)
class VerifiedOrderIdentity:
    """The customer identity confirmed by a matching order number and email."""

    order_id: int
    customer_id: int
    customer_name: str
    email: str


@dataclass(frozen=True)
class OrderSummary:
    """An order as sourced from orders.csv, scoped to a single customer."""

    order_id: int
    customer_id: int
    order_date: str
    status: str
    total_brl: float
    payment_method: str
    tracking_code: str
    estimated_delivery: str
    notes: str


class StoreRepository:
    """Read-only access to the imported store business data.

    `order_number` (as used by `find_matching_order`) is the order's
    `order_id` from orders.csv given as a string: the source data has no
    separate order-number field, so `order_id` serves as the order number
    customers are expected to quote.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def search_products(
        self, query: str, category: str | None = None, limit: int = 5
    ) -> list[ProductSummary]:
        sql = (
            "SELECT p.product_id, p.name, p.description, p.price_brl, "
            "c.name AS category, p.stock_quantity, p.status "
            "FROM products p "
            "JOIN categories c ON c.category_id = p.category_id "
            "WHERE (p.name LIKE ? OR p.description LIKE ?)"
        )
        like_pattern = f"%{query}%"
        params: list[object] = [like_pattern, like_pattern]
        if category is not None:
            sql += " AND c.name = ? COLLATE NOCASE"
            params.append(category)
        sql += " ORDER BY p.name LIMIT ?"
        params.append(limit)

        rows = self._connection.execute(sql, params).fetchall()
        return [
            ProductSummary(
                product_id=row["product_id"],
                name=row["name"],
                description=row["description"],
                price_brl=row["price_brl"],
                category=row["category"],
                stock_quantity=row["stock_quantity"],
                status=row["status"],
            )
            for row in rows
        ]

    def list_categories(self) -> list[CategorySummary]:
        rows = self._connection.execute(
            "SELECT category_id, name, description FROM categories ORDER BY name"
        ).fetchall()
        return [
            CategorySummary(
                category_id=row["category_id"],
                name=row["name"],
                description=row["description"],
            )
            for row in rows
        ]

    def list_promotions(self) -> list[PromotionSummary]:
        rows = self._connection.execute(
            "SELECT pr.promotion_id, pr.product_id, p.name AS product_name, "
            "pr.discount_percent, pr.description, pr.is_active "
            "FROM promotions pr "
            "JOIN products p ON p.product_id = pr.product_id "
            "ORDER BY pr.promotion_id"
        ).fetchall()
        return [
            PromotionSummary(
                promotion_id=row["promotion_id"],
                product_id=row["product_id"],
                product_name=row["product_name"],
                discount_percent=row["discount_percent"],
                description=row["description"],
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    def find_matching_order(self, order_number: str, email: str) -> VerifiedOrderIdentity | None:
        """Return the verified identity for `order_number` only if `email`
        belongs to the customer who placed that specific order.

        Both values must match the SAME order: an order number that exists
        under a different customer's email, or an email that exists but not
        on this order, returns None. Order numbers are matched as integers
        (see class docstring); a non-integer `order_number` never matches.
        """
        try:
            order_id = int(order_number)
        except (TypeError, ValueError):
            return None

        row = self._connection.execute(
            "SELECT o.order_id, o.customer_id, c.name AS customer_name, c.email "
            "FROM orders o "
            "JOIN customers c ON c.customer_id = o.customer_id "
            "WHERE o.order_id = ? AND c.email = ? COLLATE NOCASE",
            (order_id, email),
        ).fetchone()
        if row is None:
            return None
        return VerifiedOrderIdentity(
            order_id=row["order_id"],
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            email=row["email"],
        )

    def list_orders_for_customer(self, customer_id: str) -> list[OrderSummary]:
        """Return only the orders belonging to `customer_id`.

        Never returns another customer's orders, regardless of how many
        orders exist overall: the customer_id filter is applied in SQL via a
        parameterized query, not in Python, so there is no path that returns
        an unfiltered or partially filtered result set.
        """
        try:
            parsed_customer_id = int(customer_id)
        except (TypeError, ValueError):
            return []

        rows = self._connection.execute(
            "SELECT order_id, customer_id, order_date, status, total_brl, "
            "payment_method, tracking_code, estimated_delivery, notes "
            "FROM orders WHERE customer_id = ? ORDER BY order_date",
            (parsed_customer_id,),
        ).fetchall()
        return [
            OrderSummary(
                order_id=row["order_id"],
                customer_id=row["customer_id"],
                order_date=row["order_date"],
                status=row["status"],
                total_brl=row["total_brl"],
                payment_method=row["payment_method"],
                tracking_code=row["tracking_code"],
                estimated_delivery=row["estimated_delivery"],
                notes=row["notes"] or "",
            )
            for row in rows
        ]
