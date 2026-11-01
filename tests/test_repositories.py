"""Tests for the read-only StoreRepository built on the imported business data."""

from __future__ import annotations

import sqlite3

from store_chat.repositories import (
    CategorySummary,
    OrderSummary,
    ProductSummary,
    PromotionSummary,
    StoreRepository,
    VerifiedOrderIdentity,
)


def test_search_products_finds_by_name_and_category(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)

    results = repository.search_products("Strat")
    assert len(results) == 1
    product = results[0]
    assert isinstance(product, ProductSummary)
    assert product.product_id == 101
    assert product.name == "Stratocaster"
    assert product.category == "Guitars"

    scoped_results = repository.search_products("e", category="Basses")
    assert all(p.category == "Basses" for p in scoped_results)
    assert any(p.product_id == 102 for p in scoped_results)


def test_product_search_does_not_claim_missing_stock(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)

    results = repository.search_products("Precision Bass")
    assert len(results) == 1
    product = results[0]
    # stock_quantity and status are real CSV-sourced fields, not synthesized
    # availability claims: product 102 is discontinued with zero stock, and
    # that must be reported exactly as found in the source data.
    assert product.stock_quantity == 0
    assert product.status == "discontinued"

    active_product = next(p for p in repository.search_products("Stratocaster"))
    assert active_product.stock_quantity == 5
    assert active_product.status == "active"


def test_list_categories_returns_all_categories(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)

    categories = repository.list_categories()
    assert {c.name for c in categories} == {"Guitars", "Basses"}
    assert all(isinstance(c, CategorySummary) for c in categories)


def test_list_promotions_returns_active_and_inactive(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)

    promotions = repository.list_promotions()
    assert len(promotions) == 1
    promotion = promotions[0]
    assert isinstance(promotion, PromotionSummary)
    assert promotion.product_id == 101
    assert promotion.product_name == "Stratocaster"
    assert promotion.is_active is True


def test_find_matching_order_returns_none_for_unknown_order(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)
    assert repository.find_matching_order("999", "alice@example.com") is None


def test_find_matching_order_returns_none_for_unknown_email(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)
    assert repository.find_matching_order("1", "nobody@example.com") is None


def test_find_matching_order_returns_identity_on_correct_pairing(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)
    identity = repository.find_matching_order("1", "alice@example.com")
    assert isinstance(identity, VerifiedOrderIdentity)
    assert identity.customer_id == 1
    assert identity.order_id == 1
    assert identity.customer_name == "Alice Smith"


def test_list_orders_for_customer_returns_only_that_customers_orders(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)
    orders = repository.list_orders_for_customer("1")
    assert len(orders) == 1
    assert isinstance(orders[0], OrderSummary)
    assert orders[0].order_id == 1
    assert orders[0].customer_id == 1


def test_list_orders_for_customer_never_returns_another_customers_orders(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)

    alice_orders = repository.list_orders_for_customer("1")
    bob_orders = repository.list_orders_for_customer("2")

    assert {o.order_id for o in alice_orders} == {1}
    assert {o.order_id for o in bob_orders} == {2}

    # Cross-check: no order belonging to customer 2 ever appears in customer
    # 1's result set, and vice versa.
    assert all(o.customer_id == 1 for o in alice_orders)
    assert all(o.customer_id == 2 for o in bob_orders)
    assert 2 not in {o.order_id for o in alice_orders}
    assert 1 not in {o.order_id for o in bob_orders}


def test_list_orders_for_customer_returns_empty_for_unknown_customer(
    imported_connection: sqlite3.Connection,
) -> None:
    repository = StoreRepository(imported_connection)
    assert repository.list_orders_for_customer("999") == []
