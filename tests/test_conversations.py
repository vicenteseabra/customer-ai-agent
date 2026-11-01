"""Tests for ConversationService: persistence and verified-customer order access."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from store_chat.conversations import Conversation, ConversationService, Message
from store_chat.db import get_connection, initialize_database
from store_chat.repositories import StoreRepository


def test_create_conversation_returns_unverified_conversation(
    connection: sqlite3.Connection,
) -> None:
    service = ConversationService(connection, StoreRepository(connection))

    conversation = service.create_conversation()

    assert isinstance(conversation, Conversation)
    assert conversation.customer_id is None


def test_get_conversation_returns_none_for_unknown_id(
    connection: sqlite3.Connection,
) -> None:
    service = ConversationService(connection, StoreRepository(connection))

    assert service.get_conversation(uuid.uuid4()) is None


def test_append_message_and_get_history_preserves_order(
    connection: sqlite3.Connection,
) -> None:
    service = ConversationService(connection, StoreRepository(connection))
    conversation = service.create_conversation()

    service.append_message(conversation.id, "user", "Do you have Stratocasters?")
    service.append_message(conversation.id, "assistant", "Yes, we have one in stock.")
    service.append_message(conversation.id, "user", "Great, how much?")

    history = service.get_history(conversation.id)

    assert [m.content for m in history] == [
        "Do you have Stratocasters?",
        "Yes, we have one in stock.",
        "Great, how much?",
    ]
    assert [m.role for m in history] == ["user", "assistant", "user"]
    assert all(isinstance(m, Message) for m in history)
    assert all(m.conversation_id == conversation.id for m in history)


def test_conversation_history_survives_database_reopen(database_path: Path) -> None:
    connection = get_connection(database_path)
    initialize_database(connection)
    service = ConversationService(connection, StoreRepository(connection))
    conversation = service.create_conversation()
    service.append_message(conversation.id, "user", "Hello")
    service.append_message(conversation.id, "assistant", "Hi, how can I help?")
    connection.close()

    reopened_connection = get_connection(database_path)
    initialize_database(reopened_connection)
    try:
        reopened_service = ConversationService(
            reopened_connection, StoreRepository(reopened_connection)
        )

        history = reopened_service.get_history(conversation.id)
        assert [m.content for m in history] == ["Hello", "Hi, how can I help?"]
        assert [m.role for m in history] == ["user", "assistant"]

        restored_conversation = reopened_service.get_conversation(conversation.id)
        assert restored_conversation is not None
        assert restored_conversation.id == conversation.id
    finally:
        reopened_connection.close()


def test_failed_verification_does_not_associate_customer(
    imported_connection: sqlite3.Connection,
) -> None:
    store = StoreRepository(imported_connection)
    service = ConversationService(imported_connection, store)
    conversation = service.create_conversation()

    # Wrong email for a real order, unknown order number, and a real order
    # paired with a real but mismatched email: all must fail identically.
    assert service.verify_customer(conversation.id, "1", "wrong@example.com") is False
    assert service.verify_customer(conversation.id, "999", "alice@example.com") is False
    assert service.verify_customer(conversation.id, "2", "alice@example.com") is False

    unchanged = service.get_conversation(conversation.id)
    assert unchanged is not None
    assert unchanged.customer_id is None
    assert service.list_verified_customer_orders(conversation.id) == []


def test_matching_order_and_email_associates_customer(
    imported_connection: sqlite3.Connection,
) -> None:
    store = StoreRepository(imported_connection)
    service = ConversationService(imported_connection, store)
    conversation = service.create_conversation()

    assert service.verify_customer(conversation.id, "1", "alice@example.com") is True

    verified = service.get_conversation(conversation.id)
    assert verified is not None
    assert verified.customer_id == "1"


def test_verified_customer_can_list_all_own_orders(
    imported_connection: sqlite3.Connection,
) -> None:
    store = StoreRepository(imported_connection)
    service = ConversationService(imported_connection, store)
    conversation = service.create_conversation()
    service.verify_customer(conversation.id, "1", "alice@example.com")

    orders = service.list_verified_customer_orders(conversation.id)

    assert len(orders) == 1
    assert orders[0].order_id == 1
    assert orders[0].customer_id == 1


def test_verified_customer_cannot_list_another_customers_orders(
    imported_connection: sqlite3.Connection,
) -> None:
    store = StoreRepository(imported_connection)
    service = ConversationService(imported_connection, store)
    alice_conversation = service.create_conversation()
    service.verify_customer(alice_conversation.id, "1", "alice@example.com")

    orders = service.list_verified_customer_orders(alice_conversation.id)

    assert {o.order_id for o in orders} == {1}
    assert 2 not in {o.order_id for o in orders}

    # An unverified conversation gets nothing, even though Bob's order exists
    # in the same database and there is no customer-id parameter to request it.
    unverified_conversation = service.create_conversation()
    assert service.list_verified_customer_orders(unverified_conversation.id) == []
