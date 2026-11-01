"""Conversation persistence and verified-customer order access.

This module persists chat history (conversations and their user-visible
user/assistant messages) and implements the privacy-critical
customer-verification flow: a conversation only gains access to a customer's
orders after an order number and an email both match the same order, and
every kind of mismatch is reported identically so a caller can never learn
which part of the pair was wrong.

Only user-visible messages are persisted here: hidden tool payloads and
system prompts never reach `append_message` and are never written to the
`messages` table. Conversations and messages are never expired or deleted by
this module.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from store_chat.repositories import OrderSummary, StoreRepository

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Conversation:
    """A persisted chat conversation, optionally associated with a verified customer."""

    id: UUID
    customer_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Message:
    """A single user-visible message persisted within a conversation."""

    id: UUID
    conversation_id: UUID
    role: Role
    content: str
    created_at: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=UUID(row["id"]),
        customer_id=row["customer_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    return Message(
        id=UUID(row["id"]),
        conversation_id=UUID(row["conversation_id"]),
        role=row["role"],
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class ConversationService:
    """Persists conversations/messages and enforces verified-customer order access.

    `list_verified_customer_orders` takes only a conversation id, deriving the
    customer scope entirely from that conversation's own stored
    `customer_id`. There is no parameter through which a caller could request
    another customer's orders, so it is impossible to use this service to
    read orders belonging to anyone other than the conversation's own
    verified customer.
    """

    def __init__(self, connection: sqlite3.Connection, store: StoreRepository) -> None:
        self._connection = connection
        self._store = store

    def create_conversation(self) -> Conversation:
        """Create a new, unverified conversation with an opaque UUID id."""
        conversation_id = uuid.uuid4()
        now = _now()
        self._connection.execute(
            "INSERT INTO conversations (id, customer_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (str(conversation_id), None, now.isoformat(), now.isoformat()),
        )
        self._connection.commit()
        return Conversation(
            id=conversation_id, customer_id=None, created_at=now, updated_at=now
        )

    def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        """Return the conversation with `conversation_id`, or None if it does not exist."""
        row = self._connection.execute(
            "SELECT id, customer_id, created_at, updated_at FROM conversations WHERE id = ?",
            (str(conversation_id),),
        ).fetchone()
        if row is None:
            return None
        return _conversation_from_row(row)

    def append_message(self, conversation_id: UUID, role: Role, content: str) -> Message:
        """Persist one user-visible message and return it.

        Callers must only pass user-visible user/assistant turns: hidden tool
        payloads and system prompts must never be routed through this method.
        """
        message_id = uuid.uuid4()
        now = _now()
        self._connection.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(message_id), str(conversation_id), role, content, now.isoformat()),
        )
        self._connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now.isoformat(), str(conversation_id)),
        )
        self._connection.commit()
        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
        )

    def get_history(self, conversation_id: UUID) -> list[Message]:
        """Return all persisted messages for `conversation_id`, oldest first.

        Ordered by SQLite's implicit rowid (insertion order) rather than the
        stored timestamp, so ordering stays correct even when two messages
        are appended within the same clock tick.
        """
        rows = self._connection.execute(
            "SELECT id, conversation_id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY rowid ASC",
            (str(conversation_id),),
        ).fetchall()
        return [_message_from_row(row) for row in rows]

    def verify_customer(self, conversation_id: UUID, order_number: str, email: str) -> bool:
        """Associate the conversation with the customer who placed `order_number`
        under `email`, only when both match the same order.

        Returns the same `False` for every kind of mismatch (unknown order
        number, unknown email, an order number and email that each exist but
        do not belong to the same order): no return value or side effect here
        ever reveals which part of the pair was wrong. On any failure, the
        conversation's customer association is left untouched.
        """
        identity = self._store.find_matching_order(order_number, email)
        if identity is None:
            return False

        now = _now()
        self._connection.execute(
            "UPDATE conversations SET customer_id = ?, updated_at = ? WHERE id = ?",
            (str(identity.customer_id), now.isoformat(), str(conversation_id)),
        )
        self._connection.commit()
        return True

    def list_verified_customer_orders(self, conversation_id: UUID) -> list[OrderSummary]:
        """Return the orders of this conversation's verified customer, if any.

        Returns an empty list when the conversation does not exist or has not
        been verified. The customer scope is read from the conversation's own
        stored `customer_id`, never from caller input, so this method cannot
        be used to retrieve another customer's orders.
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None or conversation.customer_id is None:
            return []
        return self._store.list_orders_for_customer(conversation.customer_id)
