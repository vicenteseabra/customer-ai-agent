"""Contract tests for the FastAPI chat endpoint and static browser UI.

These tests build the app through `create_app`'s dependency-injection seam
(real `ConversationService`/`StoreRepository` backed by a temporary SQLite
file, plus scripted/failing `ChatClient` test doubles) so no test ever
talks to a real Gemini API or Ollama server.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from store_chat.agent import (
    AgentTool,
    ChatAgent,
    ChatMessage,
    ModelResponse,
    ModelUnavailableError,
)
from store_chat.api import create_app
from store_chat.config import Settings
from store_chat.conversations import ConversationService
from store_chat.db import get_connection, initialize_database
from store_chat.policy_rag import PolicyChunk
from store_chat.repositories import StoreRepository


class FakePolicyIndex:
    """Test double for PolicyIndex: always returns a fixed, pre-scripted chunk list."""

    def __init__(self, chunks: list[PolicyChunk] | None = None) -> None:
        self._chunks = chunks or []

    def retrieve(self, query: str, limit: int = 4) -> list[PolicyChunk]:
        return self._chunks


class ScriptedChatClient:
    """Test double for ChatClient: replays a scripted sequence of responses."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def chat(self, messages: list[ChatMessage], tools: list[AgentTool]) -> ModelResponse:
        if not self._responses:
            raise AssertionError("ScriptedChatClient ran out of scripted responses")
        return self._responses.pop(0)


class FailingChatClient:
    """Test double for ChatClient: always raises, simulating an unreachable model."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def chat(self, messages: list[ChatMessage], tools: list[AgentTool]) -> ModelResponse:
        raise self._error


def _test_settings(database_path: Path) -> Settings:
    return Settings(
        database_path=database_path,
        source_data_dir=Path("unused"),
        vector_store_path=Path("unused"),
        ollama_base_url="http://localhost:11434",
        chat_model="unused",
        gemini_api_key="unused",
        embedding_model="unused",
        policy_chunk_size=500,
        policy_chunk_overlap=50,
        policy_relevance_distance_threshold=0.95,
    )


def _build_app(
    database_path: Path, model_client, policies=None
) -> tuple[object, ConversationService, sqlite3.Connection]:
    """Build an app the same way `build_app` would, but with injected test doubles.

    Opens a fresh connection to `database_path`, initializing the schema if
    needed -- calling this twice against the same path simulates reconnecting
    to the same on-disk database across an app restart.
    """
    connection = get_connection(database_path)
    initialize_database(connection)
    store = StoreRepository(connection)
    conversations = ConversationService(connection, store)
    agent = ChatAgent(model_client, policies or FakePolicyIndex(), store, conversations)
    settings = _test_settings(database_path)
    app = create_app(settings, agent, conversations)
    return app, conversations, connection


def test_post_chat_creates_and_persists_conversation(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.sqlite3"
    model = ScriptedChatClient([ModelResponse(content="Hello! How can I help?", tool_calls=[])])
    app, conversations, connection = _build_app(db_path, model)
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "Hi there"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Hello! How can I help?"
    conversation_id = body["conversation_id"]

    history = conversations.get_history(UUID(conversation_id))
    assert [m.content for m in history] == ["Hi there", "Hello! How can I help?"]
    connection.close()


def test_post_chat_resumes_conversation_after_app_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.sqlite3"

    app1, _conversations1, connection1 = _build_app(
        db_path, ScriptedChatClient([ModelResponse(content="First reply", tool_calls=[])])
    )
    client1 = TestClient(app1)
    response1 = client1.post("/api/chat", json={"message": "First message"})
    assert response1.status_code == 200
    conversation_id = response1.json()["conversation_id"]
    connection1.close()

    # Simulate an app restart: a brand new connection to the same database
    # file, a brand new ConversationService/ChatAgent/app -- nothing shared
    # in memory with the first app instance.
    app2, conversations2, connection2 = _build_app(
        db_path, ScriptedChatClient([ModelResponse(content="Second reply", tool_calls=[])])
    )
    client2 = TestClient(app2)
    response2 = client2.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Second message"}
    )

    assert response2.status_code == 200
    assert response2.json()["conversation_id"] == conversation_id

    history = conversations2.get_history(UUID(conversation_id))
    assert [m.content for m in history] == [
        "First message",
        "First reply",
        "Second message",
        "Second reply",
    ]
    connection2.close()


def test_post_chat_rejects_blank_message(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.sqlite3"
    app, _conversations, connection = _build_app(db_path, ScriptedChatClient([]))
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422
    connection.close()


def test_post_chat_rejects_message_over_4000_characters(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.sqlite3"
    app, _conversations, connection = _build_app(db_path, ScriptedChatClient([]))
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "a" * 4001})

    assert response.status_code == 422
    connection.close()


def test_post_chat_rejects_unknown_conversation(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.sqlite3"
    app, _conversations, connection = _build_app(db_path, ScriptedChatClient([]))
    client = TestClient(app)

    response = client.post(
        "/api/chat", json={"conversation_id": str(uuid4()), "message": "Hello"}
    )

    assert response.status_code == 404
    connection.close()


def test_post_chat_surfaces_model_unavailable(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.sqlite3"
    model = FailingChatClient(ModelUnavailableError("local chat model is unreachable"))
    app, conversations, connection = _build_app(db_path, model)
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 503
    body = response.json()
    # No fabricated success reply, and no debug trace/internals leaked.
    assert "reply" not in body
    assert "Traceback" not in response.text

    # The user message is persisted; no assistant reply exists because the
    # model call failed.
    history = conversations.get_history
    conversation_rows = connection.execute("SELECT id FROM conversations").fetchall()
    assert len(conversation_rows) == 1
    conversation_id = UUID(conversation_rows[0]["id"])
    messages = history(conversation_id)
    assert [m.role for m in messages] == ["user"]
    connection.close()


def test_get_root_serves_english_chat_ui(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.sqlite3"
    app, _conversations, connection = _build_app(db_path, ScriptedChatClient([]))
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()
    assert "lang=\"en\"" in response.text.lower()
    connection.close()
