"""FastAPI application exposing the chat agent over HTTP, plus the static browser UI.

`create_app` is the dependency-injection seam: it takes an already-built
`ChatAgent` and `ConversationService`, so tests can supply test doubles
(scripted/failing `ChatClient`, a temporary SQLite file) without ever
touching a real Gemini API or a shared database. `build_app` is the runtime
wiring that constructs the real implementations from `Settings`: local
embeddings/RAG (`OllamaEmbeddingClient`, `ChromaVectorStore`) and a Gemini
chat client (`GeminiChatClient`).

Exception-to-status-code mapping:
- `ModelUnavailableError` (the chat model could not be reached, or never
  produced a final answer) -> 503 Service Unavailable. This is a
  "try again later" condition, not a client input problem.
- `InvalidToolCallError` (the model itself called an undeclared tool, or
  called a declared tool with invalid arguments) -> 502 Bad Gateway. This is
  server-side/model misbehavior -- the user's request was well-formed, but
  the upstream model did not honor the tool contract this server enforces.
  It is deliberately not a 503 (the model process answered; it just
  misbehaved) and never a 500 (the handler is expected and provides a
  clean, generic message rather than an unhandled-exception traceback).

In both cases the route lets the exception propagate as an `HTTPException`
with a short, generic English message: no stack trace, SQL, or other
internal detail is ever included in the response body.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from uuid import UUID

from store_chat.agent import (
    ChatAgent,
    GeminiChatClient,
    InvalidToolCallError,
    ModelUnavailableError,
)
from store_chat.config import Settings, load_settings
from store_chat.conversations import ConversationService
from store_chat.db import get_connection, initialize_database
from store_chat.policy_rag import ChromaVectorStore, OllamaEmbeddingClient, PolicyIndex
from store_chat.repositories import StoreRepository

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    """One chat turn request from the browser UI.

    `message` is stripped of surrounding whitespace before length
    validation, so a blank-looking message (all whitespace) is rejected as
    empty rather than accepted as a 4000-character-eligible string of
    spaces.
    """

    conversation_id: UUID | None = None
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message", mode="before")
    @classmethod
    def _strip_message(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ChatResponse(BaseModel):
    """The assistant's reply to one chat turn, and the conversation it belongs to."""

    conversation_id: UUID
    reply: str


def create_app(
    settings: Settings, agent: ChatAgent, conversations: ConversationService
) -> FastAPI:
    """Build the FastAPI app from already-constructed dependencies.

    This is the dependency-injection seam tests use: callers supply the
    `ChatAgent`/`ConversationService` (built from test doubles or real
    implementations), rather than this function constructing them itself.
    """
    app = FastAPI(title="Store Chat")
    app.state.settings = settings

    # Defined as `async def` (even though the underlying SQLite/model calls
    # are synchronous) so FastAPI runs the handler directly on the event
    # loop thread rather than offloading it to a worker thread: the
    # injected `sqlite3.Connection` is bound to a single thread and is not
    # safe to use from FastAPI's default sync-route threadpool.
    @app.post("/api/chat", response_model=ChatResponse)
    async def post_chat(request: ChatRequest) -> ChatResponse:
        if request.conversation_id is None:
            conversation = conversations.create_conversation()
        else:
            conversation = conversations.get_conversation(request.conversation_id)
            if conversation is None:
                raise HTTPException(
                    status_code=404, detail="This conversation does not exist."
                )

        try:
            reply = agent.respond(conversation.id, request.message)
        except ModelUnavailableError:
            raise HTTPException(
                status_code=503,
                detail="The chat model is unavailable right now. Please try again shortly.",
            )
        except InvalidToolCallError:
            raise HTTPException(
                status_code=502,
                detail="The assistant could not complete this request. Please try again.",
            )

        return ChatResponse(conversation_id=conversation.id, reply=reply)

    @app.get("/")
    async def get_root() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def build_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app with real runtime dependencies.

    Constructs a SQLite connection, the store repository, the conversation
    service, a local-PDF-backed policy index (local Ollama embeddings + a
    persistent on-disk ChromaDB collection), and a Gemini chat client --
    all driven by `settings` (loaded from the environment when not
    supplied).
    """
    if settings is None:
        settings = load_settings()

    connection = get_connection(settings.database_path)
    initialize_database(connection)

    store = StoreRepository(connection)
    conversations = ConversationService(connection, store)

    embedding_client = OllamaEmbeddingClient(
        model=settings.embedding_model, base_url=settings.ollama_base_url
    )
    vector_store = ChromaVectorStore(path=settings.vector_store_path)
    policies = PolicyIndex(
        embedding_client,
        vector_store,
        chunk_size=settings.policy_chunk_size,
        chunk_overlap=settings.policy_chunk_overlap,
        relevance_distance_threshold=settings.policy_relevance_distance_threshold,
    )

    model_client = GeminiChatClient(model=settings.chat_model, api_key=settings.gemini_api_key)
    agent = ChatAgent(model_client, policies, store, conversations)

    return create_app(settings, agent, conversations)
