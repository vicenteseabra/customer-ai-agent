# Local Instrument Store Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, English-language self-service chat for a musical instrument store that uses local RAG for policies, read-only database tools for store data, and permanently persisted conversation history.

**Architecture:** Serve a plain browser UI and a Python API bound to `127.0.0.1`. Store imported business data and conversations in SQLite, index the policies PDF in a persistent local vector store, and use Ollama for local generation and embeddings. The agent may call typed, read-only backend functions; it never constructs SQL.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, SQLite (`sqlite3`), Ollama, ChromaDB, pypdf, plain HTML/CSS/JavaScript, pytest, HTTPX for API tests, and ReportLab for generated PDF test fixtures. Use the Ollama client for local model requests. No hosted AI API is part of the selected path.

**Spec:** [docs/superpowers/specs/2026-11-01-local-instrument-store-chat-design.md](../specs/2026-11-01-local-instrument-store-chat-design.md)

## Global Constraints

- The entire project is in English: source code, comments, configuration, documentation, browser UI, and customer-facing agent responses.
- Run locally for one person and bind the server to `127.0.0.1`; do not expose it on the local network or the internet.
- Use SQLite for the six CSV-backed tables and chat history; use a persistent local vector index for the policies PDF.
- Use Ollama locally for generation and embeddings; do not send policies, customer data, orders, or conversations to an external service.
- Support the supplied `categories`, `customers`, `order_items`, `orders`, `products`, and `promotions` CSVs and the policies PDF; inspect actual headers before mapping fields.
- Store conversations and messages without automatic expiration. The UI resumes the active conversation using an opaque conversation ID.
- Require the order number and associated email to match the same order before associating a conversation with a customer. After verification, permit access only to that customer's orders.
- Business-data tools are read-only. Failed order verification must not disclose whether either identifier exists independently.
- Do not commit generated SQLite files, vector indexes, logs, or secrets. Only fictitious challenge data may be included in the public repository.
- Document setup so a new user can run the app locally without prior repository knowledge.

## Review Focus

- **Missing/malformed CSVs or inconsistent rows:** the importer should fail clearly and avoid a partial import; pin this in Task 2's missing-file, missing-column, invalid-row, and atomicity tests.
- **Mismatched or cross-customer order access:** only a matching order number/email pair verifies a customer, and later order queries remain scoped to that customer; pin this in Task 3's identity and isolation tests.
- **Empty, unreadable, or irrelevant policy PDF:** the agent must not invent a policy answer when retrieval has no relevant source; pin this in Task 4's indexing and retrieval tests.
- **Unavailable local model or malformed tool call:** the request must surface an actionable error and must not return a success-shaped guess; pin this in Task 5's model/tool failure tests.
- **Invalid chat input, unknown conversation, or restart:** reject invalid requests clearly and preserve valid conversation history across a backend restart; pin this in Tasks 3 and 6.

## Execution Precondition

Before implementing the importer in Task 2, the user must place the fictitious challenge CSVs and policies PDF in `data/source/`. Inspect their headers and relationships before fixing the importer mapping. Do not invent missing fields; if an essential relation or a stock/availability field is absent, document the limitation and keep the corresponding answer unavailable rather than guessing.

The current workspace is not a Git repository. Do not run Git commit commands until the user initializes or supplies the public repository.

## File Structure

- Create `pyproject.toml` — package metadata, runtime dependencies, and pytest configuration.
- Create `.gitignore` — exclude local databases, vector indexes, environment files, caches, and logs.
- Create `.env.example` — document local model and file-path settings without secrets.
- Create `src/store_chat/__init__.py` — package marker.
- Create `src/store_chat/config.py` — typed settings and local default paths.
- Create `src/store_chat/db.py` — SQLite connection setup and schema initialization for conversations and messages.
- Create `src/store_chat/importer.py` — validated, atomic import of the six source CSVs using mappings based on the supplied headers.
- Create `src/store_chat/repositories.py` — parameterized read-only queries for categories, products, promotions, and orders.
- Create `src/store_chat/conversations.py` — conversation/message persistence and verified-customer association.
- Create `src/store_chat/policy_rag.py` — PDF extraction, local embedding, Chroma persistence, and policy retrieval.
- Create `src/store_chat/agent.py` — Ollama orchestration, English response instructions, and typed tool dispatch.
- Create `src/store_chat/cli.py` — documented `import-data` and `index-policies` setup commands.
- Create `src/store_chat/api.py` — request/response models and `/api/chat` route.
- Create `src/store_chat/main.py` — app construction, static-file serving, and local startup entry point.
- Create `src/store_chat/static/index.html`, `styles.css`, and `app.js` — English browser chat and conversation resume.
- Create `tests/conftest.py` — temporary SQLite database, fixture CSVs, and fake model/vector dependencies.
- Create `tests/test_db.py`, `test_importer.py`, `test_repositories.py`, `test_conversations.py`, `test_policy_rag.py`, `test_agent.py`, `test_api.py`, and `test_cli.py` — focused unit and API coverage.
- Create `README.md` — English clone, installation, Ollama setup, import, startup, and troubleshooting instructions for Windows, macOS, and Linux.
- Use `data/source/` for the supplied fictitious CSVs and PDF at execution time; do not create or assume these files in the plan.

---

### Task 1: Create the Python package and persistent database foundation

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/store_chat/__init__.py`
- Create: `src/store_chat/config.py`
- Create: `src/store_chat/db.py`
- Test: `tests/conftest.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces `Settings` with `database_path: Path`, `source_data_dir: Path`, `vector_store_path: Path`, `ollama_base_url: str`, `chat_model: str`, and `embedding_model: str`.
- Produces `load_settings() -> Settings`, reading environment overrides and using local defaults.
- Reads `STORE_CHAT_DB_PATH`, `STORE_CHAT_SOURCE_DIR`, `STORE_CHAT_VECTOR_PATH`, `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, and `OLLAMA_EMBEDDING_MODEL`.
- Produces `get_connection(database_path: Path) -> sqlite3.Connection`, with SQLite foreign-key enforcement enabled.
- Produces `initialize_database(connection: sqlite3.Connection) -> None`, creating the conversation tables idempotently.
- Produces test fixtures `database_path` and `connection` backed by pytest temporary directories.
- Configure the `src` package layout and include the static UI files in package data so the documented app command can serve them.

- [ ] **Step 1: Write the failing database tests**

  Add `test_initialize_database_creates_conversation_tables`, asserting `conversations` and `messages` exist with their required columns and foreign key, and `test_initialize_database_is_idempotent`, asserting a second initialization succeeds without deleting stored messages. Add `pyproject.toml` with runtime dependencies, pytest, HTTPX, and ReportLab as development dependencies.

- [ ] **Step 2: Install the project and test dependencies**

  Run: `python -m pip install -e ".[dev]"`
  Expected: package and declared development dependencies install successfully.

- [ ] **Step 3: Run the tests and verify they fail**

  Run: `python -m pytest tests/test_db.py -q`
  Expected: FAIL because database functions are not implemented.

- [ ] **Step 4: Implement package settings and database foundation**

  Implement the interfaces above. `conversations` stores an opaque ID, nullable verified `customer_id`, and timestamps. `messages` stores an ID, conversation ID, role (`user` or `assistant`), content, and timestamp. Enable foreign keys and use parameterized SQL for all database operations.

- [ ] **Step 5: Run the focused tests**

  Run: `python -m pytest tests/test_db.py -q`
  Expected: PASS; initialization is repeatable and existing messages remain.

### Task 2: Import CSV data and implement read-only store repositories

**Files:**
- Create: `src/store_chat/importer.py`
- Create: `src/store_chat/repositories.py`
- Test: `tests/test_importer.py`
- Test: `tests/test_repositories.py`
- Input: User-provided fictitious CSVs in `data/source/`

**Interfaces:**
- Produces `ImportReport` with `imported_rows: dict[str, int]`.
- Produces `CsvImportError(errors: list[str])` for actionable import failures.
- Produces `import_csv_directory(connection: sqlite3.Connection, source_dir: Path) -> ImportReport`; on invalid input it raises `CsvImportError` and rolls back the full import.
- Produces `StoreRepository(connection: sqlite3.Connection)` with:
  - `search_products(query: str, category: str | None = None, limit: int = 5) -> list[ProductSummary]`
  - `list_categories() -> list[CategorySummary]`
  - `list_promotions() -> list[PromotionSummary]`
  - `find_matching_order(order_number: str, email: str) -> VerifiedOrderIdentity | None`
  - `list_orders_for_customer(customer_id: str) -> list[OrderSummary]`
- `ProductSummary`, `CategorySummary`, `PromotionSummary`, `VerifiedOrderIdentity`, and `OrderSummary` are typed records containing only fields present in the supplied CSVs. Map source headers explicitly after inspection; never synthesize stock, availability, status, or relationship values.

- [ ] **Step 1: Inspect the supplied CSV headers and write importer tests**

  Record the actual required columns and relationships from the supplied fictitious files. Add fixture rows based on those headers. Write tests named `test_import_csv_directory_imports_all_six_tables`, `test_import_csv_directory_rejects_missing_file`, `test_import_csv_directory_rejects_missing_required_column`, `test_import_csv_directory_rolls_back_invalid_import`, and `test_find_matching_order_requires_order_and_email_on_same_order`.

  Assert that valid rows are imported with correct relationships; a missing file/column or invalid relationship reports the table and row; and a failed import leaves no partial rows.

- [ ] **Step 2: Run the importer/repository tests and verify they fail**

  Run: `python -m pytest tests/test_importer.py tests/test_repositories.py -q`
  Expected: FAIL because importer and repositories are not implemented.

- [ ] **Step 3: Implement atomic CSV import and typed read-only queries**

  Create SQLite business tables from the inspected source schema and explicit mappings. Normalize source values without discarding source columns needed by the challenge. Wrap all six imports in one transaction. Use parameterized SQL and return only fields needed by the chat tools.

- [ ] **Step 4: Add repository isolation tests**

  Add `test_list_orders_for_customer_never_returns_another_customers_orders` and `test_product_search_does_not_claim_missing_stock`.

- [ ] **Step 5: Run the focused tests**

  Run: `python -m pytest tests/test_importer.py tests/test_repositories.py -q`
  Expected: PASS; valid fixtures import completely, invalid imports roll back, and repository results are customer-scoped.

### Task 3: Persist conversations and enforce verified-customer order access

**Files:**
- Create: `src/store_chat/conversations.py`
- Test: `tests/test_conversations.py`

**Interfaces:**
- `Conversation` contains `id: UUID`, `customer_id: str | None`, `created_at: datetime`, and `updated_at: datetime`.
- `Message` contains `id: UUID`, `conversation_id: UUID`, `role: Literal["user", "assistant"]`, `content: str`, and `created_at: datetime`.
- Produces `ConversationService(connection: sqlite3.Connection, store: StoreRepository)`.
- `create_conversation() -> Conversation` creates an opaque UUID conversation ID.
- `get_conversation(conversation_id: UUID) -> Conversation | None`.
- `append_message(conversation_id: UUID, role: Literal["user", "assistant"], content: str) -> Message`.
- `get_history(conversation_id: UUID) -> list[Message]`, ordered oldest to newest.
- `verify_customer(conversation_id: UUID, order_number: str, email: str) -> bool` associates the conversation with the customer only when `find_matching_order` matches both values on the same order.
- `list_verified_customer_orders(conversation_id: UUID) -> list[OrderSummary]` returns no orders unless the conversation has a verified customer, and otherwise only that customer's orders.

- [ ] **Step 1: Write failing persistence and access-control tests**

  Add `test_conversation_history_survives_database_reopen`, `test_failed_verification_does_not_associate_customer`, `test_matching_order_and_email_associates_customer`, `test_verified_customer_can_list_all_own_orders`, and `test_verified_customer_cannot_list_another_customers_orders`.

  Assert that history is returned in order after closing and reopening SQLite; mismatched credentials yield `False` and no association; a verified conversation lists all and only the associated customer's orders.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `python -m pytest tests/test_conversations.py -q`
  Expected: FAIL because conversation services are not implemented.

- [ ] **Step 3: Implement conversation persistence and verification**

  Use the repository interface from Task 2. Persist only user-visible user/assistant messages; do not save hidden tool payloads. Do not set automatic expiry. Return the same generic failed-verification result for all non-matching order/email pairs.

- [ ] **Step 4: Run the focused tests**

  Run: `python -m pytest tests/test_conversations.py -q`
  Expected: PASS; valid conversations persist and all order reads remain scoped to the verified customer.

### Task 4: Build local PDF policy retrieval

**Files:**
- Create: `src/store_chat/policy_rag.py`
- Test: `tests/test_policy_rag.py`

**Interfaces:**
- Produces `PolicyChunk` with `text: str` and `source: str`.
- Defines `LocalEmbeddingClient.embed(texts: list[str]) -> list[list[float]]`.
- Defines `LocalVectorStore.upsert(chunks: list[PolicyChunk], embeddings: list[list[float]]) -> None` and `query(embedding: list[float], limit: int) -> list[tuple[PolicyChunk, float]]`.
- Produces `PolicyIndex(embedding_client: LocalEmbeddingClient, vector_store: LocalVectorStore)`.
- `index_pdf(pdf_path: Path) -> int` extracts text with pypdf, creates deterministic chunks, embeds locally through Ollama, and persists the collection in ChromaDB; it raises a clear error for unreadable PDFs or PDFs with no extractable text.
- `retrieve(query: str, limit: int = 4) -> list[PolicyChunk]` returns relevant chunks and source metadata, or an empty list when there is no relevant match.

- [ ] **Step 1: Write failing extraction, indexing, and retrieval tests**

  Use ReportLab to create a small generated test PDF and fake local embedding/vector clients. Add `test_index_pdf_persists_chunks`, `test_index_pdf_rejects_unreadable_pdf`, `test_index_pdf_rejects_empty_pdf`, `test_retrieve_returns_relevant_chunks_with_source`, and `test_retrieve_returns_empty_for_irrelevant_query`.

  Assert that a second index construction can retrieve the persisted chunks, and that an irrelevant query returns an empty result rather than a fabricated policy.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `python -m pytest tests/test_policy_rag.py -q`
  Expected: FAIL because PDF indexing and retrieval are not implemented.

- [ ] **Step 3: Implement local extraction, chunking, and persistent retrieval**

  Use pypdf for extraction, local Ollama embeddings, and a persistent ChromaDB collection. Keep chunking and relevance threshold settings explicit in configuration; do not fall back to hosted embeddings.

- [ ] **Step 4: Run the focused tests**

  Run: `python -m pytest tests/test_policy_rag.py -q`
  Expected: PASS; empty or irrelevant policy sources produce no answerable chunks.

### Task 5: Implement the local agent and typed tool dispatch

**Files:**
- Create: `src/store_chat/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces `ChatAgent(model_client: LocalChatClient, policies: PolicyIndex, store: StoreRepository, conversations: ConversationService)`.
- Defines `LocalChatClient.chat(messages: list[ChatMessage], tools: list[AgentTool]) -> ModelResponse`; `OllamaChatClient` implements it with the configured local model.
- `ChatMessage` contains `role: Literal["system", "user", "assistant", "tool"]`, `content: str`, and `tool_call_id: str | None`.
- `AgentTool` contains `name: str`, `description: str`, and a JSON-schema parameter definition; its executable handler is kept in a server-side allowlist, not sent to the model.
- `ModelResponse` contains `content: str | None` and `tool_calls: list[ToolCall]`; `ToolCall` contains `name: str`, `arguments: dict[str, object]`, and `call_id: str`.
- `respond(conversation_id: UUID, message: str) -> str` persists the user message, constructs an English prompt with relevant policy chunks and conversation history, dispatches only declared tool calls, persists the final English response, and returns it.
- Exposes typed tools for product search, categories, promotions, customer verification, and verified-customer order listing. The order-list tool accepts only a conversation ID and obtains customer scope from `ConversationService`; it must not accept an arbitrary customer ID from the model.
- Raises `ModelUnavailableError` or `InvalidToolCallError` for local model or tool failures; errors must be visible to the API layer and must not become guessed answers.

- [ ] **Step 1: Write failing agent behavior tests**

  Use a fake local model client. Add `test_policy_question_uses_retrieved_chunks_and_answers_in_english`, `test_missing_policy_context_does_not_claim_a_store_policy`, `test_product_question_uses_store_repository_tool`, `test_product_question_without_stock_data_does_not_claim_availability`, `test_order_verification_requires_both_values`, `test_failed_verification_uses_generic_response`, `test_order_tool_uses_verified_conversation_scope`, `test_agent_rejects_undeclared_tool`, and `test_agent_surfaces_unavailable_model`.

  Assert that every final response is English, business facts come from repository tool results, no policy is asserted without relevant retrieval, and an unavailable/malformed tool result raises an explicit error.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `python -m pytest tests/test_agent.py -q`
  Expected: FAIL because agent orchestration is not implemented.

- [ ] **Step 3: Implement the Ollama adapter and typed tool loop**

  Use the configured local chat model only. Parse and validate every tool call against the declared tool names and Pydantic input types before invoking repository methods. Keep the system instructions and all tool descriptions in English.

- [ ] **Step 4: Run the focused tests**

  Run: `python -m pytest tests/test_agent.py -q`
  Expected: PASS; generated answers use retrieved/database evidence and errors are not disguised as success.

### Task 6: Expose the chat API and browser UI

**Files:**
- Create: `src/store_chat/api.py`
- Create: `src/store_chat/main.py`
- Create: `src/store_chat/static/index.html`
- Create: `src/store_chat/static/styles.css`
- Create: `src/store_chat/static/app.js`
- Test: `tests/test_api.py`

**Interfaces:**
- `ChatRequest` contains `conversation_id: UUID | None` and `message: str` with non-empty, trimmed content and a 4,000-character maximum.
- `ChatResponse` contains `conversation_id: UUID` and `reply: str`.
- `create_app(settings: Settings, agent: ChatAgent, conversations: ConversationService) -> FastAPI` injects dependencies for tests and serves the English static UI.
- `build_app(settings: Settings | None = None) -> FastAPI` constructs runtime dependencies from settings.
- `main() -> None` runs Uvicorn bound to `127.0.0.1`.
- `POST /api/chat` creates a conversation when no ID is supplied, otherwise resumes the existing conversation, and returns `ChatResponse`.
- `GET /` serves the chat page. Bind the runnable server to `127.0.0.1` only.

- [ ] **Step 1: Write failing API contract tests**

  Add `test_post_chat_creates_and_persists_conversation`, `test_post_chat_resumes_conversation_after_app_restart`, `test_post_chat_rejects_blank_message`, `test_post_chat_rejects_message_over_4000_characters`, `test_post_chat_rejects_unknown_conversation`, `test_post_chat_surfaces_model_unavailable`, and `test_get_root_serves_english_chat_ui`.

  Assert response schema and status codes (`200` for success, `404` for unknown conversation, `422` for request validation, and `503` for local-model unavailability); assert that no failed model request returns a fabricated reply.

- [ ] **Step 2: Run the API tests and verify they fail**

  Run: `python -m pytest tests/test_api.py -q`
  Expected: FAIL because the API and UI are not implemented.

- [ ] **Step 3: Implement the FastAPI app and static UI**

  Create the app through dependency injection. Store the returned conversation ID in browser localStorage and send it with subsequent requests. Render messages as text, not unsanitized HTML. Display clear English loading and error states; do not expose debug traces or database contents.

- [ ] **Step 4: Run the focused tests**

  Run: `python -m pytest tests/test_api.py -q`
  Expected: PASS; the UI route works, sessions resume, invalid input is rejected, and backend/model errors are surfaced.

### Task 7: Document clean-machine setup and run the complete suite

**Files:**
- Create: `README.md`
- Create: `src/store_chat/cli.py`
- Modify: `.gitignore`
- Modify: `.env.example`
- Test: `tests/test_cli.py`
- Test: all files under `tests/`
- Input: User-provided fictitious CSVs and policies PDF in `data/source/`

**Interfaces:**
- `main(argv: list[str] | None = None) -> int` dispatches the `import-data` and `index-policies` subcommands and returns a non-zero status with an English error for invalid/missing inputs.
- Provide `python -m store_chat.cli import-data --source data/source` and `python -m store_chat.cli index-policies --pdf data/source/policies.pdf` commands for setup.
- Document commands to install Python dependencies, install/start Ollama, pull the configured local chat and embedding models, import the fictitious files, index the PDF, run `python -m store_chat.main`, and open the local UI.
- Document supported environment variables and expected CSV/PDF paths; include Windows PowerShell and macOS/Linux command variants.
- Keep generated files ignored and include no API keys or secrets.

- [ ] **Step 1: Write failing CLI tests**

  Add `test_cli_import_data_reports_missing_source_directory` and `test_cli_index_policies_reports_missing_pdf`; assert non-zero exit codes and clear English errors.

- [ ] **Step 2: Run the CLI tests and verify they fail**

  Run: `python -m pytest tests/test_cli.py -q`
  Expected: FAIL because the CLI commands are not implemented.

- [ ] **Step 3: Implement the CLI commands**

  Implement `main(argv)` using the same import and indexing services as the app. Do not duplicate business logic.

- [ ] **Step 4: Write the clean-machine README and verify repository exclusions**

  In `README.md`, write exact prerequisites and ordered steps for a new clone. Include expected success signals for the CSV import, PDF index, backend startup, and browser URL. Document how to diagnose missing files, bad CSV headers, unavailable Ollama, and model download/resource failures. Add ignore rules for `.env`, `*.sqlite`, `*.sqlite3`, the configured Chroma persistence directory, Python caches, pytest cache, and logs. Confirm fictitious source files remain trackable while generated runtime databases/indexes do not.

- [ ] **Step 5: Run the complete test suite**

  Run: `python -m pytest -q`
  Expected: PASS without requiring a running Ollama server or network access; tests use fake model/embedding clients.

- [ ] **Step 6: Perform a documented local smoke test**

  Follow the README on a clean local setup using the supplied fictitious files. Verify a policy question, a catalog lookup, failed order verification, successful order verification followed by a second associated order query, and history resumption after restart. Confirm the app listens only on `127.0.0.1`.
