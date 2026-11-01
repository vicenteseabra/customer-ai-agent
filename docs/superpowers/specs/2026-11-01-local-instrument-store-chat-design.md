# Design: Local Instrument Store Self-Service Chat

**Status:** awaiting approval.

## Intent

Build a local MVP chat for a musical instrument store. The agent should answer questions using the store's internal policies and query authoritative catalog, promotion, and order data while persisting store data and conversation history. The goal is to validate the experience without recurring AI API charges. The entire project, including source code, documentation, browser UI, and customer-facing agent responses, will be in English.

## Goals

- Provide a simple browser chat UI and a Python backend.
- Use RAG to retrieve information from the internal policies PDF.
- Query categories, products, promotions, and orders through structured operations over relational data.
- Import the existing CSVs into the local database: `categories`, `customers`, `order_items`, `orders`, `products`, and `promotions`.
- Persist conversations and messages on the server; associate a conversation with a customer only after verification.
- Require the order number and associated email to verify a customer; after verification, allow queries about that customer's associated orders.
- Run response generation locally, without per-call AI API charges.
- Document how to clone, configure, and run the project on another machine without prior knowledge of the repository.

## Out of scope

- Internet deployment or initial WhatsApp integration.
- Access by other users/devices on the local network; the MVP is for one person on their own machine.
- Human-agent handoff.
- An admin panel, product creation/editing through the UI, or continuous import from external systems.
- Creating, changing, or canceling orders, or modifying customer data.
- Full account authentication; order verification must not be treated as general-purpose login.

## Success criteria

- The agent answers policy questions based on the PDF and does not present information absent from it as store policy.
- Catalog, promotion, and order answers come from local tables; the model does not invent availability or status.
- Verification requires the order number and email to match the same order; subsequent queries are limited to that customer's orders. Failed verification does not reveal whether either value exists independently.
- Messages and conversations persist without automatic expiration, including across backend restarts, and can be resumed in the browser.
- The main flow works locally without paid AI service calls.
- Someone unfamiliar with the project can start the MVP by following the repository documentation.
- All project and customer-facing text is in English.

## Approaches considered

1. **Local application and AI with SQLite (selected):** simple web UI, Python backend, SQLite for structured data and conversations, a local vector index for the PDF, and Ollama for inference. This is the simplest option for local testing without per-call costs; it depends on the computer's resources and is not a public deployment.
2. **Local application and AI with PostgreSQL + pgvector:** also avoids per-call charges and can keep data and vectors in a more capable database, but adds installation and maintenance for a local MVP.
3. **Local database with an AI API free tier:** reduces local inference requirements, but needs internet, has limits that may change, and sends the content needed for generation to the provider. It does not guarantee ongoing free use or keep all processing local.

The first approach was selected because it fits the local MVP and the preference to avoid extra costs. A hosted option can be reconsidered if the app must be published or local hardware is insufficient.

## Proposed design

- **UI:** a simple browser chat page with no admin panel. Plain HTML/CSS/JavaScript without a framework is a builder choice to minimize setup.
- **Backend:** a local Python API. FastAPI is the proposed choice to serve the UI and expose the chat endpoint; the internal protocol should allow switching models without rewriting business rules.
- **Persistence:** SQLite stores the six imported tables plus conversations and messages. The importer validates expected columns and relationships and reports invalid rows clearly. Actual CSV headers and data types must be inspected when the files are provided.
- **Local use and public source repository:** run only on the user's machine and bind the server to `127.0.0.1`; no multi-user login is required. The public GitHub repository must document prerequisites, configuration, fictitious data import, and how to start each component. Do not commit the generated SQLite database, vector indexes, logs, or secrets.
- **RAG:** extract text from the PDF, split it into chunks, generate embeddings locally, and persist a local vector index. Policy answers should use retrieved chunks and say when the information is not present in the document.
- **Model:** Ollama runs a local model. The exact model is a builder choice after checking available resources; downloading and inference use disk space and computer resources even though there is no per-call charge.
- **Agent actions:** typed read-only functions query products, categories, promotions, and orders in SQLite. The model does not construct SQL or access the database directly. For this MVP, “API” means these internal backend operations; there is no external commerce API because the current sources are CSVs.
- **Orders and privacy:** request the order number and associated email before identifying the customer; compare both against the same order and return only information needed to answer. After verification, allow queries only for orders associated with that customer. Failure should return a generic verification message without confirming whether either value exists independently.
- **History:** persist messages and the conversation identifier on the server without automatic expiration. The UI may keep an opaque conversation identifier to resume after a page reload. Associate history with a customer only after verification; if deletion is needed, it is done manually in the local database, outside the MVP UI.
- **Unknown data:** the column that represents availability/stock has not been specified. If the CSVs do not contain this information, the agent must not claim that a product is available and should say that it cannot confirm.
- **Language:** code, comments, configuration, documentation, UI, and agent responses are in English. The supplied policies PDF may be in another language; the builder should preserve the source text for retrieval while generating English responses.

## Constraints and builder decisions

- The workspace does not yet contain the application, CSVs, or PDF; schema mapping and indexing depend on the fictitious challenge files being provided.
- Keep the first version local and read-only for business queries.
- Select lightweight local dependencies and document how to install/start the database, Ollama, backend, and UI.
- Keep store files and content local; do not send the PDF, customer data, orders, or conversations to external services in the selected path. Fictitious challenge files supplied by the user may be included in the public repository; generated runtime data remains ignored by Git.
- Before implementation, check whether the computer can run the selected model; if not, propose an alternative and its tradeoffs before switching to hosted inference.
