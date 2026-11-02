"""Local chat agent orchestration: prompt building and typed tool dispatch.

`ChatAgent` is the only piece of this system that talks to a local chat
model. It builds an English system prompt from retrieved policy chunks and
conversation history, asks the model what to do, and dispatches any tool
calls the model makes to the real repository/service methods -- but only
after validating each call against a server-side allowlist of declared tools
and a Pydantic input schema per tool. The model never sees Python callables;
it only ever sees tool name/description/JSON-schema.

Two invariants this module exists to protect:

- No business fact (product availability, order status, ...) is ever
  asserted unless it came back from a tool call; no store policy is ever
  asserted unless `PolicyIndex.retrieve` returned a relevant chunk for it.
- The order-listing tool can only ever return the calling conversation's own
  verified customer's orders: it takes no model-suppliable customer
  identifier at all (an attempt to supply one is a validation failure, not a
  silently-ignored extra field), so there is no path through this module
  that leaks another customer's orders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from store_chat.config import DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_BASE_URL
from store_chat.conversations import ConversationService
from store_chat.policy_rag import PolicyChunk, PolicyIndex
from store_chat.repositories import StoreRepository

# Safety margin against a model that never stops calling tools.
MAX_TOOL_ITERATIONS = 6


class ModelUnavailableError(Exception):
    """Raised when the local chat model cannot be reached or fails to respond."""


class InvalidToolCallError(Exception):
    """Raised when the model calls an undeclared tool, or calls a declared tool
    with arguments that fail validation against its input schema.

    Never silently coerced or ignored: this always propagates out of
    `ChatAgent.respond` rather than becoming a guessed answer.
    """


@dataclass(frozen=True)
class ChatMessage:
    """One message in a model conversation turn.

    `tool_call_id` and `name` only apply to `role="tool"` messages: `name`
    is the tool that produced this result (the real tool name, e.g.
    "search_products") and `tool_call_id` is the id of the specific call it
    answers -- the two are deliberately separate fields so one is never
    mistaken for the other when building the wire format for a real model.

    `tool_calls` only applies to `role="assistant"` messages: it records the
    tool calls the model itself requested on that turn, so that turn can be
    represented in message history exactly as it happened (assistant turn
    that requested tool calls, followed by the tool-result message(s)
    answering them) rather than silently omitted.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list["ToolCall"] | None = None


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    name: str
    arguments: dict[str, object]
    call_id: str


@dataclass(frozen=True)
class ModelResponse:
    """What the local chat model returned for one `chat()` call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class AgentTool:
    """A tool declaration sent to the model: name, description, and JSON schema only.

    The executable handler for a tool of this name is kept in
    `ChatAgent`'s own server-side allowlist and is never part of this
    object, so it never reaches the model.
    """

    name: str
    description: str
    parameters: dict[str, object]


class LocalChatClient(Protocol):
    """A local chat model that can be given tool declarations and asked to respond."""

    def chat(self, messages: list[ChatMessage], tools: list[AgentTool]) -> ModelResponse:
        """Return the model's next turn, given the conversation so far and the
        declared tools it may call.

        Implementations must raise `ModelUnavailableError` (not return a
        guessed/success-shaped response) when the local model cannot be
        reached or fails.
        """
        ...


def _to_ollama_message(message: ChatMessage) -> dict[str, object]:
    """Convert a `ChatMessage` to the plain dict shape the installed `ollama`
    package's `Client.chat` expects.

    A `role="tool"` message's `tool_name` field is the real tool name
    (`message.name`), never `message.tool_call_id` -- the id has no place in
    Ollama's wire format and is purely this module's own internal
    correlation key (see `ChatMessage`'s docstring). A `role="assistant"`
    message that recorded the model's own tool-call request carries those
    calls back out as Ollama's `tool_calls` shape, so a resumed conversation
    looks like a standard assistant-requests-tools -> tool-results transcript.
    """
    payload: dict[str, object] = {"role": message.role, "content": message.content}
    if message.role == "tool":
        if message.name is not None:
            payload["tool_name"] = message.name
    elif message.role == "assistant" and message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return payload


def _to_ollama_tool(tool: AgentTool) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class OllamaChatClient:
    """`LocalChatClient` backed by a local Ollama server.

    Never contacts a hosted model: `base_url` must point at a local Ollama
    instance (default `http://localhost:11434`). Any failure to reach the
    server or to get a usable response is raised as `ModelUnavailableError`.
    """

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ) -> None:
        import ollama

        self._model = model
        self._client = ollama.Client(host=base_url)

    def chat(self, messages: list[ChatMessage], tools: list[AgentTool]) -> ModelResponse:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[_to_ollama_message(message) for message in messages],
                tools=[_to_ollama_tool(tool) for tool in tools],
            )
        except Exception as exc:  # the ollama client raises several distinct types
            raise ModelUnavailableError(
                f"local chat model '{self._model}' is unavailable: {exc}"
            ) from exc

        message = response.message
        raw_tool_calls = message.tool_calls or []
        tool_calls = [
            ToolCall(
                name=raw_call.function.name,
                arguments=dict(raw_call.function.arguments),
                call_id=f"call-{index}",
            )
            for index, raw_call in enumerate(raw_tool_calls)
        ]
        return ModelResponse(content=message.content, tool_calls=tool_calls)


# --- Typed tool input schemas -----------------------------------------------
#
# Every tool's arguments are validated against one of these Pydantic models
# before the corresponding repository/service method is ever invoked.
# `extra="forbid"` means an undeclared argument (e.g. a model-supplied
# customer id on the order-listing tool) is a validation failure, not a
# silently-ignored extra field.


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchProductsInput(_StrictModel):
    query: str
    category: str | None = None
    limit: int = 5


class ListCategoriesInput(_StrictModel):
    pass


class ListPromotionsInput(_StrictModel):
    pass


class VerifyCustomerInput(_StrictModel):
    order_number: str
    email: str


class ListVerifiedOrdersInput(_StrictModel):
    """Deliberately has no fields: the order-listing tool accepts no
    model-suppliable parameters at all. Customer scope comes only from the
    conversation id the agent already holds, never from a tool argument.
    """


@dataclass(frozen=True)
class _ToolDefinition:
    tool: AgentTool
    input_model: type[BaseModel]
    handler: Callable[["ChatAgent", BaseModel, UUID], str]


def _search_products_handler(agent: "ChatAgent", args: SearchProductsInput, _conversation_id: UUID) -> str:
    results = agent._store.search_products(args.query, args.category, args.limit)
    return json.dumps(
        [
            {
                "product_id": r.product_id,
                "name": r.name,
                "description": r.description,
                "price_brl": r.price_brl,
                "category": r.category,
                "stock_quantity": r.stock_quantity,
                "status": r.status,
            }
            for r in results
        ]
    )


def _list_categories_handler(agent: "ChatAgent", _args: ListCategoriesInput, _conversation_id: UUID) -> str:
    results = agent._store.list_categories()
    return json.dumps(
        [
            {"category_id": c.category_id, "name": c.name, "description": c.description}
            for c in results
        ]
    )


def _list_promotions_handler(agent: "ChatAgent", _args: ListPromotionsInput, _conversation_id: UUID) -> str:
    results = agent._store.list_promotions()
    return json.dumps(
        [
            {
                "promotion_id": p.promotion_id,
                "product_id": p.product_id,
                "product_name": p.product_name,
                "discount_percent": p.discount_percent,
                "description": p.description,
                "is_active": p.is_active,
            }
            for p in results
        ]
    )


def _verify_customer_handler(agent: "ChatAgent", args: VerifyCustomerInput, conversation_id: UUID) -> str:
    verified = agent._conversations.verify_customer(conversation_id, args.order_number, args.email)
    # Deliberately the only field returned, win or lose: never reveals which
    # half of (order_number, email) was wrong.
    return json.dumps({"verified": verified})


def _list_verified_orders_handler(
    agent: "ChatAgent", _args: ListVerifiedOrdersInput, conversation_id: UUID
) -> str:
    # Customer scope comes only from the conversation id, via
    # ConversationService.list_verified_customer_orders -- there is no
    # argument here through which a different customer could be requested.
    orders = agent._conversations.list_verified_customer_orders(conversation_id)
    return json.dumps(
        [
            {
                "order_id": o.order_id,
                "customer_id": o.customer_id,
                "order_date": o.order_date,
                "status": o.status,
                "total_brl": o.total_brl,
                "payment_method": o.payment_method,
                "tracking_code": o.tracking_code,
                "estimated_delivery": o.estimated_delivery,
                "notes": o.notes,
            }
            for o in orders
        ]
    )


_SYSTEM_PROMPT_HEADER = """You are the customer service assistant for Empório da Música, a musical
instrument store in Campo Grande, Mato Grosso do Sul, Brazil, founded in 2008.
The store specializes exclusively in musical instruments for beginner,
intermediate, and professional musicians -- it does not sell accessories
such as strings, picks, cables, cases, pedals, or amplifiers. Speak in a
warm, knowledgeable, and professional tone, like a specialist who
genuinely wants to help musicians find the right instrument and get
accurate answers about their orders.

Only discuss topics related to Empório da Música: its products,
categories, promotions, orders, and store policies. If a customer asks
about something unrelated to the store (general knowledge, other
businesses, or anything outside this role), politely decline and steer
the conversation back to how you can help with their shopping or order.

Rules you must always follow:
- Always answer in English, no matter what language any reference material below is written in.
- Use the provided tools to look up real product, category, promotion, and order information. Never guess or invent business facts.
- Only report product availability (stock or status) using the stock_quantity/status fields a tool result actually returned; never claim, imply, or estimate availability beyond what a tool call returned for that specific product.
- Only state a store policy if it is backed by the "Relevant store policy excerpts" section below. If that section says none were found, or the excerpts don't cover the question, say plainly that you don't have that information -- never assert a policy that isn't backed by a retrieved excerpt.
- To verify a customer's identity or list their orders, call the verification/order-listing tools. You can never supply or guess a customer id yourself; the order-listing tool does not accept one."""


def _build_system_prompt(policy_chunks: list[PolicyChunk]) -> str:
    lines = [_SYSTEM_PROMPT_HEADER, ""]
    if policy_chunks:
        lines.append("Relevant store policy excerpts:")
        for chunk in policy_chunks:
            lines.append(f"- ({chunk.source}) {chunk.text}")
    else:
        lines.append("Relevant store policy excerpts: none found for this question.")
    return "\n".join(lines)


class ChatAgent:
    """Orchestrates one user turn: prompt + retrieval + typed tool dispatch."""

    def __init__(
        self,
        model_client: LocalChatClient,
        policies: PolicyIndex,
        store: StoreRepository,
        conversations: ConversationService,
    ) -> None:
        self._model_client = model_client
        self._policies = policies
        self._store = store
        self._conversations = conversations
        self._tool_definitions = self._build_tool_definitions()

    def _build_tool_definitions(self) -> dict[str, _ToolDefinition]:
        definitions = [
            _ToolDefinition(
                tool=AgentTool(
                    name="search_products",
                    description=(
                        "Search the store's product catalog by a free-text query, "
                        "optionally filtered by category name. Returns matching "
                        "products with their real price, category, stock_quantity, "
                        "and status fields."
                    ),
                    parameters=SearchProductsInput.model_json_schema(),
                ),
                input_model=SearchProductsInput,
                handler=_search_products_handler,
            ),
            _ToolDefinition(
                tool=AgentTool(
                    name="list_categories",
                    description="List all product categories the store carries.",
                    parameters=ListCategoriesInput.model_json_schema(),
                ),
                input_model=ListCategoriesInput,
                handler=_list_categories_handler,
            ),
            _ToolDefinition(
                tool=AgentTool(
                    name="list_promotions",
                    description="List current and past promotions, including whether each is active.",
                    parameters=ListPromotionsInput.model_json_schema(),
                ),
                input_model=ListPromotionsInput,
                handler=_list_promotions_handler,
            ),
            _ToolDefinition(
                tool=AgentTool(
                    name="verify_customer",
                    description=(
                        "Verify the current customer's identity using an order "
                        "number and the email on that order. Both must be supplied "
                        "and must belong to the same order. Returns only whether "
                        "verification succeeded, never which part was wrong."
                    ),
                    parameters=VerifyCustomerInput.model_json_schema(),
                ),
                input_model=VerifyCustomerInput,
                handler=_verify_customer_handler,
            ),
            _ToolDefinition(
                tool=AgentTool(
                    name="list_verified_orders",
                    description=(
                        "List the orders belonging to the customer already verified "
                        "on this conversation (via verify_customer). Takes no "
                        "parameters: it can never be used to look up another "
                        "customer's orders, and returns an empty list if this "
                        "conversation has not been verified yet."
                    ),
                    parameters=ListVerifiedOrdersInput.model_json_schema(),
                ),
                input_model=ListVerifiedOrdersInput,
                handler=_list_verified_orders_handler,
            ),
        ]
        return {definition.tool.name: definition for definition in definitions}

    def _declared_tools(self) -> list[AgentTool]:
        return [definition.tool for definition in self._tool_definitions.values()]

    def _dispatch_tool_call(self, call: ToolCall, conversation_id: UUID) -> str:
        definition = self._tool_definitions.get(call.name)
        if definition is None:
            raise InvalidToolCallError(f"model called an undeclared tool: '{call.name}'")
        try:
            validated_args = definition.input_model(**call.arguments)
        except ValidationError as exc:
            raise InvalidToolCallError(
                f"invalid arguments for tool '{call.name}': {exc}"
            ) from exc
        return definition.handler(self, validated_args, conversation_id)

    def respond(self, conversation_id: UUID, message: str) -> str:
        """Handle one user turn and return the final English assistant reply.

        Persists the user message immediately, then repeatedly asks the
        model for its next turn, dispatching any (validated) tool calls it
        makes, until the model returns a final answer with no further tool
        calls. The final reply is persisted before being returned. Any
        `ModelUnavailableError`/`InvalidToolCallError` raised during this
        process propagates to the caller rather than being turned into a
        guessed reply.
        """
        self._conversations.append_message(conversation_id, "user", message)

        policy_chunks = self._policies.retrieve(message)
        system_prompt = _build_system_prompt(policy_chunks)

        history = self._conversations.get_history(conversation_id)
        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
        messages.extend(ChatMessage(role=m.role, content=m.content) for m in history)

        declared_tools = self._declared_tools()

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._model_client.chat(messages, declared_tools)

            if not response.tool_calls:
                final_reply = response.content or ""
                self._conversations.append_message(conversation_id, "assistant", final_reply)
                return final_reply

            # Record the assistant turn that requested these tool calls
            # before the tool-result messages answering them, so the next
            # round sees the standard transcript shape: assistant-with-
            # tool-calls -> tool-result(s) -> (eventual) next assistant turn.
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=list(response.tool_calls),
                )
            )

            for call in response.tool_calls:
                result = self._dispatch_tool_call(call, conversation_id)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result,
                        tool_call_id=call.call_id,
                        name=call.name,
                    )
                )

        raise ModelUnavailableError(
            "local chat model did not produce a final answer after "
            f"{MAX_TOOL_ITERATIONS} tool-call rounds"
        )
