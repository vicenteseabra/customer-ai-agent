"""Tests for ChatAgent: prompt construction, typed tool dispatch, and safety."""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from store_chat.agent import (
    AgentTool,
    ChatAgent,
    ChatMessage,
    InvalidToolCallError,
    ModelResponse,
    ModelUnavailableError,
    ToolCall,
    _to_ollama_message,
)
from store_chat.conversations import ConversationService
from store_chat.policy_rag import PolicyChunk
from store_chat.repositories import StoreRepository


class FakePolicyIndex:
    """Test double for PolicyIndex: returns a fixed, pre-scripted chunk list."""

    def __init__(self, chunks: list[PolicyChunk]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query: str, limit: int = 4) -> list[PolicyChunk]:
        self.queries.append(query)
        return self._chunks


class ScriptedChatClient:
    """Test double for LocalChatClient: replays a scripted sequence of responses.

    Records every (messages, tools) pair it was called with, so tests can
    assert on exactly what the agent sent to the model (system prompt
    content, tool results fed back, etc).
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[ChatMessage], list[AgentTool]]] = []

    def chat(self, messages: list[ChatMessage], tools: list[AgentTool]) -> ModelResponse:
        self.calls.append((list(messages), list(tools)))
        if not self._responses:
            raise AssertionError("ScriptedChatClient ran out of scripted responses")
        return self._responses.pop(0)


class FailingChatClient:
    """Test double for LocalChatClient: always raises, simulating an unreachable model."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def chat(self, messages: list[ChatMessage], tools: list[AgentTool]) -> ModelResponse:
        raise self._error


def _make_agent(
    model_client,
    policies,
    connection: sqlite3.Connection,
) -> tuple[ChatAgent, StoreRepository, ConversationService]:
    store = StoreRepository(connection)
    conversations = ConversationService(connection, store)
    agent = ChatAgent(model_client, policies, store, conversations)
    return agent, store, conversations


def _system_message(calls: list[tuple[list[ChatMessage], list[AgentTool]]]) -> str:
    messages, _tools = calls[0]
    assert messages[0].role == "system"
    return messages[0].content


def test_policy_question_uses_retrieved_chunks_and_answers_in_english(
    imported_connection: sqlite3.Connection,
) -> None:
    chunk = PolicyChunk(
        text="Returns are accepted within 30 days of delivery with a receipt.",
        source="policies.pdf",
    )
    policies = FakePolicyIndex(chunks=[chunk])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content="You can return this item within 30 days of delivery with a receipt.",
                tool_calls=[],
            )
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    reply = agent.respond(conversation.id, "What is your return policy?")

    assert reply == "You can return this item within 30 days of delivery with a receipt."
    assert policies.queries == ["What is your return policy?"]
    system_prompt = _system_message(model.calls)
    assert chunk.text in system_prompt
    assert reply.isascii()
    history = conversations.get_history(conversation.id)
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[-1].content == reply


def test_system_prompt_establishes_store_persona_and_tone(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content="We're open Monday to Saturday -- happy to help!",
                tool_calls=[],
            )
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    agent.respond(conversation.id, "What are your store hours?")

    system_prompt = _system_message(model.calls)
    assert "Empório da Música" in system_prompt
    assert "Campo Grande" in system_prompt
    assert "founded in 2008" in system_prompt
    assert "does not sell accessories" in system_prompt


def test_system_prompt_instructs_agent_to_decline_off_topic_questions(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content="I can only help with questions about Empório da Música.",
                tool_calls=[],
            )
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    agent.respond(conversation.id, "What's the capital of France?")

    system_prompt = _system_message(model.calls)
    assert "Only discuss topics related to Empório da Música" in system_prompt
    assert "politely decline" in system_prompt


def test_off_topic_decline_reply_is_persisted_unmodified(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    decline_text = (
        "I can only help with questions about Empório da Música -- "
        "our instruments, orders, and policies."
    )
    model = ScriptedChatClient(
        [ModelResponse(content=decline_text, tool_calls=[])]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    reply = agent.respond(conversation.id, "What's the capital of France?")

    assert reply == decline_text
    history = conversations.get_history(conversation.id)
    assert history[-1].role == "assistant"
    assert history[-1].content == decline_text


def test_indirectly_phrased_in_scope_question_still_reaches_product_tool(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        name="search_products",
                        arguments={"query": ""},
                        call_id="call-1",
                    )
                ],
            ),
            ModelResponse(
                content="Here are a few options under R$1000.",
                tool_calls=[],
            ),
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    reply = agent.respond(conversation.id, "What do you have under $1000?")

    assert reply == "Here are a few options under R$1000."
    first_call_messages, first_call_tools = model.calls[0]
    assert any(tool.name == "search_products" for tool in first_call_tools)
    assert len(model.calls) == 2


def test_missing_policy_context_does_not_claim_a_store_policy(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content="I don't have that information about our policies.",
                tool_calls=[],
            )
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    reply = agent.respond(conversation.id, "Can I return a used amplifier after a year?")

    system_prompt = _system_message(model.calls)
    # No fabricated policy text is injected when retrieval is empty.
    assert "30 days" not in system_prompt
    assert "none found" in system_prompt.lower() or "no relevant" in system_prompt.lower()
    assert reply == "I don't have that information about our policies."


def test_product_question_uses_store_repository_tool(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        name="search_products",
                        arguments={"query": "Stratocaster"},
                        call_id="call-1",
                    )
                ],
            ),
            ModelResponse(
                content="We have the Stratocaster in stock for R$999.90.",
                tool_calls=[],
            ),
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    reply = agent.respond(conversation.id, "Do you have a Stratocaster?")

    assert reply == "We have the Stratocaster in stock for R$999.90."
    assert len(model.calls) == 2
    second_call_messages, _tools = model.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call-1"
    assert tool_messages[0].name == "search_products"
    results = json.loads(tool_messages[0].content)
    assert results[0]["name"] == "Stratocaster"
    assert results[0]["price_brl"] == 999.9
    assert results[0]["stock_quantity"] == 5
    assert results[0]["status"] == "active"

    # The history sent on the second round must show the standard
    # assistant-requested-tool-calls -> tool-result transcript shape: an
    # assistant turn recording the request immediately followed by the tool
    # result answering it, not just the tool result on its own.
    assistant_tool_turn_index = next(
        i for i, m in enumerate(second_call_messages) if m.role == "assistant" and m.tool_calls
    )
    assistant_tool_turn = second_call_messages[assistant_tool_turn_index]
    assert assistant_tool_turn.tool_calls == [
        ToolCall(name="search_products", arguments={"query": "Stratocaster"}, call_id="call-1")
    ]
    assert second_call_messages[assistant_tool_turn_index + 1] is tool_messages[0]


def test_product_question_without_stock_data_does_not_claim_availability(
    imported_connection: sqlite3.Connection,
) -> None:
    """The agent may report availability, but only exactly what the product-search
    tool returned -- never a qualifier or number the tool result didn't contain.

    Reframed per the plan-vs-real-data ruling: products.csv has real
    stock_quantity/status columns (confirmed complete in Task 2), so claiming
    availability from a tool result is correct behavior now. The remaining
    safety property is narrower: the agent must never embellish beyond the
    tool's own fields. This test uses the fixture's Precision Bass, which has
    stock_quantity=0 and status="discontinued", and asserts (a) the tool
    result handed back to the model contains exactly those real values with
    nothing invented or added, and (b) the system prompt instructs the model
    to never claim availability beyond a tool result.
    """
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        name="search_products",
                        arguments={"query": "Precision Bass"},
                        call_id="call-1",
                    )
                ],
            ),
            ModelResponse(
                content="The Precision Bass is discontinued with 0 units in stock.",
                tool_calls=[],
            ),
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    reply = agent.respond(conversation.id, "Is the Precision Bass in stock?")

    second_call_messages, _tools = model.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    result = json.loads(tool_messages[0].content)[0]
    assert result["stock_quantity"] == 0
    assert result["status"] == "discontinued"
    assert set(result.keys()) == {
        "product_id",
        "name",
        "description",
        "price_brl",
        "category",
        "stock_quantity",
        "status",
    }

    system_prompt = _system_message(model.calls)
    assert "availability" in system_prompt.lower()
    assert reply == "The Precision Bass is discontinued with 0 units in stock."


def test_order_verification_requires_both_values(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        name="verify_customer",
                        arguments={"order_number": "1"},
                        call_id="call-1",
                    )
                ],
            )
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    with pytest.raises(InvalidToolCallError):
        agent.respond(conversation.id, "My order number is 1, verify me.")


def test_failed_verification_uses_generic_response(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        name="verify_customer",
                        arguments={"order_number": "1", "email": "wrong@example.com"},
                        call_id="call-1",
                    )
                ],
            ),
            ModelResponse(
                content="I couldn't verify your identity with that order number and email.",
                tool_calls=[],
            ),
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    reply = agent.respond(conversation.id, "Verify order 1, email wrong@example.com")

    second_call_messages, _tools = model.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    result = json.loads(tool_messages[0].content)
    # The result must not reveal which half of the pair was wrong.
    assert result == {"verified": False}
    assert reply == "I couldn't verify your identity with that order number and email."
    unchanged = conversations.get_conversation(conversation.id)
    assert unchanged is not None
    assert unchanged.customer_id is None


def test_order_tool_uses_verified_conversation_scope(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])

    # Part 1: a model-supplied customer id in the tool arguments must be
    # structurally rejected, not silently ignored or honored -- it is not a
    # declared parameter of the order-listing tool.
    smuggling_model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        name="list_verified_orders",
                        arguments={"customer_id": "2"},
                        call_id="call-1",
                    )
                ],
            )
        ]
    )
    agent, _store, conversations = _make_agent(smuggling_model, policies, imported_connection)
    conversation = conversations.create_conversation()
    conversations.verify_customer(conversation.id, "1", "alice@example.com")

    with pytest.raises(InvalidToolCallError):
        agent.respond(conversation.id, "Show me customer 2's orders")

    # Part 2: with no arguments, the tool must return exactly the verified
    # conversation's own orders (Alice's order 1), never another customer's.
    honest_model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(name="list_verified_orders", arguments={}, call_id="call-1")
                ],
            ),
            ModelResponse(content="Your order 1 was delivered.", tool_calls=[]),
        ]
    )
    agent2, _store2, conversations2 = _make_agent(honest_model, policies, imported_connection)
    reply = agent2.respond(conversation.id, "Show me my orders")

    second_call_messages, _tools = honest_model.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    orders = json.loads(tool_messages[0].content)
    assert {o["customer_id"] for o in orders} == {1}
    assert 2 not in {o["customer_id"] for o in orders}
    assert reply == "Your order 1 was delivered."


def test_agent_rejects_undeclared_tool(imported_connection: sqlite3.Connection) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(name="delete_order", arguments={"order_id": 1}, call_id="call-1")
                ],
            )
        ]
    )
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    with pytest.raises(InvalidToolCallError):
        agent.respond(conversation.id, "Delete order 1")


def test_agent_surfaces_unavailable_model(imported_connection: sqlite3.Connection) -> None:
    policies = FakePolicyIndex(chunks=[])
    # Per the LocalChatClient contract, an implementation raises
    # ModelUnavailableError itself (OllamaChatClient does this by catching
    # the underlying ollama client's errors); this fake honors that contract
    # so the test exercises respond()'s propagation, not translation.
    model = FailingChatClient(ModelUnavailableError("local model host is unreachable"))
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    with pytest.raises(ModelUnavailableError):
        agent.respond(conversation.id, "Hello")

    # The user message is still persisted; no guessed assistant reply is.
    history = conversations.get_history(conversation.id)
    assert [m.role for m in history] == ["user"]


def test_respond_persists_user_message_before_dispatch(
    imported_connection: sqlite3.Connection,
) -> None:
    policies = FakePolicyIndex(chunks=[])
    model = ScriptedChatClient([ModelResponse(content="Hi there!", tool_calls=[])])
    agent, _store, conversations = _make_agent(model, policies, imported_connection)
    conversation = conversations.create_conversation()

    agent.respond(conversation.id, "Hello")

    history = conversations.get_history(conversation.id)
    assert [m.content for m in history] == ["Hello", "Hi there!"]


def test_ollama_message_conversion_uses_tool_name_not_call_id() -> None:
    """A role='tool' message's wire-format 'tool_name' must be the real tool
    name (ChatMessage.name), never the call id (ChatMessage.tool_call_id) --
    the two are different pieces of information and must never be conflated.
    """
    message = ChatMessage(
        role="tool",
        content='{"verified": false}',
        tool_call_id="call-1",
        name="verify_customer",
    )

    payload = _to_ollama_message(message)

    assert payload["tool_name"] == "verify_customer"
    assert payload["tool_name"] != "call-1"


def test_ollama_message_conversion_carries_assistant_tool_calls() -> None:
    """An assistant turn that requested tool calls must round-trip those
    calls into Ollama's own tool_calls wire shape, so a resumed conversation
    sent to a real Ollama server shows the standard assistant-requested-
    tool-calls transcript rather than silently dropping the request.
    """
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(name="search_products", arguments={"query": "bass"}, call_id="call-1")
        ],
    )

    payload = _to_ollama_message(message)

    assert payload["tool_calls"] == [
        {"function": {"name": "search_products", "arguments": {"query": "bass"}}}
    ]


def test_ollama_message_conversion_plain_assistant_message_has_no_tool_calls_key() -> None:
    """A plain final-answer assistant message (no tool calls) must not carry
    a spurious empty tool_calls key in the wire format."""
    message = ChatMessage(role="assistant", content="Hello there.")

    payload = _to_ollama_message(message)

    assert "tool_calls" not in payload
