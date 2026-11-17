"""Deterministic fake chat model for offline tests.

Responses are a function of the conversation (not a fixed sequence), so a run
that is cancelled and resumed from a checkpoint still behaves deterministically.
"""

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

Policy = Callable[[list[BaseMessage]], AIMessage]


class PolicyChatModel(BaseChatModel):
    policy: Policy
    seen: list[list[BaseMessage]] = []  # every prompt the model received

    @property
    def _llm_type(self) -> str:
        return "policy-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "PolicyChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.policy(messages))])


def call_tool(name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}]
    )


def tool_then_answer(name: str, args: dict, answer: Callable[[str], str]) -> Policy:
    """Call `name` once; after its result arrives, answer using the tool output."""

    def policy(messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if last.type == "tool":
            return AIMessage(content=answer(last.text))
        return call_tool(name, **args)

    return policy
