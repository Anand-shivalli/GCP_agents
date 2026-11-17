"""Orchestrator: the outer, deterministic workflow.

    START -> guard_input -> supervisor -> finalize -> END
                  \\-> (rejected) -> END

Why an orchestrator around the supervisor agent?
  * Deterministic steps (validation, policy, audit, writes to other systems)
    belong in plain graph nodes, not in an LLM's judgement.
  * It is the ONLY graph compiled with a checkpointer. Supervisor and workers
    inherit it, so one thread_id captures the whole run and resume works
    across all three levels.
"""

from typing import Literal, NotRequired

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.context import RequestContext
from app.core.config import Settings


class OrchestratorState(MessagesState):
    final_answer: NotRequired[str | None]


def make_guard_input(settings: Settings):
    async def guard_input(
        state: OrchestratorState, runtime: Runtime[RequestContext]
    ) -> Command[Literal["supervisor", "__end__"]]:
        last = state["messages"][-1]
        text = last.text if isinstance(last, HumanMessage) else ""
        if not text.strip() or len(text) > settings.max_input_chars:
            reply = f"Please send a non-empty message under {settings.max_input_chars} characters."
            return Command(goto=END, update={"messages": [AIMessage(reply)], "final_answer": reply})
        return Command(goto="supervisor")

    return guard_input


async def finalize(state: OrchestratorState) -> dict:
    # Extension point: audit events, outbound notifications, structured output.
    return {"final_answer": state["messages"][-1].text}


def build_orchestrator(
    supervisor: CompiledStateGraph, checkpointer: BaseCheckpointSaver, settings: Settings
) -> CompiledStateGraph:
    builder = StateGraph(OrchestratorState, context_schema=RequestContext)
    builder.add_node("guard_input", make_guard_input(settings))
    # A compiled agent is a valid node: it shares the `messages` channel.
    builder.add_node("supervisor", supervisor)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "guard_input")
    builder.add_edge("supervisor", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="orchestrator")
