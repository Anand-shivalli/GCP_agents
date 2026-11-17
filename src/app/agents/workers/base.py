"""What every worker agent shares: the spec type and the one factory that builds them.

A worker folder only *describes* an agent (a WorkerSpec). It never builds one
itself, so every worker gets the same middleware, identity and persistence
behaviour.
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.agents.context import RequestContext
from app.agents.middleware import worker_middleware
from app.core.config import Settings
from app.llm.client import ModelSet


@dataclass(frozen=True)
class WorkerSpec:
    name: str  # becomes the supervisor's tool name - snake_case, unique
    description: str  # the supervisor routes on this text - write it carefully
    system_prompt: str
    tool_names: frozenset[str] | None = None  # MCP tools it may use; None = all (avoid)


def build_worker(
    spec: WorkerSpec, tools: list[BaseTool], settings: Settings, models: ModelSet
) -> CompiledStateGraph:
    allowed = [t for t in tools if spec.tool_names is None or t.name in spec.tool_names]
    # No checkpointer here: subagents inherit the orchestrator's checkpointer at
    # runtime (official guidance). Giving them their own breaks interrupts/resume.
    return create_agent(
        model=models.worker,
        tools=allowed,
        system_prompt=spec.system_prompt,
        middleware=worker_middleware(settings, models),
        context_schema=RequestContext,
        name=spec.name,
    )
