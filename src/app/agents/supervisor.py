"""Supervisor: a `create_agent` whose tools are the worker agents.

This is the current LangChain pattern ("subagents as tools"). The old
`langgraph-supervisor` package (`create_supervisor` + handoff tools) is no
longer maintained - see docs.langchain.com/oss/python/migrate/langgraph-supervisor.

Workers get an isolated context: only the task string, not the chat history.
That keeps their prompts small and stops PII in history leaking into tools.
"""

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.agents.context import RequestContext
from app.agents.middleware import supervisor_middleware
from app.agents.prompts import SUPERVISOR_PROMPT
from app.agents.workers import WorkerSpec
from app.core.config import Settings
from app.llm.client import ModelSet


def worker_as_tool(spec: WorkerSpec, worker: CompiledStateGraph) -> BaseTool:
    @tool(spec.name, description=spec.description)
    async def delegate(task: str, runtime: ToolRuntime[RequestContext]) -> str:
        # `runtime` is injected by LangChain and hidden from the model's schema.
        # Forward the caller's identity so worker tools can enforce access.
        result = await worker.ainvoke(
            {"messages": [HumanMessage(content=task)]}, context=runtime.context
        )
        return result["messages"][-1].text

    return delegate


def build_supervisor(
    workers: dict[WorkerSpec, CompiledStateGraph], settings: Settings, models: ModelSet
) -> CompiledStateGraph:
    return create_agent(
        model=models.supervisor,
        tools=[worker_as_tool(spec, graph) for spec, graph in workers.items()],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=supervisor_middleware(settings, models),
        context_schema=RequestContext,
        name="supervisor",
    )
