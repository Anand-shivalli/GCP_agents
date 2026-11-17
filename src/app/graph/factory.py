"""Composition root: builds workers -> supervisor -> orchestrator.

Everything is passed in explicitly (tools, checkpointer, models), so the same
function serves the API, tests (fakes + in-memory saver) and LangGraph Studio.
"""

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.agents.supervisor import build_supervisor
from app.agents.workers import WORKERS, build_worker
from app.core.config import Settings
from app.graph.orchestrator import build_orchestrator
from app.llm.client import ModelSet


def build_application_graph(
    *,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    settings: Settings,
    models: ModelSet,
) -> CompiledStateGraph:
    workers = {spec: build_worker(spec, tools, settings, models) for spec in WORKERS}
    supervisor = build_supervisor(workers, settings, models)
    return build_orchestrator(supervisor, checkpointer, settings)
