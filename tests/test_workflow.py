"""End-to-end workflow tests: real graph, real MCP (in-process), real middleware.

Only the LLMs are fake and Postgres is replaced by InMemorySaver - the
RunService/graph code paths are the same ones production runs.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.context import RequestContext
from app.core.config import Settings
from app.graph.factory import build_application_graph
from app.llm.client import ModelSet
from app.services.runs import RunService, SessionNotFoundError, SessionStateError
from app.tools.mcp import MCPAdapter
from mcp_servers.customer_server import mcp as customer_server
from tests.fakes import PolicyChatModel, tool_then_answer


@asynccontextmanager
async def _local_lock(_session_id: str):
    # Stand-in for the Postgres advisory lock (covered by the integration test).
    yield


def _models() -> ModelSet:
    supervisor = PolicyChatModel(
        policy=tool_then_answer(
            "customer_records_agent",
            {"task": "Get profile for customer C-1001"},
            answer=lambda worker_reply: f"Here is what I found: {worker_reply}",
        )
    )
    worker = PolicyChatModel(
        policy=tool_then_answer(
            "get_customer", {"customer_id": "C-1001"}, answer=lambda tool_out: tool_out
        )
    )
    return ModelSet(supervisor=supervisor, worker=worker, summarizer=worker)


def _service(tools) -> RunService:
    settings = Settings()
    graph = build_application_graph(
        tools=tools, checkpointer=InMemorySaver(), settings=settings, models=_models()
    )
    return RunService(graph, _local_lock, settings)


async def test_orchestrator_supervisor_worker_mcp_with_pii_redaction():
    tools = await MCPAdapter(customer_server).list_tools()
    runs = _service(tools)
    alice = RequestContext(user_id="alice", session_id="s-1")

    result = await runs.send_message(alice, "Hi, I'm Asha Rao (asha.rao@example.com). Status?")

    assert result.status == "completed"
    # Worker fetched the record over MCP; PIIMiddleware redacted it on the way back.
    assert "P-77" in result.output and "active" in result.output
    assert "Asha Rao" not in result.output
    assert "asha.rao@example.com" not in result.output
    assert "555 0134" not in result.output

    # The persisted user message is redacted too.
    status, messages = await runs.get_session(alice)
    assert status == "completed"
    assert "Asha Rao" not in messages[0]["content"]
    assert "[REDACTED_EMAIL]" in messages[0]["content"]

    # Follow-up on the same session continues the same thread.
    await runs.send_message(alice, "Thanks, and any open claims?")
    _, messages = await runs.get_session(alice)
    assert [m["role"] for m in messages].count("human") == 2


async def test_session_is_bound_to_its_user():
    runs = _service(await MCPAdapter(customer_server).list_tools())
    await runs.send_message(RequestContext(user_id="alice", session_id="s-2"), "hello C-1001")

    with pytest.raises(SessionNotFoundError):
        await runs.get_session(RequestContext(user_id="mallory", session_id="s-2"))
    with pytest.raises(SessionNotFoundError):
        await runs.send_message(RequestContext(user_id="mallory", session_id="s-2"), "hi")


async def test_crashed_run_resumes_from_last_checkpoint():
    """Simulate a pod dying mid-tool-call, then resume on 'another instance'."""
    started, calls = asyncio.Event(), {"n": 0}

    @tool
    async def get_customer(customer_id: str) -> str:
        """Fetch a customer."""
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            await asyncio.sleep(30)  # never finishes: the task is cancelled
        return "policy P-77 active"

    runs = _service([get_customer])
    ctx = RequestContext(user_id="alice", session_id="s-3")

    task = asyncio.create_task(runs.send_message(ctx, "status for C-1001?"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    status, _ = await runs.get_session(ctx)
    assert status == "incomplete"
    with pytest.raises(SessionStateError):  # can't start a new turn over a broken one
        await runs.send_message(ctx, "hello?")

    result = await runs.resume(ctx)
    assert result.status == "completed"
    assert "P-77" in result.output


async def test_guard_rejects_oversized_input():
    runs = _service([])
    result = await runs.send_message(RequestContext("alice", "s-4"), "x" * 9000)
    assert result.status == "completed"
    assert "under 8000 characters" in result.output


async def test_human_approval_interrupt_propagates_and_resumes():
    """HITL inside the worker pauses the whole orchestrator; resume continues it."""
    settings = Settings(hitl_tools=["get_customer"])
    graph = build_application_graph(
        tools=await MCPAdapter(customer_server).list_tools(),
        checkpointer=InMemorySaver(),
        settings=settings,
        models=_models(),
    )
    runs = RunService(graph, _local_lock, settings)
    ctx = RequestContext(user_id="alice", session_id="s-5")

    paused = await runs.send_message(ctx, "status for C-1001?")
    assert paused.status == "interrupted"
    assert paused.interrupts[0]["action_requests"][0]["name"] == "get_customer"

    done = await runs.resume(ctx, {"decisions": [{"type": "approve"}]})
    assert done.status == "completed"
    assert "P-77" in done.output
