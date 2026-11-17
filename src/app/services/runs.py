"""Run service: the only code that calls the graph. The API layer stays thin.

Every call follows the same steps:
  1. take the cluster-wide session lock (one active run per session);
  2. check the caller owns the session (user_id stored in checkpoint metadata);
  3. run the graph with thread_id=session_id and durability="sync";
  4. read the result back from the checkpoint - the source of truth.

Each run is wrapped in a `session.<operation>` span (child of the HTTP span)
and recorded in the app.runs / app.run.duration metrics.

Session states (derived from the latest checkpoint, never from memory):
  completed    - last run finished; ready for a new message
  interrupted  - paused for human input (HITL); call resume with a value
  incomplete   - a run died mid-way (crash, deploy, outage); call resume with no value
"""

import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from opentelemetry import trace

from app.agents.context import RequestContext
from app.core.config import Settings
from app.core.telemetry import lock_rejections, run_duration, runs_total, tracer
from app.persistence.locks import SessionBusyError

SessionStatus = Literal["new", "completed", "interrupted", "incomplete"]
LockFactory = Callable[[str], AbstractAsyncContextManager[None]]


class SessionNotFoundError(Exception):
    """Unknown session, or owned by another user (indistinguishable on purpose)."""


class SessionStateError(Exception):
    """The requested action is not valid for the session's current status."""


@dataclass
class RunResult:
    session_id: str
    run_id: str
    status: SessionStatus
    output: str | None = None
    interrupts: list[Any] = field(default_factory=list)


class RunService:
    def __init__(self, graph: CompiledStateGraph, lock: LockFactory, settings: Settings):
        self._graph = graph
        self._lock = lock
        self._settings = settings

    # ---- public API ---------------------------------------------------------------

    async def send_message(self, ctx: RequestContext, message: str) -> RunResult:
        run_id = _new_run_id()
        async with self._observe("send_message", ctx, run_id) as outcome, self._locked(ctx):
            snapshot = await self._owned_snapshot(ctx, allow_new=True)
            self._require(snapshot, {"new", "completed"}, action="send a message")
            await self._graph.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                self._config(ctx, run_id),
                context=ctx,
                durability="sync",  # checkpoint is committed before the next step starts
            )
            result = await self._result(ctx, run_id)
            outcome["status"] = result.status
            return result

    async def resume(self, ctx: RequestContext, value: Any | None = None) -> RunResult:
        """Continue from the last checkpoint.

        value=None   -> retry a run that died ("incomplete").
        value=<json> -> answer a human-in-the-loop interrupt ("interrupted").
        """
        run_id = _new_run_id()
        async with self._observe("resume", ctx, run_id) as outcome, self._locked(ctx):
            snapshot = await self._owned_snapshot(ctx)
            expected = "interrupted" if value is not None else "incomplete"
            self._require(snapshot, {expected}, action="resume")
            graph_input = Command(resume=value) if value is not None else None
            await self._graph.ainvoke(
                graph_input, self._config(ctx, run_id), context=ctx, durability="sync"
            )
            result = await self._result(ctx, run_id)
            outcome["status"] = result.status
            return result

    async def stream_message(
        self, ctx: RequestContext, message: str
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yield ("token", {...}) events while running, then ("result", {...})."""
        run_id = _new_run_id()
        async with self._observe("stream_message", ctx, run_id) as outcome, self._locked(ctx):
            snapshot = await self._owned_snapshot(ctx, allow_new=True)
            self._require(snapshot, {"new", "completed"}, action="send a message")
            async for chunk, meta in self._graph.astream(
                {"messages": [HumanMessage(content=message)]},
                self._config(ctx, run_id),
                context=ctx,
                stream_mode="messages",
                durability="sync",
            ):
                # Only stream the supervisor's tokens to the user; worker tokens are internal.
                if isinstance(chunk, AIMessageChunk) and chunk.text and _from_supervisor(meta):
                    yield "token", {"text": chunk.text}
            result = await self._result(ctx, run_id)
            outcome["status"] = result.status
            yield "result", result.__dict__

    async def get_session(self, ctx: RequestContext) -> tuple[SessionStatus, list[dict]]:
        snapshot = await self._owned_snapshot(ctx)
        messages = [
            {"role": m.type, "content": m.text}
            for m in snapshot.values.get("messages", [])
            if m.type in ("human", "ai") and m.text
        ]
        return _status(snapshot), messages

    # ---- internals -------------------------------------------------------------------

    @asynccontextmanager
    async def _observe(
        self, operation: str, ctx: RequestContext, run_id: str
    ) -> AsyncIterator[dict[str, str]]:
        """Span + metrics around one run. The caller sets outcome["status"] on success."""
        ids = {"user.id": ctx.user_id, "session.id": ctx.session_id, "run.id": run_id}
        trace.get_current_span().set_attributes(ids)  # the HTTP span, for filtering by user
        outcome = {"status": "error"}
        started = time.perf_counter()
        with tracer.start_as_current_span(f"session.{operation}", attributes=ids) as span:
            try:
                yield outcome
            except (SessionBusyError, SessionNotFoundError, SessionStateError):
                outcome["status"] = "rejected"  # client-side outcome, not a server error
                raise
            finally:
                span.set_attribute("session.status", outcome["status"])
                # Low-cardinality labels only: never put user/session IDs on metrics.
                labels = {"operation": operation, "status": outcome["status"]}
                runs_total.add(1, labels)
                run_duration.record(time.perf_counter() - started, labels)

    @asynccontextmanager
    async def _locked(self, ctx: RequestContext) -> AsyncIterator[None]:
        """Hold the session lock; the acquire step gets its own span."""
        async with AsyncExitStack() as stack:
            with tracer.start_as_current_span("session.lock.acquire"):
                try:
                    await stack.enter_async_context(self._lock(ctx.session_id))
                except SessionBusyError:
                    lock_rejections.add(1)
                    raise
            yield

    def _config(self, ctx: RequestContext, run_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": ctx.session_id},
            # Metadata is stored on every checkpoint (ownership check below) and
            # attached to every trace (LangSmith/OTel filtering by user/session).
            "metadata": {"user_id": ctx.user_id, "session_id": ctx.session_id, "run_id": run_id},
            "run_id": uuid.UUID(run_id),
            "recursion_limit": self._settings.graph_recursion_limit,
        }

    async def _owned_snapshot(self, ctx: RequestContext, allow_new: bool = False) -> StateSnapshot:
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": ctx.session_id}})
        if snapshot.created_at is None:  # no checkpoint yet
            if allow_new:
                return snapshot
            raise SessionNotFoundError(ctx.session_id)
        if (snapshot.metadata or {}).get("user_id") != ctx.user_id:
            raise SessionNotFoundError(ctx.session_id)
        return snapshot

    @staticmethod
    def _require(snapshot: StateSnapshot, allowed: set[str], *, action: str) -> None:
        status = _status(snapshot)
        if status not in allowed:
            raise SessionStateError(f"cannot {action}: session is {status}")

    async def _result(self, ctx: RequestContext, run_id: str) -> RunResult:
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": ctx.session_id}})
        return RunResult(
            session_id=ctx.session_id,
            run_id=run_id,
            status=_status(snapshot),
            output=snapshot.values.get("final_answer") if not snapshot.next else None,
            interrupts=[i.value for i in snapshot.interrupts],
        )


def _new_run_id() -> str:
    """Run ID = the request's OTel trace ID when tracing is on (both are 128 bits).

    The run_id the API returns is then exactly the trace ID to search for in any
    OTel backend, and it is also the root run ID in LangSmith - one ID everywhere.
    """
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        return str(uuid.UUID(int=span_context.trace_id))
    return str(uuid.uuid4())


def _status(snapshot: StateSnapshot) -> SessionStatus:
    if snapshot.created_at is None:
        return "new"
    if snapshot.interrupts:
        return "interrupted"
    if snapshot.next:
        return "incomplete"
    return "completed"


def _from_supervisor(meta: dict) -> bool:
    return meta.get("lc_agent_name", "supervisor") == "supervisor"
