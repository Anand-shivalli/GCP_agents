"""FastAPI application factory and lifespan.

Startup order matters:
  telemetry -> DB pool -> checkpointer -> PII model -> MCP tools -> graph.
Everything created here is per-process and stateless (pools, clients, the
compiled graph). Per-session state is in Postgres.

Run:  uvicorn app.main:app        (Linux/containers)
      python -m app               (Windows dev - sets the event loop psycopg needs)
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.pii import warm_up as warm_up_pii
from app.api.v1.routes import router as v1_router
from app.core.config import get_settings
from app.core.telemetry import (
    configure_logging,
    instrument_app,
    setup_telemetry,
    shutdown_telemetry,
)
from app.graph.factory import build_application_graph
from app.llm.client import ModelSet
from app.persistence.checkpointer import build_checkpointer, build_pool
from app.persistence.locks import SessionBusyError, session_lock
from app.services.runs import RunService, SessionNotFoundError, SessionStateError
from app.tools.mcp import mcp_tools


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    setup_telemetry(settings)  # before any LangChain run, so LangSmith adopts our provider

    async with build_pool(settings) as pool:
        checkpointer = build_checkpointer(pool, settings)
        if settings.db_run_setup_on_startup:
            await checkpointer.setup()
        await asyncio.to_thread(warm_up_pii)

        async with mcp_tools(settings.mcp_config_path) as tools:
            graph = build_application_graph(
                tools=tools,
                checkpointer=checkpointer,
                settings=settings,
                models=ModelSet.from_settings(settings),
            )
            app.state.pool = pool
            app.state.run_service = RunService(graph, partial(session_lock, pool), settings)
            yield

    shutdown_telemetry()


def create_app() -> FastAPI:
    app = FastAPI(title="LangGraph Starter", version="0.1.0", lifespan=lifespan)
    app.include_router(v1_router)
    instrument_app(app)

    # Domain errors -> stable HTTP contract.
    @app.exception_handler(SessionNotFoundError)
    async def _not_found(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            {"code": "session_not_found", "detail": "Session not found"}, status_code=404
        )

    @app.exception_handler(SessionBusyError)
    async def _busy(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            {"code": "session_busy", "detail": "A run is already in progress for this session"},
            status_code=409,
            headers={"Retry-After": "2"},
        )

    @app.exception_handler(SessionStateError)
    async def _state(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"code": "invalid_session_state", "detail": str(exc)}, status_code=409)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:  # liveness: the process is up
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:  # readiness: dependencies reachable
        try:
            async with request.app.state.pool.connection() as conn:
                await conn.execute("SELECT 1")
        except Exception:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ready"})

    return app


app = create_app()
