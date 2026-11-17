"""Postgres integration: real checkpointer + cross-replica session lock.

Runs only when TEST_DATABASE_URL is set, e.g.
    docker compose up -d postgres
    TEST_DATABASE_URL=postgresql://app:app@localhost:5432/app pytest tests/test_postgres.py
"""

import asyncio
import os
import sys
from functools import partial

import pytest
from pydantic import SecretStr

from app.agents.context import RequestContext
from app.core.config import Settings
from app.graph.factory import build_application_graph
from app.persistence.checkpointer import build_checkpointer, build_pool
from app.persistence.locks import SessionBusyError, session_lock
from app.services.runs import RunService
from app.tools.mcp import MCPAdapter
from mcp_servers.customer_server import mcp as customer_server
from tests.test_workflow import _models

DB_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_two_replicas_share_state_and_lock():
    settings = Settings(database_url=SecretStr(DB_URL))
    tools = await MCPAdapter(customer_server).list_tools()
    # Two pools + two graphs = two "replicas" talking to one database.
    async with build_pool(settings) as pool_a, build_pool(settings) as pool_b:
        saver_a = build_checkpointer(pool_a, settings)
        saver_b = build_checkpointer(pool_b, settings)
        await saver_a.setup()
        replica_a = RunService(
            build_application_graph(
                tools=tools, checkpointer=saver_a, settings=settings, models=_models()
            ),
            partial(session_lock, pool_a),
            settings,
        )
        replica_b = RunService(
            build_application_graph(
                tools=tools, checkpointer=saver_b, settings=settings, models=_models()
            ),
            partial(session_lock, pool_b),
            settings,
        )
        ctx = RequestContext(user_id="alice", session_id=f"pg-{os.getpid()}-{id(pool_a)}")

        await replica_a.send_message(ctx, "status of C-1001?")
        await replica_b.send_message(ctx, "and claims?")  # follow-up lands elsewhere
        _, messages = await replica_a.get_session(ctx)
        assert [m["role"] for m in messages].count("human") == 2

        async with session_lock(pool_a, ctx.session_id):
            with pytest.raises(SessionBusyError):
                async with session_lock(pool_b, ctx.session_id):
                    pass
