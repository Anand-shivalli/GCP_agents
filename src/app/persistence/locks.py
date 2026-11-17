"""Per-session mutual exclusion across replicas.

Two requests for the same session can arrive at two different instances
(double-click, client retry, load balancer). Running both would fork the
conversation. A Postgres advisory lock keyed on session_id guarantees one
active run per session cluster-wide, with no extra infrastructure.

The lock is session-scoped to the DB connection, so if the instance crashes
the connection drops and Postgres releases the lock automatically.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool


class SessionBusyError(Exception):
    """Another run for this session is in progress (maybe on another replica)."""


@asynccontextmanager
async def session_lock(pool: AsyncConnectionPool, session_id: str) -> AsyncIterator[None]:
    # NOTE: holds one pooled connection for the duration of the run; size the
    # pool for (concurrent runs + checkpointer traffic).
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS locked", (session_id,)
        )
        row = await cur.fetchone()
        if not row or not row["locked"]:
            raise SessionBusyError(session_id)
        try:
            yield
        finally:
            await conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (session_id,))
