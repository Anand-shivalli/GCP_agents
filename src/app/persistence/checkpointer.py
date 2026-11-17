"""Postgres connection pool and LangGraph checkpointer.

The checkpointer stores a snapshot of graph state after every step, keyed by
thread_id (= our session_id). That is what makes the service stateless:
  * a follow-up message can land on any replica and continue the conversation;
  * a run that died mid-way (pod killed, LLM outage) resumes from its last
    completed step instead of starting over.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings


def build_pool(settings: Settings) -> AsyncConnectionPool:
    """One pool per process, shared by the checkpointer and session locks."""
    return AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        max_size=settings.db_pool_max_size,
        open=False,  # opened explicitly in the app lifespan
        kwargs={
            "autocommit": True,  # required by AsyncPostgresSaver
            "row_factory": dict_row,  # required by AsyncPostgresSaver
            "prepare_threshold": None,  # no server-side prepared stmts: safe behind PgBouncer
        },
    )


def build_checkpointer(pool: AsyncConnectionPool, settings: Settings) -> AsyncPostgresSaver:
    # Checkpoints hold raw user input and tool output (possibly PHI) - encrypt in prod.
    serde = EncryptedSerializer.from_pycryptodome_aes() if settings.checkpoint_encryption else None
    return AsyncPostgresSaver(pool, serde=serde)
