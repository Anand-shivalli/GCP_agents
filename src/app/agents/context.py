"""Per-request runtime context.

Passed as `context=` on every graph call. Nodes, middleware and tools read it
via `runtime.context`. It is NOT part of graph state, so it is never written
to checkpoints and cannot be changed by the model - which is exactly what you
want for identity (who is calling) as opposed to conversation data.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    session_id: str
    tenant_id: str | None = None
