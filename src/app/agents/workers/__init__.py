"""Worker registry. One folder per worker agent; this file lists which ones are live.

To add a worker:
  1. Copy `agent_1/` to `agent_<n>/` (or a descriptive name).
  2. Add its system prompt to `app/agents/prompts.py`, then edit the `SPEC` in
     its `agent.py` (name, description, prompt, allowed MCP tools).
  3. Import its SPEC below and add it to WORKERS.

The supervisor gets one tool per entry automatically - no graph wiring to change.
The list is explicit on purpose: you can see every live agent in one place, and
an unfinished folder can't go live by accident.
"""

from app.agents.workers.agent_1 import SPEC as AGENT_1
from app.agents.workers.agent_2 import SPEC as AGENT_2
from app.agents.workers.base import WorkerSpec, build_worker

WORKERS: tuple[WorkerSpec, ...] = (AGENT_1, AGENT_2)

__all__ = ["WORKERS", "WorkerSpec", "build_worker"]
