"""Agent 2: claims. Lists and explains a customer's open insurance claims."""

from app.agents.prompts import CLAIMS_AGENT_PROMPT
from app.agents.workers.base import WorkerSpec

SPEC = WorkerSpec(
    name="claims_agent",
    description=(
        "Lists a customer's open insurance claims with status and amount. "
        "Input: a complete task in plain language, including the customer ID (e.g. C-1001)."
    ),
    system_prompt=CLAIMS_AGENT_PROMPT,
    tool_names=frozenset({"list_open_claims"}),
)
