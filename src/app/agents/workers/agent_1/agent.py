"""Agent 1: customer records. Looks up profiles and policy status."""

from app.agents.prompts import CUSTOMER_RECORDS_AGENT_PROMPT
from app.agents.workers.base import WorkerSpec

SPEC = WorkerSpec(
    name="customer_records_agent",
    description=(
        "Looks up a customer's profile and policy status. "
        "Input: a complete task in plain language, including the customer ID (e.g. C-1001)."
    ),
    system_prompt=CUSTOMER_RECORDS_AGENT_PROMPT,
    # Least privilege: only the MCP tools this agent needs.
    tool_names=frozenset({"get_customer"}),
)
