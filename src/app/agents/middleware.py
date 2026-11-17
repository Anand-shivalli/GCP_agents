"""Middleware stacks - all from `langchain.agents.middleware`, none hand-rolled.

Order matters: `before_model` hooks run top-to-bottom, so PII redaction runs
before summarisation (the summariser never sees raw PII) and the model call is
wrapped by limits -> fallback -> retry.
"""

from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from app.agents.pii import presidio_detector
from app.core.config import Settings
from app.llm.client import ModelSet


def pii_stack(
    settings: Settings, *, on_input: bool, on_output: bool, on_tool_results: bool
) -> list[AgentMiddleware]:
    flags = {
        "apply_to_input": on_input,
        "apply_to_output": on_output,
        "apply_to_tool_results": on_tool_results,
    }
    builtin = [
        PIIMiddleware("email", strategy="redact", **flags),
        PIIMiddleware("credit_card", strategy="mask", **flags),
    ]
    presidio = PIIMiddleware("presidio", detector=presidio_detector, strategy="redact", **flags)
    return [*builtin, presidio] if settings.pii_entities else builtin


def resilience_stack(settings: Settings, models: ModelSet) -> list[AgentMiddleware]:
    stack: list[AgentMiddleware] = []
    if models.fallback is not None:
        stack.append(ModelFallbackMiddleware(models.fallback))
    stack.append(ModelRetryMiddleware(max_retries=settings.llm_max_retries))
    return stack


def supervisor_middleware(settings: Settings, models: ModelSet) -> list[AgentMiddleware]:
    return [
        # User input and the final answer are scrubbed.
        *pii_stack(settings, on_input=True, on_output=True, on_tool_results=False),
        # Sessions persist forever; keep the context window bounded.
        SummarizationMiddleware(
            model=models.summarizer,
            trigger=("tokens", settings.summary_trigger_tokens),
            keep=("messages", settings.summary_keep_messages),
        ),
        # Stop runaway loops gracefully instead of burning tokens.
        ModelCallLimitMiddleware(
            run_limit=settings.supervisor_max_model_calls_per_run, exit_behavior="end"
        ),
        *resilience_stack(settings, models),
    ]


def worker_middleware(settings: Settings, models: ModelSet) -> list[AgentMiddleware]:
    stack: list[AgentMiddleware] = [
        # Tool results (CRM records, clinical data...) are where PII/PHI enters.
        *pii_stack(settings, on_input=True, on_output=False, on_tool_results=True),
        ToolCallLimitMiddleware(run_limit=settings.worker_max_tool_calls_per_run),
        ToolRetryMiddleware(max_retries=2),
        *resilience_stack(settings, models),
    ]
    if settings.hitl_tools:
        # Pauses the WHOLE graph (interrupt propagates to the orchestrator);
        # the API resumes it via POST /v1/sessions/{id}/resume.
        stack.append(
            HumanInTheLoopMiddleware(interrupt_on={name: True for name in settings.hitl_tools})
        )
    return stack
