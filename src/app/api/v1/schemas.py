"""Public API contract (v1). Changing a field here is a breaking change for clients.

These models are the OpenAPI schema served at /docs - share that with
front-end / Java / .NET teams instead of hand-written docs.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

SessionStatus = Literal["new", "completed", "interrupted", "incomplete"]


class CreateSessionResponse(BaseModel):
    session_id: str


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)


class ResumeRequest(BaseModel):
    # Omit to retry a run that died. Provide to answer a human-in-the-loop
    # interrupt, e.g. {"decisions": [{"type": "approve"}]}.
    value: Any | None = None


class RunResponse(BaseModel):
    session_id: str
    run_id: str  # matches the trace/run id in LangSmith / OTel
    status: SessionStatus
    output: str | None = None
    interrupts: list[Any] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["human", "ai"]
    content: str


class SessionResponse(BaseModel):
    session_id: str
    status: SessionStatus
    messages: list[ChatMessage]


class ErrorResponse(BaseModel):
    code: str
    detail: str
