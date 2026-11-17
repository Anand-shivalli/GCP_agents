"""FastAPI dependencies: identity and service lookup."""

from typing import Annotated

from fastapi import Depends, Header, Request

from app.agents.context import RequestContext
from app.services.runs import RunService


async def current_user_id(x_user_id: Annotated[str, Header(min_length=1)]) -> str:
    # PLACEHOLDER: replace with real auth (validate a JWT, return its `sub`).
    # Never trust a client-supplied user id in production.
    return x_user_id


def request_context(
    session_id: str, user_id: Annotated[str, Depends(current_user_id)]
) -> RequestContext:
    return RequestContext(user_id=user_id, session_id=session_id)


def run_service(request: Request) -> RunService:
    return request.app.state.run_service


Ctx = Annotated[RequestContext, Depends(request_context)]
Runs = Annotated[RunService, Depends(run_service)]
