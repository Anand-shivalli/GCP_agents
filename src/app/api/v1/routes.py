"""HTTP routes. Thin by design: parse -> call RunService -> map result.

Every endpoint is safe to hit on any replica: nothing here reads or writes
process memory; the session lives in Postgres under `session_id`.
"""

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import Ctx, Runs, current_user_id
from app.api.v1.schemas import (
    CreateSessionResponse,
    ErrorResponse,
    MessageRequest,
    ResumeRequest,
    RunResponse,
    SessionResponse,
)

router = APIRouter(
    prefix="/v1",
    tags=["sessions"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(_: Annotated[str, Depends(current_user_id)]) -> CreateSessionResponse:
    # IDs are server-generated, so they are unguessable. The session is
    # persisted (and bound to the user) by its first message.
    return CreateSessionResponse(session_id=str(uuid.uuid4()))


@router.post("/sessions/{session_id}/messages", response_model=RunResponse)
async def send_message(body: MessageRequest, ctx: Ctx, runs: Runs) -> RunResponse:
    result = await runs.send_message(ctx, body.message)
    return RunResponse(**result.__dict__)


@router.post("/sessions/{session_id}/messages:stream")
async def stream_message(body: MessageRequest, ctx: Ctx, runs: Runs) -> StreamingResponse:
    """Server-Sent Events: `event: token` while generating, then one `event: result`."""

    async def sse():
        try:
            async for event, data in runs.stream_message(ctx, body.message):
                yield f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
        except Exception as exc:  # headers are already sent; report in-band
            yield f"event: error\ndata: {json.dumps({'detail': type(exc).__name__})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/resume", response_model=RunResponse)
async def resume(body: ResumeRequest, ctx: Ctx, runs: Runs) -> RunResponse:
    result = await runs.resume(ctx, body.value)
    return RunResponse(**result.__dict__)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(ctx: Ctx, runs: Runs) -> SessionResponse:
    status, messages = await runs.get_session(ctx)
    return SessionResponse(session_id=ctx.session_id, status=status, messages=messages)
