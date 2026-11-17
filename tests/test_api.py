"""API contract tests: routes, status codes and error mapping (no DB needed).

httpx's ASGITransport does not run the lifespan, so we wire the RunService
ourselves with the same in-memory graph used in test_workflow.
"""

import httpx
import pytest

from app.main import create_app
from app.tools.mcp import MCPAdapter
from mcp_servers.customer_server import mcp as customer_server
from tests.test_workflow import _service


@pytest.fixture
async def client():
    app = create_app()
    app.state.run_service = _service(await MCPAdapter(customer_server).list_tools())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_session_lifecycle(client: httpx.AsyncClient):
    alice = {"X-User-Id": "alice"}
    session_id = (await client.post("/v1/sessions", headers=alice)).json()["session_id"]

    r = await client.post(
        f"/v1/sessions/{session_id}/messages", json={"message": "status of C-1001?"}, headers=alice
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and "P-77" in body["output"]

    r = await client.get(f"/v1/sessions/{session_id}", headers=alice)
    assert r.json()["status"] == "completed"
    assert [m["role"] for m in r.json()["messages"]] == ["human", "ai"]


async def test_errors_follow_the_contract(client: httpx.AsyncClient):
    alice, bob = {"X-User-Id": "alice"}, {"X-User-Id": "bob"}
    sid = (await client.post("/v1/sessions", headers=alice)).json()["session_id"]
    await client.post(f"/v1/sessions/{sid}/messages", json={"message": "hi"}, headers=alice)

    r = await client.get(f"/v1/sessions/{sid}", headers=bob)  # someone else's session
    assert r.status_code == 404 and r.json()["code"] == "session_not_found"

    r = await client.post(f"/v1/sessions/{sid}/resume", json={}, headers=alice)  # nothing to resume
    assert r.status_code == 409 and r.json()["code"] == "invalid_session_state"

    r = await client.post(f"/v1/sessions/{sid}/messages", json={"message": "hi"})  # no identity
    assert r.status_code == 422


async def test_streaming_ends_with_result_event(client: httpx.AsyncClient):
    alice = {"X-User-Id": "alice"}
    sid = (await client.post("/v1/sessions", headers=alice)).json()["session_id"]
    async with client.stream(
        "POST", f"/v1/sessions/{sid}/messages:stream", json={"message": "C-1001?"}, headers=alice
    ) as r:
        text = "".join([chunk async for chunk in r.aiter_text()])
    assert "event: result" in text and '"status": "completed"' in text
