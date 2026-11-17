"""Telemetry: one request -> one trace, spanning every layer, plus metrics and log IDs.

Uses in-memory exporters instead of OTLP; everything else (instrumentors,
OpenInference LangChain bridge, FastMCP telemetry) is the production setup.
"""

import logging

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings
from app.core.telemetry import check_tracing_config, configure_logging, setup_telemetry
from app.main import create_app
from app.tools.mcp import MCPAdapter
from mcp_servers.customer_server import mcp as customer_server
from tests.test_workflow import _service

SPANS = InMemorySpanExporter()
METRICS = InMemoryMetricReader()


@pytest.fixture(scope="module", autouse=True)
def telemetry():
    # Global providers can only be installed once per process; this module owns them.
    setup_telemetry(Settings(otel_enabled=True), span_exporter=SPANS, metric_reader=METRICS)


async def _post_message(user: str = "alice") -> dict:
    app = create_app()
    app.state.run_service = _service(await MCPAdapter(customer_server).list_tools())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        headers = {"X-User-Id": user}
        sid = (await client.post("/v1/sessions", headers=headers)).json()["session_id"]
        r = await client.post(
            f"/v1/sessions/{sid}/messages",
            json={"message": "I'm Asha Rao, status of C-1001?"},
            headers=headers,
        )
    trace.get_tracer_provider().force_flush()
    return {"session_id": sid, **r.json()}


async def test_one_request_is_one_trace_across_all_layers():
    SPANS.clear()
    body = await _post_message()
    spans = SPANS.get_finished_spans()
    names = [s.name for s in spans]

    http = next(s for s in spans if s.name == "POST /v1/sessions/{session_id}/messages")
    run = next(s for s in spans if s.name == "session.send_message")

    # Our spans: HTTP -> run -> lock, all carrying the correlation IDs.
    assert run.parent.span_id == http.context.span_id
    assert "session.lock.acquire" in names
    for span in (http, run):
        assert span.attributes["user.id"] == "alice"
        assert span.attributes["session.id"] == body["session_id"]
        assert span.attributes["run.id"] == body["run_id"]
    assert run.attributes["session.status"] == "completed"

    # The run_id the API returns is the trace ID.
    assert http.context.trace_id == int(body["run_id"].replace("-", ""), 16)

    # LangGraph/agent spans (OpenInference) and FastMCP tool calls land in the
    # SAME trace, and every span's parent exists (no broken tree).
    same_trace = [s for s in spans if s.context.trace_id == http.context.trace_id]
    same_names = {s.name for s in same_trace}
    assert {"orchestrator", "supervisor", "customer_records_agent"} <= same_names, same_names
    assert "MCP send tools/call get_customer" in same_names, same_names
    assert "tools/call get_customer" in same_names  # the MCP server's own span
    ids = {s.context.span_id for s in same_trace}
    orphans = [s.name for s in same_trace if s.parent and s.parent.span_id not in ids]
    assert orphans == []


async def test_spans_carry_no_message_content():
    """PII/PHI must not leak into traces (OTEL_CAPTURE_CONTENT defaults to false)."""
    SPANS.clear()
    await _post_message()
    for span in SPANS.get_finished_spans():
        for value in (span.attributes or {}).values():
            assert "Asha Rao" not in str(value), span.name
            assert "asha.rao@example.com" not in str(value), span.name


async def test_run_metrics_are_recorded():
    await _post_message()
    points = {
        m.name: m
        for rm in METRICS.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }
    runs = points["app.runs"].data.data_points
    assert any(
        p.attributes == {"operation": "send_message", "status": "completed"} and p.value >= 1
        for p in runs
    )
    assert "app.run.duration" in points


def test_logs_carry_trace_ids(caplog):
    configure_logging(Settings())
    with trace.get_tracer("test").start_as_current_span("x") as span, caplog.at_level("INFO"):
        logging.getLogger("app").info("hello")
    record = next(r for r in caplog.records if r.getMessage() == "hello")
    assert record.otelTraceID == format(span.get_span_context().trace_id, "032x")


def test_langsmith_otel_bridge_is_rejected():
    with pytest.raises(RuntimeError, match="LANGSMITH_TRACING_MODE"):
        check_tracing_config(Settings(langsmith_tracing_mode="otel", otel_enabled=True))
