"""Demo MCP server (FastMCP) with synthetic customer data.

Runs as its own service, like real MCP servers will. Records deliberately
contain PII so you can watch the worker's PIIMiddleware redact tool results.

    python mcp_servers/customer_server.py          # HTTP on :8001/mcp

Tracing: FastMCP emits OTel spans natively and continues the caller's trace
(context arrives in the MCP request's `_meta`). This file only installs a
provider; set OTEL_ENABLED=true plus OTEL_EXPORTER_OTLP_ENDPOINT, or
OTEL_TRACES_EXPORTER=console to print spans.
"""

import os

from fastmcp import FastMCP

mcp = FastMCP("customer-records")

_CUSTOMERS = {
    "C-1001": {
        "customer_id": "C-1001",
        "name": "Asha Rao",
        "email": "asha.rao@example.com",
        "phone": "+1 415 555 0134",
        "policy_id": "P-77",
        "policy_status": "active",
    },
}
_CLAIMS = {"C-1001": [{"claim_id": "CL-9", "status": "under review", "amount": 1200}]}


@mcp.tool
def get_customer(customer_id: str) -> dict:
    """Return the customer profile and policy status for a customer ID like 'C-1001'."""
    return _CUSTOMERS.get(customer_id, {"error": f"no customer {customer_id}"})


@mcp.tool
def list_open_claims(customer_id: str) -> list[dict]:
    """List open insurance claims for a customer ID."""
    return _CLAIMS.get(customer_id, [])


def _setup_tracing() -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": "mcp-customer-records"}))
    if os.getenv("OTEL_TRACES_EXPORTER") == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


if __name__ == "__main__":
    if os.getenv("OTEL_ENABLED", "").lower() == "true":
        _setup_tracing()
    mcp.run(transport="http", host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
