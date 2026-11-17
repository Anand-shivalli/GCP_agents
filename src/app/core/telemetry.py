"""Logging, tracing and metrics (OpenTelemetry).

The service emits standard OTel signals; configuration decides where they go.
No agent code knows about a vendor.

What produces spans                     How
--------------------------------------  ---------------------------------------------
HTTP requests                           FastAPI instrumentation
Run / resume / stream, lock wait        custom spans in services/runs.py
Postgres (checkpoints, locks)           psycopg instrumentation
Outbound HTTP (LLM APIs, MCP transport) httpx + httpx2 instrumentation (openai and mcp use httpx2)
MCP tool calls, both sides              FastMCP built-in OTel (trace context travels in MCP `_meta`)
LLM calls, agents, graph steps          OpenInference LangChain instrumentor (Arize; read by
                                        Phoenix, Langfuse, Opik, Arize and any OTLP backend)

LangSmith is separate and needs no OTel: LANGSMITH_TRACING=true sends runs to it
natively, alongside the OTel pipeline. Do NOT use LANGSMITH_TRACING_MODE=otel/hybrid:
that bridge activates span IDs it never exports, so everything nested inside a
LangChain tool (e.g. the MCP calls) ends up pointing at a parent that doesn't exist.

Logs carry otelTraceID/otelSpanID (logging instrumentation), so logs and traces join up.
Metrics: see the instruments at the bottom of this file.

PHI: prompts, messages and tool outputs are NOT recorded on spans unless
OTEL_CAPTURE_CONTENT=true (for LangSmith: LANGSMITH_HIDE_INPUTS/OUTPUTS=true).
"""

import logging

from fastapi import FastAPI
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPX2ClientInstrumentor, HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from app.core.config import Settings

_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def configure_logging(settings: Settings) -> None:
    # Adds otelTraceID / otelSpanID to every log record ("0" when there is no span).
    LoggingInstrumentor().instrument(set_logging_format=False, inject_trace_context=True)
    logging.basicConfig(
        level=settings.log_level,
        format=(
            "%(asctime)s %(levelname)s %(name)s "
            "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] %(message)s"
        ),
    )


def check_tracing_config(settings: Settings) -> None:
    """Fail fast on a tracing config that produces broken traces."""
    if settings.langsmith_tracing_mode in ("otel", "hybrid"):
        raise RuntimeError(
            "LANGSMITH_TRACING_MODE=otel/hybrid breaks parent links for spans nested in tools. "
            "Use LANGSMITH_TRACING=true (native) for LangSmith and OTEL_ENABLED=true for OTLP "
            "backends - they can run together."
        )


def setup_telemetry(
    settings: Settings,
    *,
    span_exporter: SpanExporter | None = None,
    metric_reader: MetricReader | None = None,
) -> None:
    """Install tracer + meter providers and library instrumentation.

    Exporters default to OTLP (endpoint/headers from OTEL_EXPORTER_OTLP_* env vars).
    Tests pass in-memory ones.
    """
    global _tracer_provider, _meter_provider
    check_tracing_config(settings)
    if not settings.otel_enabled or _tracer_provider is not None:
        return

    resource = Resource.create(
        {"service.name": settings.otel_service_name, "deployment.environment": settings.app_env}
    )
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter or OTLPSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)

    reader = metric_reader or PeriodicExportingMetricReader(OTLPMetricExporter())
    _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(_meter_provider)

    # Library instrumentation. Must run before the DB pool opens its connections.
    PsycopgInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    HTTPX2ClientInstrumentor().instrument()  # openai and mcp SDKs are on httpx2
    hide = not settings.otel_capture_content
    LangChainInstrumentor().instrument(
        tracer_provider=_tracer_provider,
        config=TraceConfig(
            hide_inputs=hide,
            hide_outputs=hide,
            hide_input_messages=hide,
            hide_output_messages=hide,
        ),
    )


def instrument_app(app: FastAPI) -> None:
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")


def shutdown_telemetry() -> None:
    # Flush buffered spans and metrics before the process exits.
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
    if _meter_provider is not None:
        _meter_provider.shutdown()


# ---- Instruments -----------------------------------------------------------------
# Created through the global API, so they are no-ops until setup_telemetry() runs.
# Token usage is not duplicated here: LLM spans carry gen_ai.usage.* attributes.

tracer = trace.get_tracer("app")
_meter = metrics.get_meter("app")

runs_total = _meter.create_counter(
    "app.runs", unit="{run}", description="Graph runs by operation and outcome status"
)
run_duration = _meter.create_histogram(
    "app.run.duration", unit="s", description="Wall time of a graph run"
)
lock_rejections = _meter.create_counter(
    "app.session.lock_rejections",
    unit="{request}",
    description="Requests refused because the session was busy (possibly on another replica)",
)
