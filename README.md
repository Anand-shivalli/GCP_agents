# LangGraph Starter

A small, production-shaped template: **orchestrator → supervisor → MCP tool agent**, on
LangGraph 1.x / LangChain 1.x. State is persisted in Postgres, so any replica can serve any
request. The middleware (PII/PHI, summarisation, limits, retries, human approval) comes from
libraries.

```
            HTTP (FastAPI, /v1)                         Postgres
                   │                                ┌──────────────────┐
             RunService ── session lock ──────────▶ │ advisory locks   │
                   │                                │ checkpoints      │◀─ every step
   ┌───────────────▼──────────────┐                 └──────────────────┘
   │ Orchestrator  (StateGraph)   │  only graph with a checkpointer
   │  guard_input → supervisor → finalize           │
   └───────────────┬──────────────┘
   ┌───────────────▼──────────────┐
   │ Supervisor  (create_agent)   │  PII in/out · summarisation · call limit · retry/fallback
   │  tools = worker agents       │
   └───────────────┬──────────────┘
   ┌───────────────▼──────────────┐       ┌─────────────────────────┐
   │ Worker  (create_agent)       │──MCP─▶│ MCP server (own service)│
   │  tools = MCP tools           │       └─────────────────────────┘
   └──────────────────────────────┘  PII on tool results · tool limit · tool retry · HITL
```

## Layout: one job per module

| Path | What it does | You change it when... |
|---|---|---|
| `core/config.py` | All settings (pydantic-settings) | adding a config knob |
| `core/telemetry.py` | Logging, OTel tracing and metrics, all instrumentation | changing the observability backend (usually just env vars) |
| `llm/client.py` | **The only place models are built** (`init_chat_model`) | changing provider, gateway or per-role models |
| `persistence/checkpointer.py` | Pool and `AsyncPostgresSaver` (optional encryption) | tuning the DB |
| `persistence/locks.py` | One active run per session, across replicas | — |
| `tools/mcp.py` | MCP servers → LangChain tools (`langchain.mcp`) | — (servers go in `config/mcp.json`) |
| `agents/workers/__init__.py` | Registry: the list of live worker agents | **adding a specialist agent** |
| `agents/workers/base.py` | `WorkerSpec` and the one factory that builds every worker | rarely |
| `agents/workers/agent_1/`, `agent_2/` | One folder per worker: `agent.py` (its spec) | changing that agent's tools or routing description |
| `agents/supervisor.py` | Wraps workers as tools; supervisor factory | changing how work is delegated |
| `agents/middleware.py` | Middleware stacks per role | adding or tuning guardrails |
| `agents/pii.py` | Presidio as the PII/PHI detector for `PIIMiddleware` | adding entity types or recognizers |
| `agents/prompts.py` | **All system prompts** (supervisor and every worker) | changing agent behaviour |
| `graph/orchestrator.py` | Deterministic outer workflow | adding validation, policy or audit steps |
| `graph/factory.py` | Composition root (wires everything) | rarely |
| `services/runs.py` | Run/resume/stream, ownership, status | changing run semantics |
| `api/` | HTTP contract (`v1/schemas.py`), routes, auth dependency | adding endpoints, **plugging in real auth** |

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
python -m spacy download en_core_web_sm      # tests; prod image uses en_core_web_lg
pytest -q                                    # offline: fake LLMs, in-process MCP, in-memory saver
```

Full stack:

```bash
cp .env.example .env                         # set OPENAI_API_KEY (or LLM_BASE_URL for a gateway)
docker compose up --build                    # postgres + MCP server + API on :8000
```

Without Docker for the API (Postgres and the MCP server still need to be running):
`python mcp_servers/customer_server.py`, then `python -m app`. On Windows always use
`python -m app`, because it sets the event loop psycopg needs.

OpenAPI docs: http://localhost:8000/docs

## API contract (v1)

Identity comes from `X-User-Id`. **This is a placeholder:** swap `api/deps.py:current_user_id`
for JWT validation before exposing the service.

| Method & path | Purpose | Response |
|---|---|---|
| `POST /v1/sessions` | New session id (server-generated) | `201 {session_id}` |
| `POST /v1/sessions/{id}/messages` | Send a message, wait for the result | `RunResponse` |
| `POST /v1/sessions/{id}/messages:stream` | Same, as SSE (`token`… then `result`) | `text/event-stream` |
| `POST /v1/sessions/{id}/resume` | `{}` retries a crashed run; `{"value": …}` answers an approval | `RunResponse` |
| `GET /v1/sessions/{id}` | Status and visible transcript | `SessionResponse` |
| `GET /healthz`, `/readyz` | Liveness, readiness (DB reachable) | |

`RunResponse = {session_id, run_id, status, output, interrupts}`, where `status` is one of
`completed | interrupted | incomplete`. `run_id` equals the OTel trace ID of the request (and
is the root run ID in LangSmith), so a user's bug report maps straight to one trace.

Errors look like `{code, detail}`:
- `404 session_not_found`: the session is unknown or belongs to another user. The two cases
  return the same error on purpose.
- `409 session_busy` (with a `Retry-After` header): another replica is running this session.
- `409 invalid_session_state`: for example, a new message sent while an approval is pending.

## Built for many replicas, run on one

- **No process memory holds conversation state.** `thread_id = session_id`. Every graph step is
  checkpointed to Postgres with `durability="sync"`, so a follow-up can land on any instance.
- **Concurrent requests for one session** are serialised by a Postgres advisory lock. If an
  instance dies, Postgres releases its lock.
- **Crash recovery.** A run that died mid-way shows as `incomplete`. `POST …/resume` with `{}`
  continues from the last completed step (see `test_crashed_run_resumes_from_last_checkpoint`).
- **User and session IDs** go on every checkpoint (`metadata`) and every trace. Sessions are bound
  to their first user.
- **Migrations** run at startup locally. In prod, set `DB_RUN_SETUP_ON_STARTUP=false` and run
  `python -m app.persistence.migrate` once per release.
- **Checkpoints at rest** contain raw input and tool output. Set `CHECKPOINT_ENCRYPTION=true`
  and `LANGGRAPH_AES_KEY` for PHI workloads, and set a retention policy on the checkpoint tables.

## Middleware (all from `langchain.agents.middleware`)

| Where | Middleware | Why |
|---|---|---|
| Supervisor | `PIIMiddleware` ×3 (email, card mask, Presidio) on input and output | User text and answers are scrubbed before the model sees them and before they are persisted |
| Supervisor | `SummarizationMiddleware` | Long-lived sessions stay inside the context window |
| Supervisor | `ModelCallLimitMiddleware`, `ModelFallbackMiddleware`, `ModelRetryMiddleware` | Stop runaway loops; survive provider errors |
| Worker | `PIIMiddleware` on tool results | MCP records are where PII/PHI enters |
| Worker | `ToolCallLimitMiddleware`, `ToolRetryMiddleware` | Bound tool usage; retry flaky tools |
| Worker | `HumanInTheLoopMiddleware` (when `HITL_TOOLS` is set) | Approval pauses the *whole* graph; resume through the API |

PII/PHI detection uses **Microsoft Presidio** (spaCy NER plus validated recognizers) plugged in as
a `PIIMiddleware` detector. Presidio only finds the spans; LangChain does the redaction. Tune
`PII_ENTITIES` and `PII_SCORE_THRESHOLD`. The threshold is low on purpose, because for PHI a
missed identifier costs more than an extra redaction.

## Observability: set env vars, change no code

One request produces **one trace** covering every layer. This is verified against real Postgres
and an MCP server running as its own process: about 175 spans, no broken parent links.

| Layer | Spans from |
|---|---|
| HTTP API | `opentelemetry-instrumentation-fastapi` |
| Run, resume and stream, plus the session-lock wait | custom spans in `services/runs.py` (`session.*`), tagged with `user.id`, `session.id` and `run.id` |
| Graph, supervisor, workers, LLM and tool calls | `openinference-instrumentation-langchain` |
| MCP calls, client and server side | FastMCP's built-in OTel. The trace continues into the MCP server process |
| Postgres (checkpoints, locks) | `opentelemetry-instrumentation-psycopg` |
| Outbound HTTP (LLM APIs, MCP transport) | httpx and **httpx2** instrumentation. The openai and mcp SDKs both moved to httpx2 |

**Logs** carry `trace_id` and `span_id`. **Metrics:** `app.runs` (by operation and status),
`app.run.duration` and `app.session.lock_rejections`. Token usage is recorded on the LLM spans.

Backends:
- **Opik, Langfuse, Phoenix, Arize or any collector:** set `OTEL_ENABLED=true` and
  `OTEL_EXPORTER_OTLP_*`. They all read OpenInference spans.
- **LangSmith:** `LANGSMITH_TRACING=true`. This is native and independent of OTel, and can run
  alongside it. Do **not** use `LANGSMITH_TRACING_MODE=otel/hybrid`, because that bridge breaks
  parent links for anything nested in a tool. Startup refuses it.

**PII/PHI in traces:** prompts, messages and tool outputs are **not** recorded on spans unless you
set `OTEL_CAPTURE_CONTENT=true`. A test checks this. For LangSmith, set
`LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true`.

## Adding things

- **A new specialist:** add its prompt to `agents/prompts.py`, copy `agents/workers/agent_1/`
  to a new folder, edit its `agent.py` (name, description, prompt, allowed MCP tools), then add
  its `SPEC` to `WORKERS` in `agents/workers/__init__.py`. The supervisor gets the tool
  automatically.
- **A new MCP server:** add it to `config/mcp.json` (the standard `mcpServers` format).
- **A deterministic step** (policy check, audit write): add a node in `graph/orchestrator.py`.

## Upstream changes this template follows (Nov 2026)

- `langgraph-supervisor` (`create_supervisor`) is **no longer maintained**. The replacement is a
  `create_agent` supervisor that calls subagents wrapped as tools.
  [Migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-supervisor)
- `create_react_agent` is deprecated in favour of `langchain.agents.create_agent` with middleware.
- `langchain-mcp-adapters` is **no longer maintained**. The replacement is `langchain[mcp]` →
  `langchain.mcp.MCPAdapter` (built on FastMCP). That module is still marked beta, which is why
  all MCP access sits behind `tools/mcp.py`.
- The openai and mcp SDKs now use `httpx2`. Use `HTTPX2ClientInstrumentor`: the plain httpx
  instrumentor sees none of their traffic.
- Subagents must **not** get their own checkpointer. They inherit the parent's at runtime, which
  is what makes interrupt and resume work across levels.
