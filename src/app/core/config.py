"""Application settings.

One typed object, loaded once from environment variables (and `.env` locally).
Every module reads configuration from here - never from `os.environ` directly -
so a reader can see every knob the service has in one file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Service -------------------------------------------------------------
    app_env: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"

    # --- Persistence (Postgres) ------------------------------------------------
    # All conversation state lives in Postgres, never in process memory, so any
    # replica behind a load balancer can serve any request for any session.
    database_url: SecretStr = SecretStr("postgresql://app:app@localhost:5432/app")
    db_pool_max_size: int = 20
    # Creates/migrates checkpoint tables on startup. Fine locally; in prod run
    # `python -m app.persistence.migrate` once per release instead, so N replicas
    # do not race each other at boot.
    db_run_setup_on_startup: bool = True
    # Encrypt checkpoints at rest (they contain user messages and tool results).
    # Reads the key from the LANGGRAPH_AES_KEY env var (16/24/32 bytes).
    checkpoint_encryption: bool = False

    # --- LLMs ---------------------------------------------------------------------
    # "<provider>:<model>" strings understood by langchain's init_chat_model.
    llm_supervisor_model: str = "openai:gpt-5.4-mini"
    llm_worker_model: str = "openai:gpt-5.4-mini"
    llm_summary_model: str = "openai:gpt-5.4-nano"
    llm_fallback_model: str | None = None
    # Point at an AI gateway / OpenAI-compatible proxy instead of the vendor.
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_temperature: float | None = None  # None = provider default (some models reject it)
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- Agent guardrails -----------------------------------------------------------
    supervisor_max_model_calls_per_run: int = 8
    worker_max_tool_calls_per_run: int = 10
    # Every middleware hook is a graph step, so this is higher than the number of LLM turns.
    graph_recursion_limit: int = 100
    max_input_chars: int = 8000
    # Conversation summarisation (supervisor keeps long sessions bounded).
    summary_trigger_tokens: int = 6000
    summary_keep_messages: int = 12
    # MCP tool names that need human approval before they run (interrupt/resume).
    hitl_tools: list[str] = Field(default_factory=list)

    # --- PII / PHI ---------------------------------------------------------------------
    # Presidio entity types to redact. Email + credit cards are handled by
    # LangChain's built-in regex detectors. DATE_TIME is PHI under HIPAA but
    # very noisy - add it if your domain needs it.
    pii_entities: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "PHONE_NUMBER",
            "LOCATION",
            "US_SSN",
            "MEDICAL_LICENSE",
            "UK_NHS",
            "IBAN_CODE",
            "US_BANK_NUMBER",
        ]
    )
    pii_spacy_model: str = "en_core_web_lg"
    # Low on purpose: for PHI, a missed identifier costs more than an extra redaction.
    # (Phone numbers without a nearby keyword like "phone" score 0.4.)
    pii_score_threshold: float = 0.35

    # --- MCP ------------------------------------------------------------------------------
    # Standard MCP config file ({"mcpServers": {...}}), same format as Claude/Cursor.
    mcp_config_path: str = "config/mcp.json"

    # --- Observability ---------------------------------------------------------------------
    # OTLP endpoint/headers are read by the exporter from the standard
    # OTEL_EXPORTER_OTLP_* env vars - see .env.example for LangSmith/Opik/Langfuse.
    otel_enabled: bool = False
    otel_service_name: str = "langgraph-starter"
    # Record prompts / messages / tool outputs on spans. Keep false for PII/PHI workloads.
    otel_capture_content: bool = False
    # Read by the LangSmith SDK itself; declared here only so startup can reject otel/hybrid.
    langsmith_tracing_mode: Literal["langsmith", "otel", "hybrid"] | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
