"""LLM client factory - the ONLY place chat models are constructed.

Why a separate module:
  * Swapping provider, gateway URL or model per role is a config change.
  * Retries/timeouts are set consistently (provider-level retries are disabled
    because ModelRetryMiddleware owns retrying - otherwise retries multiply).
  * Tests inject fakes through `ModelSet` instead of patching imports.
"""

from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


def build_chat_model(model_id: str, settings: Settings) -> BaseChatModel:
    kwargs: dict = {"timeout": settings.llm_timeout_s, "max_retries": 0}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key.get_secret_value()
    return init_chat_model(model_id, **kwargs)


@dataclass(frozen=True)
class ModelSet:
    """The models each role uses. Build once at startup; pass down explicitly."""

    supervisor: BaseChatModel
    worker: BaseChatModel
    summarizer: BaseChatModel
    fallback: BaseChatModel | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelSet":
        return cls(
            supervisor=build_chat_model(settings.llm_supervisor_model, settings),
            worker=build_chat_model(settings.llm_worker_model, settings),
            summarizer=build_chat_model(settings.llm_summary_model, settings),
            fallback=(
                build_chat_model(settings.llm_fallback_model, settings)
                if settings.llm_fallback_model
                else None
            ),
        )
