"""Microsoft Presidio as detectors for LangChain's PIIMiddleware.

LangChain's built-in detectors are regexes (email, credit card, IP, MAC, URL).
Names, addresses, phone numbers, medical/national IDs need NER + validators,
which is what Presidio provides. We do not re-implement redaction: Presidio
only *finds* spans; PIIMiddleware does the redact/mask/hash/block and state
updates.
"""

from functools import lru_cache

from langchain.agents.middleware import PIIMatch
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    settings = get_settings()
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": settings.pii_spacy_model}],
        }
    ).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


# Tool results are often compact JSON ({"name":"Asha Rao"}), which defeats
# spaCy's tokenizer. Swapping JSON punctuation for spaces keeps every character
# offset identical, so matches still map 1:1 onto the original text.
_JSON_PUNCTUATION = str.maketrans({c: " " for c in '"{}[],:'})


@lru_cache(maxsize=2048)
def _analyze(text: str) -> tuple[RecognizerResult, ...]:
    settings = get_settings()
    return tuple(
        _analyzer().analyze(
            text=text.translate(_JSON_PUNCTUATION),
            language="en",
            entities=settings.pii_entities,
            score_threshold=settings.pii_score_threshold,
        )
    )


def presidio_detector(content: str) -> list[PIIMatch]:
    """PIIMiddleware-compatible detector covering all configured Presidio entities.

    One middleware for all entities (instead of one per entity) because every
    middleware hook is a graph step. Labels stay specific - PIIMiddleware names
    each placeholder from the match type, e.g. [REDACTED_PERSON].
    """
    matches: list[PIIMatch] = []
    # Longest span first, then drop overlaps so replacements never collide.
    for r in sorted(_analyze(content), key=lambda r: (r.start, -(r.end - r.start))):
        if matches and r.start < matches[-1]["end"]:
            continue
        matches.append(
            PIIMatch(type=r.entity_type, value=content[r.start : r.end], start=r.start, end=r.end)
        )
    return matches


def warm_up() -> None:
    """Load the spaCy model at startup instead of on the first user request."""
    _analyze("warm up")
