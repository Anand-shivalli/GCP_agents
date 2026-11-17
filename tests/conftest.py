import os

# Must be set before app settings are first loaded.
os.environ.setdefault("PII_SPACY_MODEL", "en_core_web_sm")
os.environ.setdefault("OTEL_ENABLED", "false")
os.environ.pop("LANGSMITH_TRACING", None)
