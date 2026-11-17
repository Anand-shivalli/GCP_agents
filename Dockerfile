FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache . \
    && python -m spacy download en_core_web_lg

COPY config ./config
COPY mcp_servers ./mcp_servers

# Stateless: scale horizontally, any replica can serve any session.
USER nobody
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
