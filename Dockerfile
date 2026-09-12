FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser
USER appuser

# Cloud Run injects $PORT (default 8080) and requires the container to bind
# to it on 0.0.0.0. Secrets (GEMINI_API_KEY, HORIZON_API_KEY) and
# MCP_SERVER_URL/MCP_AUTH_MODE are provided at deploy time, never baked in.
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn fluidbank_orchestrator.api:app --host 0.0.0.0 --port "${PORT}"
