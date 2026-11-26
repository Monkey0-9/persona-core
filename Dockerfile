FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=production \
    MALL_SEG_CONFIG=/app/configs/pipeline_config.yaml \
    DATABASE_URL=sqlite:////app/artifacts/mall_seg.db

RUN groupadd --system appgroup && useradd --system --gid appgroup --create-home appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY alembic.ini ./
COPY alembic ./alembic
COPY configs ./configs
COPY data ./data
COPY ml ./ml
COPY pipelines ./pipelines
COPY scripts ./scripts
COPY app ./app
COPY src ./src
COPY tests/fixtures ./tests/fixtures

RUN mkdir -p /app/artifacts/models /app/public/artifacts && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health/live', timeout=3)" || exit 1

CMD ["uvicorn", "mall_seg.api:app", "--host", "0.0.0.0", "--port", "8010"]
