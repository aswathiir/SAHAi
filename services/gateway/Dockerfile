FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY libs/sahai-core /srv/libs/sahai-core
RUN pip install --no-cache-dir -e /srv/libs/sahai-core

COPY services/gateway/pyproject.toml /srv/services/gateway/
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx \
    "sqlalchemy[asyncio]" asyncpg python-multipart

COPY services/gateway/app /srv/services/gateway/app
WORKDIR /srv/services/gateway

RUN useradd --create-home --uid 10001 sahai && chown -R sahai /srv
USER sahai

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health',timeout=2).status_code<500 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
