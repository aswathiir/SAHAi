FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY libs/sahai-core /srv/libs/sahai-core
RUN pip install --no-cache-dir -e /srv/libs/sahai-core

COPY services/session/pyproject.toml /srv/services/session/
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx \
    "sqlalchemy[asyncio]" asyncpg python-multipart

COPY services/session/app /srv/services/session/app
WORKDIR /srv/services/session

RUN useradd --create-home --uid 10001 sahai && chown -R sahai /srv
USER sahai

EXPOSE 8000
# urllib, not httpx: the probe runs every 15s in every container, and importing
# httpx drags in anyio and pydantic each time. Stdlib keeps the probe cheap.
# 10s rather than 3s because the old budget measured the interpreter start as
# much as the service — with other containers competing for the same Docker VM
# the probe timed out while /health was still answering 200, and compose then
# refused to start anything that depended on it.
HEALTHCHECK --interval=15s --timeout=10s --retries=5 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=8).status<500 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
