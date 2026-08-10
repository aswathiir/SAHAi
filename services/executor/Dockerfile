# sahai-executor — runs untrusted, model-generated code.
#
# Hardened deliberately and differently from the other services. The in-process
# rlimits in app/sandbox.py are fast-failure guards; *this* file plus the
# compose settings are the actual containment boundary.

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY libs/sahai-core /srv/libs/sahai-core
RUN pip install --no-cache-dir -e /srv/libs/sahai-core \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx

COPY services/executor/app /srv/services/executor/app
WORKDIR /srv/services/executor

# Non-root, no login shell, no home write access beyond the tmpdir the sandbox
# creates per run.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin sahai \
 && chown -R sahai /srv
USER sahai

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health',timeout=2).status_code<500 else 1)"

# Single worker: concurrency is bounded inside the app by a fixed thread pool,
# so the host cannot be oversubscribed by a burst of expensive submissions.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
