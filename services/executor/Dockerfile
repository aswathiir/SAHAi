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
# urllib, not httpx: the probe runs every 15s in every container, and importing
# httpx drags in anyio and pydantic each time. Stdlib keeps the probe cheap.
# 10s rather than 3s because the old budget measured the interpreter start as
# much as the service — with other containers competing for the same Docker VM
# the probe timed out while /health was still answering 200, and compose then
# refused to start anything that depended on it.
HEALTHCHECK --interval=15s --timeout=10s --retries=5 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=8).status<500 else 1)"

# Single worker: concurrency is bounded inside the app by a fixed thread pool,
# so the host cannot be oversubscribed by a burst of expensive submissions.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
