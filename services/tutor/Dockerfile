FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY libs/sahai-core /srv/libs/sahai-core
RUN pip install --no-cache-dir -e /srv/libs/sahai-core

COPY services/tutor/pyproject.toml /srv/services/tutor/
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx \
    "sqlalchemy[asyncio]" asyncpg python-multipart

# HF backend deps: CPU-only torch wheel, since this container has no GPU
# passthrough (Docker Desktop on macOS doesn't expose MPS to containers).
# Optional at the image level would need a second Dockerfile; simpler to
# always install here since this is the one image the "hf" backend runs in.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers peft accelerate safetensors

COPY services/tutor/app /srv/services/tutor/app
WORKDIR /srv/services/tutor

RUN useradd --create-home --uid 10001 sahai && chown -R sahai /srv
# Pre-create with the right owner so the *first* mount of the hf-cache named
# volume (Docker seeds a new volume from whatever's already at the mount
# point) inherits sahai's ownership instead of defaulting to root.
RUN mkdir -p /home/sahai/.cache/huggingface && chown -R sahai:sahai /home/sahai/.cache
USER sahai

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health',timeout=2).status_code<500 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
