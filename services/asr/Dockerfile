FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

# git: needed for the parler-tts pip install below (it's not on PyPI, only
# installable from its GitHub repo). python:3.11-slim doesn't ship it.
# git: parler-tts is install-from-git only.
# ffmpeg: browsers record WebM/Opus (Chrome) or MP4/AAC (Safari), and
#   libsndfile reads neither. Decoding through ffmpeg means one code path
#   for every container a MediaRecorder can produce, instead of guessing.
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY libs/sahai-core /srv/libs/sahai-core
RUN pip install --no-cache-dir -e /srv/libs/sahai-core

COPY services/asr/pyproject.toml /srv/services/asr/
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx \
    "sqlalchemy[asyncio]" asyncpg python-multipart

# ASR (IndicConformer) + TTS (Indic Parler-TTS) deps: CPU-only torch wheel,
# since this container has no GPU passthrough (Docker Desktop on macOS
# doesn't expose MPS to containers). onnxruntime/onnx pinned per
# IndicConformer's model card. parler-tts isn't on PyPI, so it's a git
# install — that's also why it's last, so a git-clone hiccup doesn't
# invalidate the (much slower) torch layer's build cache.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers onnxruntime==1.20.1 onnx==1.20.1 soundfile \
 && pip install --no-cache-dir "parler-tts @ git+https://github.com/huggingface/parler-tts.git"

COPY services/asr/app /srv/services/asr/app
WORKDIR /srv/services/asr

RUN useradd --create-home --uid 10001 sahai && chown -R sahai /srv
# Pre-create with the right owner so the *first* mount of a hf-cache named
# volume (Docker seeds a new volume from whatever's already at the mount
# point) inherits sahai's ownership instead of defaulting to root.
RUN mkdir -p /home/sahai/.cache/huggingface && chown -R sahai:sahai /home/sahai/.cache
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
