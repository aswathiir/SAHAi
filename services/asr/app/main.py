"""sahai-asr — code-mixed speech to text.

Stubbed by design for this phase. The service boundary, request/response
contract, and health check are real, so the rest of the stack can be wired to
it now and the Whisper backend dropped in behind the same interface later.

The stub does NOT pretend to transcribe. It echoes a supplied transcript when
given one and otherwise returns an explicit not-implemented error, so a caller
can never mistake stub output for a real transcription.
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

BACKEND = os.getenv("SAHAI_ASR_BACKEND", "stub").lower()
MAX_AUDIO_BYTES = int(os.getenv("SAHAI_ASR_MAX_BYTES", str(25 * 1024 * 1024)))

app = FastAPI(title="sahai-asr", version="0.1.0")


class TranscriptOut(BaseModel):
    text: str
    language: Literal["hi", "ta", "en", "mixed", "unknown"] = "unknown"
    confidence: float | None = None
    backend: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sahai-asr", "backend": BACKEND}


@app.post("/transcribe", response_model=TranscriptOut)
async def transcribe(
    audio: UploadFile | None = None, transcript: str | None = None
) -> TranscriptOut:
    if BACKEND != "stub":  # pragma: no cover - whisper path not built yet
        raise HTTPException(501, f"ASR backend {BACKEND!r} not implemented")

    if transcript is not None:
        # Text passthrough: lets clients drive the full pipeline without audio.
        return TranscriptOut(text=transcript, language="mixed", backend="stub")

    if audio is not None:
        blob = await audio.read()
        if len(blob) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio too large")
        raise HTTPException(
            501,
            "ASR is stubbed in this phase; send `transcript` to use the text path.",
        )

    raise HTTPException(400, "provide either `audio` or `transcript`")
