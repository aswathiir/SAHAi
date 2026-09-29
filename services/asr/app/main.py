"""sahai-asr — code-mixed speech to text, and text to speech.

Stub by default. The service boundary, request/response contracts, and health
check are real regardless of backend, so the rest of the stack can be wired to
it now and a real backend dropped in behind the same interface later.

The stub does NOT pretend to transcribe or synthesize. It echoes a supplied
transcript when given one (ASR) or raises 501 (both), so a caller can never
mistake stub output for real speech processing.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .backends import (
    ASRBackend,
    IndicTransTranslator,
    TTSBackend,
    build_asr_backend,
    build_tts_backend,
    has_indic_script,
)

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 2000

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("asr")

_state: dict[str, ASRBackend | TTSBackend] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["asr"] = build_asr_backend()
    _state["tts"] = build_tts_backend()
    # Constructed, not loaded: the weights arrive on the first turn that
    # actually contains Indic script.
    _state["translate"] = IndicTransTranslator()
    logger.info("asr backend ready: %s", _state["asr"].describe)
    logger.info("tts backend ready: %s", _state["tts"].describe)
    yield


app = FastAPI(title="sahai-asr", version="0.1.0", lifespan=lifespan)


class TranscriptOut(BaseModel):
    text: str
    language: Literal["hi", "ta", "en", "mixed", "unknown"] = "unknown"
    confidence: float | None = None
    backend: str


class TranslateRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    source_lang: str = Field(default="hi", max_length=8)


class TranslateOut(BaseModel):
    text: str
    translated: bool
    source_lang: str
    backend: str


class SynthesizeRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    language: Literal["hi", "ta", "en", "mixed", "unknown"] = "en"


@app.get("/health")
async def health() -> dict[str, str]:
    asr = _state.get("asr")
    tts = _state.get("tts")
    return {
        "status": "ok" if asr and tts else "starting",
        "service": "sahai-asr",
        "asr_backend": asr.describe if asr else "none",
        "tts_backend": tts.describe if tts else "none",
    }


@app.post("/transcribe", response_model=TranscriptOut)
async def transcribe(
    audio: UploadFile | None = File(default=None),
    transcript: str | None = Form(default=None),
) -> TranscriptOut:
    asr = _state["asr"]

    if transcript is not None:
        # Text passthrough: lets clients drive the full pipeline without audio,
        # regardless of which ASR backend is configured.
        return TranscriptOut(text=transcript, language="mixed", backend="passthrough")

    if audio is not None:
        blob = await audio.read()
        if len(blob) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio too large")
        try:
            result = asr.transcribe(blob, language="unknown")
        except NotImplementedError as exc:
            raise HTTPException(501, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface as a clean 500, not a 5xx traceback leak
            logger.exception("transcription failed")
            raise HTTPException(500, f"transcription failed: {exc}") from exc
        return TranscriptOut(
            text=result.text,
            language=result.language if result.language in ("hi", "ta", "en") else "mixed",
            backend=asr.describe,
        )

    raise HTTPException(400, "provide either `audio` or `transcript`")


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest) -> Response:
    tts = _state["tts"]
    try:
        wav_bytes = tts.synthesize(req.text, req.language)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("synthesis failed")
        raise HTTPException(500, f"synthesis failed: {exc}") from exc
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/translate", response_model=TranslateOut)
def translate(req: TranslateRequest) -> TranslateOut:
    """Indic script to English, for text on its way to the tutor.

    A plain `def`: the model is synchronous and CPU-bound, and on the event
    loop it would block this service's own health probe, which is the bug that
    made the tutor look dead for 47 hours.

    Text without Indic script is returned untouched and `translated=False`.
    Romanised Hinglish is left alone deliberately — it is already Latin, the
    tutor reads it better than Devanagari, and translating it away would
    discard the code-mixing the project is about.
    """
    if not has_indic_script(req.text):
        return TranslateOut(text=req.text, translated=False,
                            source_lang=req.source_lang, backend="skipped")
    backend = _state["translate"]
    try:
        out = backend.translate(req.text, req.source_lang)
    except Exception as exc:  # noqa: BLE001
        # Fail open. A turn the learner can still send in their own words beats
        # a 500, and the tutor handling it poorly is a smaller loss than the
        # turn not happening.
        logger.warning("translate failed: %s", type(exc).__name__)
        return TranslateOut(text=req.text, translated=False,
                            source_lang=req.source_lang, backend="error")
    return TranslateOut(text=out or req.text, translated=bool(out),
                        source_lang=req.source_lang, backend=backend.describe)
