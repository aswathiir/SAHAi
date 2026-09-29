"""sahai-tutor — inference for the trained tutor policy.

Stateless. Knows nothing about learners, sessions, or rewards; it turns a
message list into a tutor turn. Session state lives in sahai-session, learner
state in sahai-tracer.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sahai_core import RULES
from sahai_core.generation import strip_code

from .backends import TutorBackend, build_backend

logger = logging.getLogger(__name__)

# The serving prompt lives in sahai_core.prompt — one definition, imported by
# session (which assembles it) and echoed by /system_prompt below. It used to
# be duplicated here, and this copy was the dead one: nothing called the
# endpoint, so it read like documentation of behaviour the system lacked.

_state: dict[str, TutorBackend] = {}

# Generation is synchronous, CPU-bound and minutes long. Two consequences had
# to be handled, and neither was:
#
#   * Declared `async def`, it ran **on the event loop**, so nothing else in
#     this process could be served while a turn generated — including
#     `/health`. The container's own health probe timed out, Docker marked the
#     service unhealthy after three strikes, and a second learner's turn simply
#     waited. That is the "tutor stops accepting requests" symptom.
#   * Unbounded, N concurrent turns would each take a slice of the same CPU and
#     all of them would miss their deadline instead of one succeeding.
#
# So: the endpoint is a plain `def` (FastAPI runs it in a worker thread, the
# loop stays free), and this semaphore bounds how many run at once. Requests
# over the limit are refused immediately rather than queued behind a
# multi-minute job they would outlive anyway.
MAX_CONCURRENT = int(os.getenv("SAHAI_TUTOR_MAX_CONCURRENT", "1"))
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_inflight = 0
_inflight_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["backend"] = build_backend()
    logger.info("tutor backend ready: %s", _state["backend"].describe)
    yield
    _state.clear()


app = FastAPI(title="sahai-tutor", version="0.1.0", lifespan=lifespan)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message] = Field(max_length=64)
    # Reverted 96 -> 192. Dropping the cap was predicted to halve turn time
    # and measured as changing nothing: 209.4s and 208.6s at 96, against
    # 143.7/202.9/123.7s at 192. Replies come back around 30 tokens under
    # either cap, so the model stops well short of both and the cap was never
    # the binding constraint. A lower cap therefore bought no speed and only
    # narrowed the room before truncation.
    max_new_tokens: int = Field(default=192, ge=16, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Serving-only. Never enable for training rollouts: it would edit text the
    # policy sampled, and log-probs would then be taken over tokens the model
    # never produced.
    sanitize: bool = False


class GenerateResponse(BaseModel):
    text: str
    complete: bool
    backend: str
    sanitized: bool = False


@app.get("/health")
async def health() -> dict[str, object]:
    # Not dict[str, str]: the load fields below are an int and a bool, and
    # FastAPI validates the response against this annotation. Leaving it as
    # str made every probe return 500 — the health check failed because of
    # the health check, not because the service was unwell.
    backend = _state.get("backend")
    # Reports load but never fails on it. A busy tutor is working, not broken,
    # and returning non-200 here would make Docker restart the container
    # mid-generation — killing the very turn that caused the load.
    return {
        "status": "ok" if backend else "starting",
        "service": "sahai-tutor",
        "backend": backend.describe if backend else "none",
        "in_flight": _inflight,
        "capacity": MAX_CONCURRENT,
        "busy": _inflight >= MAX_CONCURRENT,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate one tutor turn.

    Deliberately `def`, not `async def`: the body blocks for minutes, and on
    the event loop that starved every other request in this process.
    """
    backend = _state["backend"]
    if not _slots.acquire(blocking=False):
        # 503 with Retry-After, not a queue. The caller's own deadline (the
        # session service allows 200s) is shorter than the job it would be
        # queued behind, so waiting only converts a fast refusal into a slow
        # one and holds a connection open for it.
        raise HTTPException(
            503,
            detail=f"tutor busy: {MAX_CONCURRENT} generation(s) in flight",
            headers={"Retry-After": "30"},
        )
    global _inflight
    with _inflight_lock:
        _inflight += 1
    try:
        result = backend.generate(
            messages=[m.model_dump() for m in req.messages],
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )

        text, sanitized = result.text, False
        if req.sanitize:
            cleaned = strip_code(text)
            if cleaned and cleaned != text:
                text, sanitized = cleaned, True

        return GenerateResponse(
            text=text,
            complete=result.complete,
            backend=backend.describe,
            sanitized=sanitized,
        )
    finally:
        with _inflight_lock:
            _inflight -= 1
        _slots.release()


@app.get("/system-prompt")
async def system_prompt() -> dict[str, str]:
    return {"prompt": RULES}
