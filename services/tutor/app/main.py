"""sahai-tutor — inference for the trained tutor policy.

Stateless. Knows nothing about learners, sessions, or rewards; it turns a
message list into a tutor turn. Session state lives in sahai-session, learner
state in sahai-tracer.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sahai_core.generation import strip_code

from .backends import TutorBackend, build_backend

logger = logging.getLogger(__name__)

TUTOR_SYSTEM_PROMPT = (
    "You are a tutor. You help students think, NOT give answers.\n\n"
    "STRICT RULES:\n"
    "- NEVER write code. No code blocks. No function definitions. No pseudocode.\n"
    "- NEVER show the solution or any part of it.\n"
    "- Ask ONE question per turn to guide the student's thinking.\n"
    "- Keep responses to 2-3 sentences maximum.\n"
    "- Never end a turn with a colon promising something you do not then say.\n"
    "- Match the student's language style.\n\n"
    "GOOD: 'What data structure lets you check if you have seen a character in O(1)?'\n"
    "BAD:  'Here is the solution: def func(): ...'"
)

_state: dict[str, TutorBackend] = {}


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
async def health() -> dict[str, str]:
    backend = _state.get("backend")
    return {
        "status": "ok" if backend else "starting",
        "service": "sahai-tutor",
        "backend": backend.describe if backend else "none",
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    backend = _state["backend"]
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


@app.get("/system-prompt")
async def system_prompt() -> dict[str, str]:
    return {"prompt": TUTOR_SYSTEM_PROMPT}
