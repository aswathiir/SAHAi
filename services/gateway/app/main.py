"""sahai-gateway — the single public entry point.

Everything behind this is on an internal network and is not directly reachable.
The gateway does the things that must happen exactly once per request — CORS,
request ids, rate limiting, and aggregate health — and otherwise forwards.

Auth is a deliberate stub: it establishes the seam and the header contract
without pretending to be a real identity provider.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Curated by scripts/export_problems.py. Deliberately excludes `solution` —
# this file is served straight to the browser, and the tutor's entire job is
# to never reveal the answer, so the answer must not be sitting in a
# fetchable JSON file either.
_PROBLEMS_PATH = Path(__file__).parent / "data" / "problems.json"
PROBLEMS: list[dict] = json.loads(_PROBLEMS_PATH.read_text()) if _PROBLEMS_PATH.exists() else []
PROBLEMS_BY_ID: dict[str, dict] = {p["id"]: p for p in PROBLEMS}

SESSION_URL = os.getenv("SAHAI_SESSION_URL", "http://session:8000")
TRACER_URL = os.getenv("SAHAI_TRACER_URL", "http://tracer:8000")
ASR_URL = os.getenv("SAHAI_ASR_URL", "http://asr:8000")
TUTOR_URL = os.getenv("SAHAI_TUTOR_URL", "http://tutor:8000")
EXECUTOR_URL = os.getenv("SAHAI_EXECUTOR_URL", "http://executor:8000")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("gateway")

# Carries the id from the middleware into the downstream HTTP calls without
# threading it through every handler signature.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

RATE_LIMIT_PER_MIN = int(os.getenv("SAHAI_RATE_LIMIT_PER_MIN", "60"))
_hits: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Must exceed session's own downstream timeout, or the gateway gives up
    # first and hides the real (slower but successful) response.
    app.state.http = httpx.AsyncClient(timeout=270.0)
    yield
    await app.state.http.aclose()


app = FastAPI(title="sahai-gateway", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SAHAI_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request with an id, log it, and pass it downstream.

    Returning the id in a response header is not enough on its own: if it is
    never written to the logs there is nothing to grep for, and if it is never
    forwarded you cannot follow one request across services. Both matter — a
    request that fails in the executor is only findable in the executor's logs.
    """
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    _request_id.set(request_id)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000

    response.headers["x-request-id"] = request_id
    response.headers["x-elapsed-ms"] = f"{elapsed_ms:.1f}"

    logger.info(
        "rid=%s %s %s -> %s %.1fms",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


def _trace() -> dict[str, str]:
    """Headers that carry the request id to the next service."""
    return {"x-request-id": _request_id.get()}


def _rate_limit(learner_id: str) -> None:
    """Fixed-window limiter, per learner, in-process.

    Adequate for a single-instance deployment. A multi-instance one needs this
    in Redis — noted rather than pretended.
    """
    now = time.time()
    window = _hits[learner_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "rate limit exceeded")
    window.append(now)


# Declared once so every endpoint documents the header identically, and so
# Swagger renders it with a description instead of a bare field name.
LearnerHeader = Header(
    default=None,
    alias="x-learner-id",
    description="Who you are. Any string works — this is a stub identity, "
    "not real authentication. Example: 'me'.",
    examples=["me"],
)

# Without this, Swagger labels the 401 'Undocumented' and gives no hint that
# the header is what's missing.
AUTH_RESPONSES = {
    401: {
        "description": "Missing x-learner-id header",
        "content": {
            "application/json": {
                "example": {"detail": "x-learner-id header required"}
            }
        },
    }
}


async def _learner(x_learner_id: str | None) -> str:
    # Stub identity. Real deployment terminates auth here and derives the
    # learner from a verified token instead of trusting a header.
    if not x_learner_id:
        raise HTTPException(401, "x-learner-id header required")
    return x_learner_id


class StartRequest(BaseModel):
    problem_id: str
    problem_title: str = ""
    problem_description: str = ""


class TurnRequest(BaseModel):
    content: str = Field(max_length=8000)


class SubmitRequest(BaseModel):
    code: str = Field(max_length=100_000)
    function_name: str
    test_cases: list[dict]
    skills: list[str] = Field(default_factory=list)


@app.get("/health")
async def health() -> JSONResponse:
    """Aggregate health.

    Only checks services the gateway can actually reach. The executor lives on
    the isolated `sandbox` network with no route here — that is the whole point
    of the isolation — so probing it directly would report `degraded` forever.
    Its health comes back nested inside session's, which is the only service
    with a route to it.
    """
    checks: dict[str, object] = {}
    for name, url in (
        ("session", SESSION_URL), ("tutor", TUTOR_URL),
        ("tracer", TRACER_URL), ("asr", ASR_URL),
    ):
        try:
            r = await app.state.http.get(f"{url}/health", timeout=3.0)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            checks[name] = "ok" if r.status_code == 200 else f"http {r.status_code}"
            # session reports the executor on our behalf.
            if name == "session" and isinstance(body, dict) and "executor" in body:
                checks["executor"] = body["executor"]
        except Exception as exc:  # noqa: BLE001 - report, never raise
            checks[name] = f"unreachable: {type(exc).__name__}"

    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "dependencies": checks},
        status_code=200 if healthy else 503,
    )


@app.post("/v1/sessions", responses=AUTH_RESPONSES)
async def start_session(
    req: StartRequest, x_learner_id: str | None = LearnerHeader
) -> dict:
    learner = await _learner(x_learner_id)
    _rate_limit(learner)
    r = await app.state.http.post(
        f"{SESSION_URL}/sessions",
        json={"learner_id": learner, **req.model_dump()},
        headers=_trace(),
    )
    r.raise_for_status()
    return r.json()


@app.post("/v1/sessions/{session_id}/turns", responses=AUTH_RESPONSES)
async def add_turn(
    session_id: str, req: TurnRequest, x_learner_id: str | None = LearnerHeader
) -> dict:
    learner = await _learner(x_learner_id)
    _rate_limit(learner)
    r = await app.state.http.post(
        f"{SESSION_URL}/sessions/{session_id}/turns",
        json=req.model_dump(),
        headers=_trace(),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.json().get("detail", "session error"))
    return r.json()


@app.post("/v1/sessions/{session_id}/submit", responses=AUTH_RESPONSES)
async def submit(
    session_id: str, req: SubmitRequest, x_learner_id: str | None = LearnerHeader
) -> dict:
    learner = await _learner(x_learner_id)
    _rate_limit(learner)
    r = await app.state.http.post(
        f"{SESSION_URL}/sessions/{session_id}/submit",
        json=req.model_dump(),
        headers=_trace(),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.json().get("detail", "submit error"))
    return r.json()


@app.websocket("/v1/ws/voice/{session_id}")
async def voice_turn(
    websocket: WebSocket, session_id: str, learner_id: str = "me", lang: str = "hi"
) -> None:
    """Voice front-end for an existing tutoring session.

    Record-then-send, not continuous streaming: the client sends one binary
    audio clip per utterance and gets back one JSON reply. It's still a real
    websocket connection (kept open across the whole conversation, not one
    HTTP round trip per turn) — just not frame-by-frame partial transcription,
    which IndicConformer isn't built for anyway (it's a batch model).

    Deliberately reuses the exact same /sessions/{id}/turns call the text
    chat path uses, so voice goes through identical tutor/reward logic — this
    is a new front door onto the existing pipeline, not a parallel one.
    """
    await websocket.accept()
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            try:
                _rate_limit(learner_id)

                asr_resp = await app.state.http.post(
                    f"{ASR_URL}/transcribe",
                    files={"audio": ("clip.webm", audio_bytes, "audio/webm")},
                    headers=_trace(),
                )
                if asr_resp.status_code >= 400:
                    await websocket.send_json(
                        {"error": "transcription_failed", "detail": asr_resp.text}
                    )
                    continue
                transcript = asr_resp.json()["text"]
                if not transcript.strip():
                    await websocket.send_json({"error": "empty_transcript"})
                    continue

                turn_resp = await app.state.http.post(
                    f"{SESSION_URL}/sessions/{session_id}/turns",
                    json={"content": transcript},
                    headers=_trace(),
                )
                if turn_resp.status_code >= 400:
                    await websocket.send_json(
                        {"error": "turn_failed", "detail": turn_resp.text}
                    )
                    continue
                turn_body = turn_resp.json()
                tutor_reply = turn_body["tutor_reply"]

                tts_resp = await app.state.http.post(
                    f"{ASR_URL}/synthesize",
                    json={"text": tutor_reply, "language": lang},
                    headers=_trace(),
                )
                audio_b64 = None
                if tts_resp.status_code < 400:
                    audio_b64 = base64.b64encode(tts_resp.content).decode("ascii")
                # TTS failing (e.g. still on the stub backend) shouldn't lose
                # the turn — text-only is a legitimate degraded response.

                await websocket.send_json(
                    {
                        "transcript": transcript,
                        "tutor_reply": tutor_reply,
                        "status": turn_body["status"],
                        "audio_b64": audio_b64,
                    }
                )
            except httpx.HTTPError as exc:
                await websocket.send_json({"error": "upstream_unreachable", "detail": str(exc)})
            except HTTPException as exc:
                await websocket.send_json({"error": "rate_limited", "detail": exc.detail})
    except WebSocketDisconnect:
        pass


@app.get("/v1/sessions/{session_id}", responses=AUTH_RESPONSES)
async def get_session(
    session_id: str, x_learner_id: str | None = LearnerHeader
) -> dict:
    await _learner(x_learner_id)
    r = await app.state.http.get(
        f"{SESSION_URL}/sessions/{session_id}", headers=_trace()
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, "session not found")
    return r.json()


@app.get("/v1/me/mastery", responses=AUTH_RESPONSES)
async def mastery(x_learner_id: str | None = LearnerHeader) -> dict:
    learner = await _learner(x_learner_id)
    r = await app.state.http.get(
        f"{TRACER_URL}/mastery/{learner}", headers=_trace()
    )
    r.raise_for_status()
    return r.json()


class PlacementIn(BaseModel):
    responses: dict[str, str] = Field(max_length=64)


# Curated tracks over the skills the problem bank actually contains. Ordered
# so each track opens with work that stands alone and ends with work that
# leans on the earlier entries — the sequence is the pedagogy, not decoration.
COURSE_TRACKS = [
    {
        "id": "foundations",
        "title": "Foundations",
        "blurb": "Loops, conditionals and the shapes data comes in.",
        "skills": ["general", "math", "strings"],
    },
    {
        "id": "collections",
        "title": "Collections",
        "blurb": "Arrays and hash maps — the two structures most interviews lean on.",
        "skills": ["arrays", "hash_maps"],
    },
    {
        "id": "ordering",
        "title": "Ordering & Search",
        "blurb": "Sorting, two-pointer scans and the heap.",
        "skills": ["sorting", "two_pointers", "heaps"],
    },
    {
        "id": "recursive",
        "title": "Recursive Thinking",
        "blurb": "Trees, recursion and problems that fold into themselves.",
        "skills": ["recursion", "trees"],
    },
    {
        "id": "bits",
        "title": "Bit Manipulation",
        "blurb": "Working one bit at a time.",
        "skills": ["bit_manipulation"],
    },
]


def _streak(days: list[str]) -> tuple[int, int]:
    """(current, longest) run of consecutive active days.

    `days` is a sorted list of ISO dates. The current streak counts back from
    today and tolerates today being empty — a learner mid-morning has not
    broken a streak they may still extend, and showing it as 0 until they
    practise reads as punishment for the time of day.
    """
    if not days:
        return 0, 0

    as_dates = sorted({date.fromisoformat(d) for d in days})

    longest = run = 1
    for prev, cur in zip(as_dates, as_dates[1:]):
        run = run + 1 if (cur - prev).days == 1 else 1
        longest = max(longest, run)

    today = datetime.now(timezone.utc).date()
    gap = (today - as_dates[-1]).days
    if gap > 1:
        return 0, longest

    current = 1
    for prev, cur in zip(reversed(as_dates[:-1]), reversed(as_dates[1:])):
        if (cur - prev).days != 1:
            break
        current += 1
    return current, longest


@app.post("/v1/me/placement", responses=AUTH_RESPONSES)
async def set_placement(
    body: PlacementIn, x_learner_id: str | None = LearnerHeader
) -> dict:
    """Record a self-assessment so the first assignment fits the learner.

    Without this every learner starts at BKT's flat `p_init` for every skill,
    and the opening session picks problems as if they knew nothing about
    anything.
    """
    learner = await _learner(x_learner_id)
    r = await app.state.http.post(
        f"{TRACER_URL}/placement",
        json={"learner_id": learner, "responses": body.responses},
        headers=_trace(),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:200])
    return r.json()


@app.get("/v1/me/progress", responses=AUTH_RESPONSES)
async def progress(x_learner_id: str | None = LearnerHeader) -> dict:
    """Everything the progress view needs, in one round trip.

    Derived entirely from the append-only observation log rather than new
    tables: the log already carries timestamp, skill, correctness and item, so
    streaks and activity are a projection of it. Anything stored separately
    would be a second source of truth that could drift from the history.
    """
    learner = await _learner(x_learner_id)

    obs_resp, mastery_resp = await asyncio.gather(
        app.state.http.get(
            f"{TRACER_URL}/observations/{learner}", params={"limit": 5000},
            headers=_trace(),
        ),
        app.state.http.get(f"{TRACER_URL}/mastery/{learner}", headers=_trace()),
    )
    obs_resp.raise_for_status()
    mastery_resp.raise_for_status()
    observations = obs_resp.json()
    skills = mastery_resp.json().get("skills", {})

    by_day: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "solved": 0})
    solved_problems: set[str] = set()
    attempted_problems: set[str] = set()

    for o in observations:
        day = (o.get("observed_at") or "")[:10]
        if day:
            by_day[day]["total"] += 1
            by_day[day]["solved"] += int(bool(o.get("correct")))
        pid = o.get("problem_id")
        if pid:
            attempted_problems.add(pid)
            if o.get("correct"):
                solved_problems.add(pid)

    current, longest = _streak(sorted(by_day))

    # A skill counts as covered once its posterior clears the top of the ZPD
    # band — the same threshold the sampler uses to stop offering it.
    tracks = []
    for track in COURSE_TRACKS:
        problems = [p for p in PROBLEMS if set(p["skills"]) & set(track["skills"])]
        done = [p for p in problems if p["id"] in solved_problems]
        covered = [s for s in track["skills"] if skills.get(s, 0.0) >= 0.7]
        tracks.append(
            {
                **track,
                "total": len(problems),
                "solved": len(done),
                "skills_covered": len(covered),
                "mastery": (
                    sum(skills.get(s, 0.0) for s in track["skills"]) / len(track["skills"])
                    if track["skills"] else 0.0
                ),
            }
        )

    return {
        "learner_id": learner,
        "solved": len(solved_problems),
        "attempted": len(attempted_problems),
        "total_problems": len(PROBLEMS),
        "streak_current": current,
        "streak_longest": longest,
        "active_days": len(by_day),
        "activity": [{"date": d, **v} for d, v in sorted(by_day.items())],
        "skills": skills,
        "tracks": tracks,
        "placed": bool(skills),
    }


@app.get("/v1/courses")
async def courses() -> list[dict]:
    """Track catalogue with problem counts. No auth — browsing is harmless."""
    return [
        {
            **t,
            "total": len([p for p in PROBLEMS if set(p["skills"]) & set(t["skills"])]),
        }
        for t in COURSE_TRACKS
    ]


@app.get("/v1/problems")
async def list_problems() -> list[dict]:
    """Catalogue for the assessment interface. No auth — browsing is harmless.

    Trimmed to what a picker needs; full detail (test cases, function name)
    comes from /v1/problems/{id} once a student actually starts a session.
    """
    return [
        {"id": p["id"], "title": p["title"], "difficulty": p["difficulty"], "skills": p["skills"]}
        for p in PROBLEMS
    ]


@app.get("/v1/problems/{problem_id}")
async def get_problem(problem_id: str) -> dict:
    problem = PROBLEMS_BY_ID.get(problem_id)
    if problem is None:
        raise HTTPException(404, "problem not found")
    return problem


# Mounted last so it never shadows an API route above — StaticFiles matches
# on whatever wasn't already claimed, with html=True serving index.html at "/".
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
