"""sahai-gateway — the single public entry point.

Everything behind this is on an internal network and is not directly reachable.
The gateway does the things that must happen exactly once per request — CORS,
request ids, rate limiting, and aggregate health — and otherwise forwards.

Identity is verified here and nowhere else: a bearer token in, a learner id
out, forwarded inward on an internal network. See `app/identity.py` for what
that does and does not promise.
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

from app import identity

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


# One deadline ladder for the whole request path, outermost first:
#
#   gateway  270s   ->  session  240s  ->  tutor generate  200s
#
# Each hop must outlast the one it calls. If that order ever inverts, the outer
# caller gives up on a request the inner one is about to answer successfully —
# the learner sees a 500, and the work is thrown away rather than returned.
# Stated here as constants instead of three unrelated literals in three files,
# because the invariant is the ordering, and ordering is invisible when the
# numbers live apart. `tests/test_deadline_ladder.py` asserts it.
GATEWAY_TIMEOUT_S = 270.0
SESSION_TIMEOUT_S = 240.0
# Speech synthesis is a bonus on top of a reply the learner already has, so it
# gets its own short budget rather than the request-path one. Measured on this
# stack, indic-parler-tts needs many minutes for a single short sentence on
# CPU — Docker Desktop passes no GPU through — and waiting that out would turn
# a finished answer into a hung turn. Env-tunable so a GPU deployment, where
# this is a second or two, can raise it.
TTS_BUDGET_S = float(os.getenv("SAHAI_TTS_BUDGET_S", "45"))
# Health and mastery are on the request path but must never dominate it: a slow
# tracer should cost a tailored hint, not the turn.
PROBE_TIMEOUT_S = 3.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Identity must exist before the first request can be authenticated, and a
    # gateway that cannot authenticate has nothing useful to serve — so this is
    # allowed to fail startup rather than degrade.
    await identity.create_tables()
    app.state.http = httpx.AsyncClient(
        timeout=GATEWAY_TIMEOUT_S,
        # Bounded pool. The default is unbounded, so a burst opens a connection
        # per in-flight request and the failure mode is file-descriptor
        # exhaustion in the gateway rather than honest queueing.
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
    )
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


def _trace(learner: str | None = None) -> dict[str, str]:
    """Headers passed to the next service: the request id, and who is asking.

    The learner is forwarded because session now checks that the caller owns
    the session before reading or writing it. Identity is still the stub the
    gateway uses — a header, not a verified token — but without forwarding it
    the check has nothing to compare against.
    """
    headers = {"x-request-id": _request_id.get()}
    if learner:
        headers["x-learner-id"] = learner
    return headers


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
#
# The name `LearnerHeader` is kept from when this *was* `x-learner-id`, so that
# every endpoint signature reads the same; what it carries is now a token.
LearnerHeader = Header(
    default=None,
    alias="authorization",
    description="`Bearer <token>` — the token returned once by "
    "POST /v1/auth/register. Not a learner id: sending one is no longer "
    "accepted.",
    examples=["Bearer 0oEXAMPLEtokenGOEShere"],
)

# Without this, Swagger labels the 401 'Undocumented' and gives no hint about
# what is missing.
AUTH_RESPONSES = {
    401: {
        "description": "Missing or invalid bearer token",
        "content": {
            "application/json": {
                "example": {"detail": "invalid or expired token"}
            }
        },
    }
}


def _bearer(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "Authorization: Bearer <token> required")
    return token.strip()


async def _learner(authorization: str | None) -> str:
    """The authenticated learner id, or 401.

    This used to return `x_learner_id` unchanged — a value the browser chose.
    Every ownership check in session and tracer compares against what this
    returns, so all of them were resting on the client's honesty: sending
    another learner's id was enough to read their sessions, mastery and
    submissions. The checks were right; the identity under them was not.

    One message for "no token" and "wrong token" on purpose. Distinguishing
    them tells an attacker which half of a guess was correct.
    """
    token = _bearer(authorization)
    async with identity.SessionLocal() as db:
        row = await identity.resolve(db, token)
    if row is None:
        raise HTTPException(401, "invalid or expired token")
    return row.learner_id


class StartRequest(BaseModel):
    problem_id: str
    problem_title: str = ""
    problem_description: str = ""


class TurnRequest(BaseModel):
    content: str = Field(max_length=8000)
    # Which problem the learner is on, so the gateway can look up its skills.
    # Only the id is trusted from the client: skills are resolved from the
    # gateway's own bank, never taken from the request, or a caller could
    # claim whatever skills produced the hint they wanted.
    problem_id: str | None = Field(default=None, max_length=64)


# Deliberately *not* the ZPD cut points (0.3/0.7), despite the obvious appeal
# of one set of numbers. Those govern which problems to offer; these govern how
# to pitch a hint, and the two answer different questions.
#
# Tying them made the feature inert. Placement is capped inside [0.15, 0.65] so
# that even a confident learner still gets offered work, and every level it
# produces — 0.15 / 0.30 / 0.50 / 0.65 — then fell in the mid band except the
# lowest. A learner who had just imported 320 solved problems got no guidance
# at all, which is the exact case the feature exists for.
#
# These cuts make all four placement levels say something: "none" and "seen"
# read as new, "confident" reads as solid, "practiced" stays quiet because
# mid-band is where the tutor should already be pitching.
UNTRACKED_SKILL = "general"

MASTERY_NEW = 0.35
MASTERY_SOLID = 0.65


def _learner_context(skills: list[str], mastery: dict[str, float]) -> str:
    """Turn BKT posteriors into an instruction the model can act on.

    Deliberately *not* the raw numbers. "arrays: 0.42" gives a language model
    nothing to do — it has no calibration for what 0.42 should change about a
    hint. Naming the pedagogical move instead ("build the idea" vs "assume it")
    is the part that actually alters the next turn.

    Only skills the current problem touches are mentioned. The learner's full
    profile would be mostly irrelevant to any one problem and would crowd a
    system prompt whose own rule is 2-3 sentences per reply.
    """
    if not skills:
        return ""

    # The tutor says these to a learner, and `hash_maps` next to the prior-work
    # block's "a hash map" reads like two different things.
    def human(skill: str) -> str:
        return skill.replace("_", " ")

    new, solid = [], []
    for skill in sorted(set(skills)):
        # Unseen skills have no posterior yet; treat them as new rather than
        # inventing a middling one.
        value = mastery.get(skill, 0.0)
        if value < MASTERY_NEW:
            new.append(skill)
        elif value >= MASTERY_SOLID:
            solid.append(skill)

    lines = []
    if new:
        lines.append(
            f"- New to {', '.join(human(s) for s in new)}. "
            "Build the idea before asking them to apply it."
        )
    if solid:
        lines.append(
            f"- Solid on {', '.join(human(s) for s in solid)}. "
            "Do not re-explain it; push on what is new here."
        )
    # Everything in between is exactly where the tutor should already be
    # pitching, so saying so would just spend tokens agreeing with itself.
    if not lines:
        return ""
    return "WHAT THIS LEARNER ALREADY KNOWS:\n" + "\n".join(lines)


async def _mastery_for(learner: str) -> dict[str, float]:
    """Current posteriors, or {} if the tracer cannot answer.

    Fails open on purpose: personalisation is an improvement to a turn, not a
    precondition for one. A tracer outage should cost the learner a tailored
    hint, not the ability to ask a question.
    """
    try:
        r = await app.state.http.get(
            f"{TRACER_URL}/mastery/{learner}", headers=_trace(), timeout=PROBE_TIMEOUT_S
        )
        r.raise_for_status()
        return r.json().get("skills", {})
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the turn
        logger.warning("mastery lookup failed for %s: %s", learner, type(exc).__name__)
        return {}


async def _context_for(learner: str, problem: dict | None) -> str:
    """The full personalisation block: who they are, and what they have done.

    Both the typed and the spoken path call this. They used to build the block
    separately and drifted apart once already — voice posted `{"content": ...}`
    and nothing else, so speaking to the tutor silently opted out of
    personalisation entirely.
    """
    if not problem:
        return ""
    skills = problem.get("skills", [])
    context = _learner_context(skills, await _mastery_for(learner))
    prior = _prior_work_lines(
        await _prior_work_for(learner, skills, exclude_slug=problem.get("id", ""))
    )
    blocks = [b for b in (context, "\n".join(prior)) if b]
    return "\n\n".join(blocks)


async def _prior_work_for(learner: str, skills: list[str], exclude_slug: str = "") -> list[dict]:
    """Problems this learner already solved in these skills, or [].

    Fails open like `_mastery_for`, for the same reason.

    `exclude_slug` drops the problem currently being worked on. Everything here
    is a problem the learner has *already solved*, so if this one is among them
    the "technique they used" is the answer to the question in front of them.
    The importer stores no code, so this is not a leak of source — but it is
    still the one hint that ends the lesson, and it is cheap to withhold.
    """
    if not skills:
        return []
    try:
        r = await app.state.http.get(
            f"{TRACER_URL}/prior_work/{learner}",
            params={"skills": ",".join(skills), "limit": 4},
            headers=_trace(),
            timeout=PROBE_TIMEOUT_S,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the turn
        logger.warning("prior work lookup failed for %s: %s", learner, type(exc).__name__)
        return []
    return [i for i in items if i.get("slug") != exclude_slug][:2]


def _prior_work_lines(items: list[dict], today: date | None = None) -> list[str]:
    """Render prior work as something the tutor can teach *from*.

    Names the problem and the technique, never the code — that is the whole
    reason reading a solutions repo is safe (see `PriorWork` in the tracer).
    The instruction is to *ask*, not to tell: "you used a hash map in Two Sum"
    handed over as a statement is a hint; asked back as a question is the
    analogy doing the teaching.

    Undated work says "before" rather than inventing a time. The importer
    cannot date bulk-imported commits, and a tutor confidently saying "three
    weeks ago" about something it cannot place is worse than vague.
    """
    today = today or datetime.now(timezone.utc).date()
    lines = []
    for item in items:
        when = "before"
        if item.get("dated"):
            try:
                days = (today - date.fromisoformat(item["solved_on"])).days
            except (ValueError, TypeError):
                days = None
            if days is not None and days >= 0:
                weeks = days // 7
                when = (
                    "this week" if weeks < 1
                    else f"{weeks} weeks ago" if weeks < 9
                    else f"{max(days // 30, 1)} months ago"
                )
        lines.append(f"- They solved \"{item['title']}\" with {item['technique']}, {when}.")
    if not lines:
        return []
    return [
        "WHAT THEY HAVE ALREADY DONE (ask them to connect it, do not connect it for them):"
    ] + lines


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
            r = await app.state.http.get(f"{url}/health", timeout=PROBE_TIMEOUT_S)
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


# ---------------------------------------------------------------- identity


class RegisterIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=identity.MAX_DISPLAY_NAME)


REGISTRATIONS_PER_HOUR = int(os.getenv("SAHAI_REGISTRATIONS_PER_HOUR", "10"))
_registrations: dict[str, deque[float]] = defaultdict(deque)


def _limit_registrations(client_ip: str) -> None:
    """Registration cannot be limited per learner — it is what creates one.

    Keyed on the client address instead, and on a one-hour window rather than
    the request path's one-minute one, because minting credentials is not
    something a real person does repeatedly. Without this the endpoint is an
    unauthenticated row-insert loop.
    """
    now = time.time()
    window = _registrations[client_ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= REGISTRATIONS_PER_HOUR:
        raise HTTPException(429, "too many registrations from this address")
    window.append(now)


@app.post("/v1/auth/register", status_code=201)
async def register(req: RegisterIn, request: Request) -> dict:
    """Mint a learner and their one and only token.

    The token is in this response and nowhere else — the row stores a SHA-256
    of it. There is no endpoint that returns it again, deliberately: an
    endpoint that can re-read a credential is an endpoint that can leak one.
    Losing it means `scripts/mint_token.py`.

    The learner id is generated here and is **not** taken from the request.
    Letting a caller name its own id would let it claim an id that already
    holds someone's history, which is the same hole this replaces.
    """
    _limit_registrations(request.client.host if request.client else "unknown")
    async with identity.SessionLocal() as db:
        learner_id, token = await identity.register(db, req.display_name)
    logger.info("rid=%s registered learner %s", _request_id.get(), learner_id)
    return {
        "learner_id": learner_id,
        "display_name": req.display_name.strip(),
        "token": token,
        "note": "Save this token. It is not shown again and cannot be recovered.",
    }


@app.get("/v1/auth/me", responses=AUTH_RESPONSES)
async def whoami(authorization: str | None = LearnerHeader) -> dict:
    """Who the presented token belongs to.

    The client uses this to decide whether a token it has stored is still
    good, rather than discovering it on the first real request and having to
    unwind a half-started session.
    """
    token = _bearer(authorization)
    async with identity.SessionLocal() as db:
        row = await identity.resolve(db, token)
    if row is None:
        raise HTTPException(401, "invalid or expired token")
    return {
        "learner_id": row.learner_id,
        "display_name": row.display_name,
        "created_at": row.created_at.isoformat(),
    }


# ---------------------------------------------------------------- sessions


@app.post("/v1/sessions", responses=AUTH_RESPONSES)
async def start_session(
    req: StartRequest, authorization: str | None = LearnerHeader
) -> dict:
    learner = await _learner(authorization)
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
    session_id: str, req: TurnRequest, authorization: str | None = LearnerHeader
) -> dict:
    learner = await _learner(authorization)
    _rate_limit(learner)

    # Resolve skills from the gateway's own bank rather than the request, so a
    # caller cannot name skills to steer the hint they get.
    problem = PROBLEMS_BY_ID.get(req.problem_id or "")
    context = await _context_for(learner, problem)

    payload = req.model_dump(exclude={"problem_id"})
    payload["learner_context"] = context
    # The full statement, not the truncated title the session row stores.
    if problem:
        payload["problem_statement"] = problem.get("description") or problem.get("title", "")

    r = await app.state.http.post(
        f"{SESSION_URL}/sessions/{session_id}/turns",
        json=payload,
        headers=_trace(learner),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.json().get("detail", "session error"))
    return r.json()


@app.post("/v1/sessions/{session_id}/submit", responses=AUTH_RESPONSES)
async def submit(
    session_id: str, req: SubmitRequest, authorization: str | None = LearnerHeader
) -> dict:
    learner = await _learner(authorization)
    _rate_limit(learner)
    r = await app.state.http.post(
        f"{SESSION_URL}/sessions/{session_id}/submit",
        json=req.model_dump(),
        headers=_trace(learner),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.json().get("detail", "submit error"))
    return r.json()


@app.websocket("/v1/ws/voice/{session_id}")
async def voice_turn(
    websocket: WebSocket,
    session_id: str,
    lang: str = "hi",
    problem_id: str | None = None,
) -> None:
    """Voice front-end for an existing tutoring session.

    Record-then-send, not continuous streaming: the client sends one binary
    audio clip per utterance and gets back one JSON reply. It's still a real
    websocket connection (kept open across the whole conversation, not one
    HTTP round trip per turn) — just not frame-by-frame partial transcription,
    which IndicConformer isn't built for anyway (it's a batch model).

    Deliberately reuses the exact same /sessions/{id}/turns call the text
    chat path uses, so voice goes through identical tutor/reward logic — this
    is a new front door onto the existing pipeline, not a parallel one. That
    includes the personalisation: `problem_id` is resolved to skills and
    mastery here exactly as it is for a typed turn, because a voice turn that
    quietly skipped it would be a second, worse tutor wearing the same face.

    Each stage reports as it finishes rather than the client waiting out the
    whole pipeline in silence. That matters most here: ASR takes seconds but
    generation can take minutes on CPU, and a learner who has just spoken has
    no idea whether they were heard. Sending the transcript the moment it
    exists turns a blind wait into a conversation.

    **The first frame must be `{"token": "..."}`.** This took `learner_id` as a
    *query parameter defaulting to `"me"`* — so opening
    `ws://host/v1/voice?session_id=…&learner_id=<someone>` was enough to speak
    into another learner's session and have the reply written to their
    transcript. The ownership check in the session service could not help: it
    compares against the id the gateway forwards, and the gateway was
    forwarding whatever the URL said.

    Authenticating on the first frame rather than in the query string is
    deliberate. Browsers cannot set headers on a WebSocket handshake, so the
    usual alternative is `?token=…` — and query strings land in access logs,
    proxy logs and browser history, which is the last place a long-lived
    credential should be. The frame costs one extra round trip before the first
    utterance and nothing after that.
    """
    await websocket.accept()

    try:
        hello = await websocket.receive_json()
    except Exception:  # noqa: BLE001 - a non-JSON first frame is a failed handshake
        await websocket.close(code=1008, reason="auth frame required")
        return
    try:
        learner_id = await _learner(f"Bearer {hello.get('token', '')}")
    except HTTPException:
        # 1008 = policy violation. The client cannot read an HTTP status here,
        # so the close code has to carry it.
        await websocket.close(code=1008, reason="invalid or expired token")
        return
    await websocket.send_json({"stage": "authenticated", "learner_id": learner_id})

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

                # Heard you. Sent before generation starts, not after it ends.
                await websocket.send_json(
                    {"stage": "transcribed", "transcript": transcript}
                )

                problem = PROBLEMS_BY_ID.get(problem_id or "")
                payload = {"content": transcript}
                if problem:
                    payload["learner_context"] = await _context_for(learner_id, problem)
                    payload["problem_statement"] = (
                        problem.get("description") or problem.get("title", "")
                    )

                await websocket.send_json({"stage": "thinking"})
                turn_resp = await app.state.http.post(
                    f"{SESSION_URL}/sessions/{session_id}/turns",
                    json=payload,
                    headers=_trace(learner_id),
                )
                if turn_resp.status_code >= 400:
                    await websocket.send_json(
                        {"error": "turn_failed", "detail": turn_resp.text}
                    )
                    continue
                turn_body = turn_resp.json()
                tutor_reply = turn_body["tutor_reply"]

                await websocket.send_json({"stage": "speaking", "tutor_reply": tutor_reply})

                # Bounded, and never fatal. The learner already has the reply in
                # text from the "speaking" frame above; audio that arrives late
                # or not at all costs them nothing they were waiting on.
                audio_b64 = None
                try:
                    tts_resp = await app.state.http.post(
                        f"{ASR_URL}/synthesize",
                        json={"text": tutor_reply, "language": lang},
                        headers=_trace(),
                        timeout=TTS_BUDGET_S,
                    )
                    if tts_resp.status_code < 400:
                        audio_b64 = base64.b64encode(tts_resp.content).decode("ascii")
                except httpx.HTTPError as exc:
                    logger.warning("tts unavailable, replying text-only: %s",
                                   type(exc).__name__)

                await websocket.send_json(
                    {
                        "transcript": transcript,
                        "tutor_reply": tutor_reply,
                        "status": turn_body["status"],
                        "audio_b64": audio_b64,
                        "speech": audio_b64 is not None,
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
    session_id: str, authorization: str | None = LearnerHeader
) -> dict:
    learner = await _learner(authorization)
    r = await app.state.http.get(
        f"{SESSION_URL}/sessions/{session_id}", headers=_trace(learner)
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, "session not found")
    return r.json()


@app.get("/v1/me/mastery", responses=AUTH_RESPONSES)
async def mastery(authorization: str | None = LearnerHeader) -> dict:
    learner = await _learner(authorization)
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
        "blurb": "Arrays, hash maps and stacks — the structures interviews lean on most.",
        "skills": ["arrays", "hash_maps", "stacks"],
    },
    {
        "id": "ordering",
        "title": "Ordering & Search",
        "blurb": "Sorting, binary search, two-pointer scans and the heap.",
        "skills": ["sorting", "binary_search", "two_pointers", "heaps"],
    },
    {
        "id": "recursive",
        "title": "Recursive Thinking",
        "blurb": "Trees, graphs and problems that fold into themselves.",
        "skills": ["recursion", "trees", "graphs"],
    },
    {
        "id": "optimisation",
        "title": "Optimisation",
        "blurb": "Overlapping subproblems, and paying for each one only once.",
        "skills": ["dynamic_programming"],
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


def _next_up(skills: dict[str, float], solved: set[str]) -> dict | None:
    """The weakest skill the learner can actually practise right now.

    Weakest, not "next in a syllabus": the ZPD sampler already works on
    mastery, so the recommendation and the selector agree rather than pulling
    in different directions.

    Skills with nothing left unsolved are excluded — telling someone their
    weakest area is heaps is useless if all four heaps problems are done, and
    it would keep saying so forever.
    """
    available: dict[str, int] = defaultdict(int)
    for problem in PROBLEMS:
        if problem["id"] in solved:
            continue
        for skill in problem["skills"]:
            # "general" is what _tag_skills returns when it recognises nothing,
            # so it is not in SKILL_KEYWORDS and placement never asks about it.
            # It therefore has no posterior, scores 0.0, and would win this
            # comparison for every learner forever — recommending the one tag
            # nobody can be weak at.
            if skill == UNTRACKED_SKILL:
                continue
            available[skill] += 1
    if not available:
        return None

    # Unseen skills have no posterior; 0.0 is the honest prior, and it also
    # puts "never touched" ahead of "touched and weak", which is the right
    # order to work in.
    skill = min(available, key=lambda s: (skills.get(s, 0.0), s))
    return {
        "skill": skill,
        "mastery": skills.get(skill, 0.0),
        "unsolved": available[skill],
        "href": f"/?skill={skill}",
    }


@app.post("/v1/me/placement", responses=AUTH_RESPONSES)
async def set_placement(
    body: PlacementIn, authorization: str | None = LearnerHeader
) -> dict:
    """Record a self-assessment so the first assignment fits the learner.

    Without this every learner starts at BKT's flat `p_init` for every skill,
    and the opening session picks problems as if they knew nothing about
    anything.
    """
    learner = await _learner(authorization)
    r = await app.state.http.post(
        f"{TRACER_URL}/placement",
        json={"learner_id": learner, "responses": body.responses},
        headers=_trace(),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:200])
    return r.json()


class PriorWorkIn(BaseModel):
    """Imported evidence of problems solved before this system saw the learner.

    There is deliberately no field for source code. That is the whole design
    decision behind reading a solutions repo at all — the importer parses the
    code and discards it, and what crosses this boundary is a problem title and
    a technique name. A `code` field here would put the answer one careless
    prompt change away from the tutor, which is what the entire reward function
    exists to prevent.
    """

    items: list[dict] = Field(default_factory=list, max_length=2000)


@app.post("/v1/me/prior_work", responses=AUTH_RESPONSES)
async def set_prior_work(
    body: PriorWorkIn, authorization: str | None = LearnerHeader
) -> dict:
    """Replace the caller's imported prior work.

    The learner is the authenticated one, never a field in the body — the
    importer cannot write to somebody else's record by naming them.
    """
    learner = await _learner(authorization)
    allowed = {"slug", "title", "skill", "technique", "solved_on", "dated"}
    items = [{k: v for k, v in item.items() if k in allowed} for item in body.items]
    r = await app.state.http.post(
        f"{TRACER_URL}/prior_work",
        json={"learner_id": learner, "items": items},
        headers=_trace(),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:200])
    return r.json()


@app.get("/v1/me/progress", responses=AUTH_RESPONSES)
async def progress(authorization: str | None = LearnerHeader) -> dict:
    """Everything the progress view needs, in one round trip.

    Derived entirely from the append-only observation log rather than new
    tables: the log already carries timestamp, skill, correctness and item, so
    streaks and activity are a projection of it. Anything stored separately
    would be a second source of truth that could drift from the history.
    """
    learner = await _learner(authorization)

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

    tracks = []
    for track in COURSE_TRACKS:
        problems = [p for p in PROBLEMS if set(p["skills"]) & set(track["skills"])]
        done = [p for p in problems if p["id"] in solved_problems]
        # MASTERY_SOLID, not a separate 0.7. Placement caps confidence at
        # exactly 0.65, so a hardcoded 0.7 meant a learner the tutor already
        # treats as solid on a skill still read as "0 skills covered" here —
        # two answers to the same question in one response.
        covered = [s for s in track["skills"] if skills.get(s, 0.0) >= MASTERY_SOLID]
        # `solved` and `mastery` answer different questions and a card that
        # shows them side by side reads as a contradiction: "0/31 solved, 60%
        # mastery" looks broken. `solved` is work done in this bank; `mastery`
        # is the current belief about the skill, which can come from a
        # placement survey or an imported solved-problem repo without a single
        # problem having been worked here. `estimate_only` says which case this
        # is so the page can label it instead of implying progress.
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
                "estimate_only": not done,
            }
        )

    # The one thing the page could not answer: what should I do now. Everything
    # else here describes the past. Computed server-side because the gateway is
    # the only place that holds both the posteriors and the bank — deciding it
    # in the browser would mean shipping the bank's skill counts to it and
    # keeping two copies of the rule in step.
    next_up = _next_up(skills, solved_problems)

    return {
        "learner_id": learner,
        "next_up": next_up,
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


class DiagnosticAnswer(BaseModel):
    problem_id: str = Field(max_length=64)
    code: str = Field(max_length=100_000)


# How many problems a diagnostic asks. Small on purpose: this is meant to be
# finished in one sitting, and an abandoned diagnostic produces no evidence at
# all. Four spans the skill groups without becoming an exam.
DIAGNOSTIC_SIZE = 4


@app.get("/v1/me/sessions", responses=AUTH_RESPONSES)
async def my_sessions(authorization: str | None = LearnerHeader) -> list[dict]:
    """Unfinished conversations, newest first.

    Sessions have always survived a restart — they are rows in Postgres — but
    nothing surfaced one, so leaving mid-dialogue lost it in practice while the
    data sat there untouched.

    Fails open like the mastery lookup: an unavailable session service should
    cost the learner this list, not the page.
    """
    learner = await _learner(authorization)
    try:
        r = await app.state.http.get(
            f"{SESSION_URL}/sessions",
            params={"learner_id": learner, "limit": 5},
            headers=_trace(),
            timeout=PROBE_TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the page
        logger.warning("resumable lookup failed for %s: %s", learner, type(exc).__name__)
        return []


@app.get("/v1/diagnostic", responses=AUTH_RESPONSES)
async def diagnostic(authorization: str | None = LearnerHeader) -> dict:
    """Pick a short set of problems that will actually tell us something.

    Spread across *different* skills, because four problems on arrays measures
    arrays four times and leaves the other fourteen skills exactly as unknown
    as before. Weakest-known skills first, for the same reason the "start here"
    card uses them: that is where the current estimate is least trustworthy.

    Mid-difficulty within each skill, not hardest. An item only discriminates
    near the learner's ability: four difficulty-5 problems mostly produce four
    failures, which says "not expert" and nothing else. Picking the middle of
    each skill's range is the best default when the ability is exactly what is
    being measured.
    """
    learner = await _learner(authorization)
    mastery = await _mastery_for(learner)

    by_skill: dict[str, list[dict]] = defaultdict(list)
    for problem in PROBLEMS:
        for skill in problem["skills"]:
            if skill != UNTRACKED_SKILL:
                by_skill[skill].append(problem)
    if not by_skill:
        return {"problems": [], "size": 0}

    # Least-known first; an unseen skill is 0.0, which is where a diagnostic is
    # worth the most.
    order = sorted(by_skill, key=lambda s: (mastery.get(s, 0.0), s))

    chosen: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for skill in order:
        if len(chosen) >= DIAGNOSTIC_SIZE:
            break
        pool = [p for p in by_skill[skill] if p["id"] not in seen]
        if not pool:
            continue
        pool.sort(key=lambda p: p["difficulty"])
        problem = pool[len(pool) // 2]
        # Carry the skill this item was picked to probe. A problem tagged
        # ["arrays", "sorting", "heaps"] chosen *for* heaps still lists arrays
        # first, so the tags alone do not say what is being measured — and the
        # learner deserves to know which gap each question is aimed at.
        chosen.append((skill, problem))
        seen.add(problem["id"])

    return {
        "size": len(chosen),
        "problems": [
            {
                "id": p["id"],
                "title": p.get("description") or p["title"],
                "difficulty": p["difficulty"],
                "skills": p["skills"],
                "targets": target,
                "function_name": p["function_name"],
                "function_signature": p.get("function_signature", ""),
            }
            for target, p in chosen
        ],
    }


@app.post("/v1/diagnostic/grade", responses=AUTH_RESPONSES)
async def diagnostic_grade(
    body: DiagnosticAnswer, authorization: str | None = LearnerHeader
) -> dict:
    """Grade one diagnostic answer and record it as a real observation.

    Test cases come from the gateway's own bank rather than the request. They
    are the grading criteria — accepting them from the client would let a
    caller mark their own work.
    """
    learner = await _learner(authorization)
    _rate_limit(learner)

    problem = PROBLEMS_BY_ID.get(body.problem_id)
    if problem is None:
        raise HTTPException(404, "problem not found")

    r = await app.state.http.post(
        f"{SESSION_URL}/diagnostic/grade",
        json={
            "learner_id": learner,
            "problem_id": problem["id"],
            "code": body.code,
            "function_name": problem["function_name"],
            "test_cases": problem["test_cases"],
            "skills": problem["skills"],
        },
        headers=_trace(),
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:200])
    return r.json()


@app.get("/v1/problems")
async def list_problems() -> list[dict]:
    """Catalogue for the assessment interface. No auth — browsing is harmless.

    Trimmed to what a picker needs; full detail (test cases, function name)
    comes from /v1/problems/{id} once a student actually starts a session.
    """
    # `title` is the exporter's 80-char truncation of the statement, so for 43
    # of 89 problems it ends mid-word ("...inscribed in th"). The picker shows
    # this text, so it must be the whole sentence — the learner is choosing
    # what to work on and cannot choose what they cannot read.
    return [
        {
            "id": p["id"],
            "title": p.get("description") or p["title"],
            "difficulty": p["difficulty"],
            "skills": p["skills"],
        }
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
