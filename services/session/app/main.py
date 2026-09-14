"""sahai-session — tutoring session orchestration.

This is the production counterpart to the trainer's `DialogueEngine`. The
difference is control flow, and it is the reason the two cannot share an
implementation:

    training    engine drives both sides in a loop, start to finish, offline
    production  a real learner sends one turn and waits; the service holds the
                conversation across requests and must survive a restart

So the engine becomes a state machine over a persisted transcript. Everything
downstream — tutor, executor, tracer — is called over HTTP and is stateless
with respect to this conversation.
"""

from __future__ import annotations

import contextvars
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sahai_core import (
    TERMINATION_PHRASES,
    Dialogue,
    RuleBasedJudge,
    Turn,
    read_turn,
    system_prompt,
    turn_guidance,
)
from sqlalchemy import ForeignKey, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
)

DATABASE_URL = os.getenv(
    "SAHAI_SESSION_DSN", "postgresql+asyncpg://sahai:sahai@postgres:5432/sahai"
)
TUTOR_URL = os.getenv("SAHAI_TUTOR_URL", "http://tutor:8000")
TRACER_URL = os.getenv("SAHAI_TRACER_URL", "http://tracer:8000")
EXECUTOR_URL = os.getenv("SAHAI_EXECUTOR_URL", "http://executor:8000")
MAX_TURNS = int(os.getenv("SAHAI_MAX_TURNS", "12"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("session")

# Continues the trace started by the gateway rather than minting a new id.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    learner_id: Mapped[str] = mapped_column(String(64), index=True)
    problem_id: Mapped[str] = mapped_column(String(64))
    problem_title: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[str] = mapped_column(String(32))
    turns: Mapped[list["TurnRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="TurnRow.idx"
    )


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    idx: Mapped[int]
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    complete: Mapped[bool] = mapped_column(default=True)
    session: Mapped[SessionRow] = relationship(back_populates="turns")


engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# Must stay below the gateway's 270s and above the tutor's 200s generation
# cap. See the ladder comment in services/gateway/app/main.py.
SESSION_TIMEOUT_S = 240.0
PROBE_TIMEOUT_S = 3.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # 60s was fine for the stub backend; CPU inference on the real "hf"
    # backend (no GPU passthrough into Docker) can take well over a minute
    # for ~200 tokens, so this needs headroom for that path too.
    app.state.http = httpx.AsyncClient(
        timeout=SESSION_TIMEOUT_S,
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
    )
    yield
    await app.state.http.aclose()
    await engine.dispose()


app = FastAPI(title="sahai-session", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def trace(request: Request, call_next):
    rid = request.headers.get("x-request-id", "-")
    _request_id.set(rid)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    logger.info("rid=%s %s %s -> %s", rid, request.method,
                request.url.path, response.status_code)
    return response


def _trace() -> dict[str, str]:
    return {"x-request-id": _request_id.get()}


async def get_db() -> AsyncSession:  # pragma: no cover - wiring
    async with SessionLocal() as session:
        yield session


class StartRequest(BaseModel):
    learner_id: str = Field(max_length=64)
    problem_id: str = Field(max_length=64)
    problem_title: str = ""
    problem_description: str = ""


class TurnIn(BaseModel):
    content: str = Field(max_length=8000)
    # Pre-resolved by the gateway, which owns the problem bank and the tracer
    # lookup. Optional so the session service stays usable on its own and so a
    # tracer outage degrades the hint instead of failing the turn.
    learner_context: str = Field(default="", max_length=2000)
    # The full problem statement. `problem_title` on the row is truncated to
    # 80 chars by the exporter, which left 43 of the bank's 89 problems
    # reaching the model cut mid-word. Sent per turn rather than stored,
    # because `sessions` is created by create_all with no migration story and
    # a new column would never reach the existing table.
    problem_statement: str = Field(default="", max_length=4000)


class TurnOut(BaseModel):
    role: str
    content: str


class SessionOut(BaseModel):
    session_id: str
    status: str
    turns: list[TurnOut]
    tutor_reply: str | None = None
    # Needed to resume: the page has to know which problem the conversation is
    # about before it can reattach to it. Optional so the field can be absent
    # on responses built before a row is loaded.
    problem_id: str | None = None
    problem_title: str = ""


class SubmitRequest(BaseModel):
    code: str = Field(max_length=100_000)
    function_name: str
    test_cases: list[dict]
    skills: list[str] = Field(default_factory=list)


class DiagnosticGrade(BaseModel):
    learner_id: str = Field(max_length=64)
    problem_id: str = Field(max_length=64)
    code: str = Field(max_length=100_000)
    function_name: str
    test_cases: list[dict]
    skills: list[str] = Field(default_factory=list)


class ResumableOut(BaseModel):
    session_id: str
    problem_id: str
    problem_title: str
    status: str
    created_at: str
    turns: int
    last_role: str


@app.get("/health")
async def health() -> dict[str, str]:
    """Reports the executor on the gateway's behalf.

    Session is the only service on both the `internal` and `sandbox` networks,
    so it is the only one that can see the executor at all.
    """
    try:
        r = await app.state.http.get(f"{EXECUTOR_URL}/health", timeout=PROBE_TIMEOUT_S)
        executor = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except Exception as exc:  # noqa: BLE001 - report, never raise
        executor = f"unreachable: {type(exc).__name__}"
    return {"status": "ok", "service": "sahai-session", "executor": executor}


def _to_dialogue(row: SessionRow) -> Dialogue:
    d = Dialogue(problem_id=row.problem_id)
    for t in row.turns:
        d.add(t.role, t.content, complete=t.complete)
    return d


async def _load(db: AsyncSession, session_id: str) -> SessionRow:
    # Two things are required here, and both are easy to miss:
    #
    # selectinload    — a lazy relationship accessed outside the async session's
    #                   greenlet raises MissingGreenlet under asyncio.
    # populate_existing — the session uses expire_on_commit=False, so a re-query
    #                   returns the identity-mapped object with its previously
    #                   loaded `turns` collection intact. Without this the API
    #                   responds with a transcript one exchange behind the
    #                   database, which is invisible until you diff the two.
    row = (
        await db.execute(
            select(SessionRow)
            .options(selectinload(SessionRow.turns))
            .where(SessionRow.id == session_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "session not found")
    return row


LearnerHeader = Header(
    default=None,
    alias="x-learner-id",
    description="Who is asking. Forwarded by the gateway from its own header.",
)


async def _load_owned(
    db: AsyncSession, session_id: str, learner_id: str | None
) -> SessionRow:
    """Load a session and check the caller owns it.

    `add_turn`, `submit` and `get_session` took a session id and no identity at
    all, so any learner id could read or write any session. The UI never did
    that on purpose, but nothing downstream would have refused it — which is
    what made the mid-conversation learner switch on the tutor page able to
    append one learner's turns to another's transcript.

    Identity here is still the stub the gateway uses: a header, not a verified
    token. Checking ownership against it is not authentication, but it does
    close the gap between "the UI would not do that" and "the service would not
    allow it".

    404 rather than 403 on a mismatch: whether a session exists is not
    something a non-owner should be able to probe.
    """
    if not learner_id:
        raise HTTPException(401, "x-learner-id header required")
    row = await _load(db, session_id)
    if row.learner_id != learner_id:
        raise HTTPException(404, "session not found")
    return row


@app.post("/sessions", response_model=SessionOut)
async def start(req: StartRequest, db: AsyncSession = Depends(get_db)) -> SessionOut:
    row = SessionRow(
        id=str(uuid.uuid4()),
        learner_id=req.learner_id,
        problem_id=req.problem_id,
        problem_title=req.problem_title,
        status="active",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    db.add(row)
    await db.commit()
    return SessionOut(
        session_id=row.id, status=row.status, turns=[],
        problem_id=row.problem_id, problem_title=row.problem_title,
    )


@app.post("/sessions/{session_id}/turns", response_model=SessionOut)
async def add_turn(
    session_id: str,
    req: TurnIn,
    x_learner_id: str | None = LearnerHeader,
    db: AsyncSession = Depends(get_db),
) -> SessionOut:
    """Record the learner's turn and return the tutor's reply."""
    row = await _load_owned(db, session_id, x_learner_id)
    if row.status != "active":
        raise HTTPException(409, f"session is {row.status}")
    if len(row.turns) >= MAX_TURNS * 2:
        row.status = "exhausted"
        await db.commit()
        raise HTTPException(409, "turn limit reached")

    # A turn is one exchange, so both halves commit together or neither does.
    #
    # The student's turn used to be committed before the tutor was called. When
    # that call timed out — which it does on CPU, where generation can outlast
    # the 240s client budget — the dialogue was left holding a student turn
    # with no reply, status still "active". Retrying then appended a *second*
    # student turn, because nothing checks who spoke last, and consecutive
    # same-role turns break the alternation that prompt assembly, the pedagogy
    # judge and the tutor's own chat template all assume.
    #
    # Losing the student's text on failure is the better trade: the client
    # still has it and can retry, whereas a corrupted dialogue is unrecoverable
    # and silently mis-scores every turn after it.
    if row.turns and row.turns[-1].role == "student":
        raise HTTPException(
            409,
            "last turn has no tutor reply — the previous request failed; retry it",
        )

    pending_student = TurnRow(
        session_id=row.id, idx=len(row.turns), role="student",
        content=req.content, complete=True,
    )

    messages = [
        {
            "role": "system",
            "content": _system_prompt(
                row, req.learner_context, req.problem_statement, req.content
            ),
        }
    ]
    for t in row.turns:
        messages.append(
            {"role": "assistant" if t.role == "tutor" else "user", "content": t.content}
        )
    messages.append({"role": "user", "content": req.content})

    # Before any write: a failure here must leave the session exactly as it was.
    reply = await _call_tutor(messages)

    db.add(pending_student)
    db.add(TurnRow(session_id=row.id, idx=len(row.turns) + 1, role="tutor",
                   content=reply["text"], complete=reply["complete"]))

    if any(p in req.content.lower() for p in TERMINATION_PHRASES):
        row.status = "ready_to_submit"
    await db.commit()
    row = await _load(db, session_id)

    return SessionOut(
        session_id=row.id,
        status=row.status,
        turns=[TurnOut(role=t.role, content=t.content) for t in row.turns],
        tutor_reply=reply["text"],
        problem_id=row.problem_id,
        problem_title=row.problem_title,
    )


@app.post("/sessions/{session_id}/submit")
async def submit(
    session_id: str,
    req: SubmitRequest,
    x_learner_id: str | None = LearnerHeader,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Run the learner's code, then advance their mastery on the outcome.

    This closes the loop the whole system is built around: the tutor's value is
    measured by whether the learner can now solve it, verified by execution
    rather than by anyone's opinion.
    """
    row = await _load_owned(db, session_id, x_learner_id)

    result = await _call_executor(req.code, req.function_name, req.test_cases)
    solved = bool(result["fully_passed"])

    if req.skills:
        await _call_tracer(row.learner_id, req.skills, solved, row.problem_id)

    row.status = "solved" if solved else "attempted"
    await db.commit()

    dialogue = _to_dialogue(row)
    return {
        "session_id": row.id,
        "solved": solved,
        "pass_rate": result["pass_rate"],
        "results": result["results"],
        # Diagnostics on the tutor's own behaviour, computed with the same rule
        # judge the trainer optimises against.
        "pedagogy_score": RuleBasedJudge().evaluate(dialogue),
    }


@app.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    x_learner_id: str | None = LearnerHeader,
    db: AsyncSession = Depends(get_db),
) -> SessionOut:
    row = await _load_owned(db, session_id, x_learner_id)
    return SessionOut(
        session_id=row.id,
        status=row.status,
        turns=[TurnOut(role=t.role, content=t.content) for t in row.turns],
        problem_id=row.problem_id,
        problem_title=row.problem_title,
    )


def _system_prompt(
    row: SessionRow,
    learner_context: str = "",
    statement: str = "",
    learner_message: str = "",
) -> str:
    """Delegates to the shared prompt; falls back to the stored title.

    The title is a truncated statement, so it is the degraded path — used only
    when a caller sends no statement, which keeps this service usable on its
    own without silently pretending the truncation is fine.

    `learner_message` is read here rather than at the gateway because both the
    text and the voice path funnel through `add_turn`: doing it once at the
    point the prompt is assembled means the signals are computed from exactly
    the string that reaches the model, and a transcribed turn gets the same
    treatment as a typed one without the websocket handler repeating anything.
    """
    return system_prompt(
        statement or row.problem_title,
        learner_context,
        turn_guidance(read_turn(learner_message)),
    )


async def _call_tutor(messages: list[dict]) -> dict:
    r = await app.state.http.post(
        f"{TUTOR_URL}/generate",
        json={"messages": messages, "sanitize": True},
        headers=_trace(),
    )
    r.raise_for_status()
    return r.json()


async def _call_executor(code: str, function_name: str, test_cases: list[dict]) -> dict:
    r = await app.state.http.post(
        f"{EXECUTOR_URL}/execute",
        json={"code": code, "function_name": function_name, "test_cases": test_cases},
        headers=_trace(),
    )
    r.raise_for_status()
    return r.json()


async def _call_tracer(
    learner_id: str, skills: list[str], correct: bool, problem_id: str | None = None
) -> dict:
    r = await app.state.http.post(
        f"{TRACER_URL}/observe",
        json={
            "learner_id": learner_id,
            "skills": skills,
            "correct": correct,
            # Item identity, needed for anything beyond skill-level tracing.
            "problem_id": problem_id,
        },
        headers=_trace(),
    )
    r.raise_for_status()
    return r.json()


@app.post("/diagnostic/grade")
async def diagnostic_grade(req: DiagnosticGrade) -> dict:
    """Grade one diagnostic answer — no tutor, no conversation.

    Deliberately not a tutoring session. A diagnostic is the learner working
    unaided, so there is no transcript to keep and nothing for the pedagogy
    judge to score; creating a session row for it would put dialogues with no
    dialogue in the table.

    It does record a real observation, which is the whole point. Placement
    seeds a *prior* from what someone says about themselves; this produces
    evidence from what they actually wrote, so it outranks the survey rather
    than duplicating it — and unlike a prior it can move mastery in either
    direction.
    """
    result = await _call_executor(req.code, req.function_name, req.test_cases)
    passed = bool(result.get("fully_passed"))

    if req.skills:
        await _call_tracer(req.learner_id, req.skills, passed, req.problem_id)

    return {
        "problem_id": req.problem_id,
        "passed": passed,
        "pass_rate": result.get("pass_rate", 0.0),
        "results": result.get("results", []),
    }


RESUMABLE_STATUSES = ("active", "ready_to_submit")


@app.get("/sessions", response_model=list[ResumableOut])
async def list_resumable(
    learner_id: str, limit: int = 5, db: AsyncSession = Depends(get_db)
) -> list[ResumableOut]:
    """A learner's unfinished sessions, newest first.

    Conversations already survive a restart — they are rows in Postgres — but
    nothing ever surfaced one, so walking away mid-dialogue meant losing it in
    practice while the data sat there. This is the read that makes the
    persistence worth having.

    Sessions with no turns are excluded. Selecting a problem creates a row
    before a word is exchanged, so most `active` rows are a click and nothing
    more; offering those back as "unfinished work" would bury the real ones.
    """
    rows = (
        await db.execute(
            select(SessionRow)
            .options(selectinload(SessionRow.turns))
            .where(
                SessionRow.learner_id == learner_id,
                SessionRow.status.in_(RESUMABLE_STATUSES),
            )
            .order_by(SessionRow.created_at.desc())
        )
    ).scalars()

    out: list[ResumableOut] = []
    for row in rows:
        if not row.turns:
            continue
        out.append(
            ResumableOut(
                session_id=row.id,
                problem_id=row.problem_id,
                problem_title=row.problem_title,
                status=row.status,
                created_at=row.created_at,
                turns=len(row.turns),
                # "student" means the previous request died before the tutor
                # replied; the UI can say so instead of the learner discovering
                # it by sending a turn and getting a 409.
                last_role=row.turns[-1].role,
            )
        )
        if len(out) >= limit:
            break
    return out
