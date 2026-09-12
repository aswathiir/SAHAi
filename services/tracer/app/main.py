"""sahai-tracer — persistent learner model.

In the research code the BKT tracer lived *inside* the student simulator and
died with the process. That is correct for training, where the student is
synthetic and disposable, and wrong for production, where mastery belongs to a
real person and must survive across sessions.

This service owns that state. It is the only component that knows a learner's
history, and it exposes exactly three things the rest of the system needs:
current mastery, an observation endpoint, and ZPD problem selection.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sahai_core import BKTTracer, TracerSettings
from sqlalchemy import ForeignKey, String, UniqueConstraint, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv(
    "SAHAI_TRACER_DSN", "postgresql+asyncpg://sahai:sahai@postgres:5432/sahai"
)
SETTINGS = TracerSettings()


class Base(DeclarativeBase):
    pass


class SkillMastery(Base):
    """One mastery probability per (learner, skill).

    Stored as a scalar rather than an opaque vector precisely because the
    literature review flagged that uninterpretable learner state cannot drive
    pedagogical decisions. This value is directly readable and directly used
    for ZPD selection.
    """

    __tablename__ = "skill_mastery"
    __table_args__ = (UniqueConstraint("learner_id", "skill", name="uq_learner_skill"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    learner_id: Mapped[str] = mapped_column(String(64), index=True)
    skill: Mapped[str] = mapped_column(String(64))
    mastery: Mapped[float] = mapped_column(default=SETTINGS.p_init)
    observations: Mapped[int] = mapped_column(default=0)


class Observation(Base):
    """One graded attempt. Append-only — rows are never updated or deleted.

    `SkillMastery` above holds the *current* BKT posterior, which is a cache:
    it can be recomputed by replaying this table. The reverse is not true, so
    this is the record that actually matters.

    It exists now, while BKT is what runs, because DKT trains on the *sequence*
    of interactions and that history cannot be collected retroactively. Every
    day without this table is a day of training data lost.
    """

    __tablename__ = "observations"

    # Autoincrementing id doubles as the sequence order DKT needs.
    id: Mapped[int] = mapped_column(primary_key=True)
    learner_id: Mapped[str] = mapped_column(String(64), index=True)
    skill: Mapped[str] = mapped_column(String(64), index=True)
    correct: Mapped[bool]
    # Item identity. DKT predicts per-exercise, not only per-skill, so dropping
    # this would make the log unusable for anything but skill-level models.
    problem_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    observed_at: Mapped[str] = mapped_column(String(32))
    # The BKT posterior after this observation, so a later model can be compared
    # against what BKT believed at the time.
    mastery_after: Mapped[float]


engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="sahai-tracer", version="0.1.0", lifespan=lifespan)


async def get_db() -> AsyncSession:  # pragma: no cover - wiring
    async with SessionLocal() as session:
        yield session


class ObserveRequest(BaseModel):
    learner_id: str = Field(max_length=64)
    skills: list[str] = Field(max_length=32)
    correct: bool
    # Optional so existing callers keep working, but always send it: without
    # item identity the log only supports skill-level models.
    problem_id: str | None = Field(default=None, max_length=64)


class ObservationOut(BaseModel):
    """One row of the replayable history."""

    seq: int
    skill: str
    correct: bool
    problem_id: str | None
    observed_at: str
    mastery_after: float


class MasteryOut(BaseModel):
    learner_id: str
    skills: dict[str, float]
    ability: float


class ZPDRequest(BaseModel):
    learner_id: str
    candidates: list[dict] = Field(max_length=1000)
    n: int = Field(default=4, ge=1, le=64)


# Self-reported confidence -> BKT prior. Deliberately compressed into [0.15,
# 0.65]: a learner saying "I know this" is evidence, not proof, and anchoring
# a prior at 0.9 would take several failures to walk back. The top of the range
# sits just inside the ZPD band (0.3-0.7) so a confident learner is handed
# work at the edge of what they claim rather than being skipped past it.
PLACEMENT_PRIORS = {
    "none": 0.15,
    "seen": 0.30,
    "practiced": 0.50,
    "confident": 0.65,
}


class PlacementRequest(BaseModel):
    learner_id: str = Field(max_length=64)
    # skill -> one of PLACEMENT_PRIORS
    responses: dict[str, str] = Field(max_length=64)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sahai-tracer"}


async def _load(db: AsyncSession, learner_id: str) -> dict[str, float]:
    rows = (
        await db.execute(
            select(SkillMastery).where(SkillMastery.learner_id == learner_id)
        )
    ).scalars()
    return {r.skill: r.mastery for r in rows}


def _tracer_from(skills: dict[str, float]) -> BKTTracer:
    tracer = BKTTracer(SETTINGS)
    tracer.skills = dict(skills)
    return tracer


@app.get("/mastery/{learner_id}", response_model=MasteryOut)
async def get_mastery(learner_id: str, db: AsyncSession = Depends(get_db)) -> MasteryOut:
    skills = await _load(db, learner_id)
    return MasteryOut(
        learner_id=learner_id,
        skills=skills,
        ability=_tracer_from(skills).get_ability(),
    )


@app.post("/observe", response_model=MasteryOut)
async def observe(req: ObserveRequest, db: AsyncSession = Depends(get_db)) -> MasteryOut:
    """Record an outcome and advance the BKT posterior for each skill."""
    if not req.skills:
        raise HTTPException(400, "at least one skill required")

    existing = await _load(db, req.learner_id)
    tracer = _tracer_from(existing)
    tracer.update_batch(req.skills, req.correct)

    now = datetime.now(timezone.utc).isoformat()

    for skill in req.skills:
        row = (
            await db.execute(
                select(SkillMastery).where(
                    SkillMastery.learner_id == req.learner_id,
                    SkillMastery.skill == skill,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            # `default=` on mapped_column is a *column* default applied at
            # INSERT; the Python attribute stays None until flush, so the
            # counter has to be seeded explicitly here.
            row = SkillMastery(
                learner_id=req.learner_id,
                skill=skill,
                mastery=SETTINGS.p_init,
                observations=0,
            )
            db.add(row)
        row.mastery = tracer.skills[skill]
        row.observations = (row.observations or 0) + 1

        # Append the raw event. Written in the same transaction as the mastery
        # update so the cache and the history can never disagree.
        db.add(
            Observation(
                learner_id=req.learner_id,
                skill=skill,
                correct=req.correct,
                problem_id=req.problem_id,
                observed_at=now,
                mastery_after=tracer.skills[skill],
            )
        )

    await db.commit()
    return MasteryOut(
        learner_id=req.learner_id,
        skills=tracer.skills,
        ability=tracer.get_ability(),
    )


@app.post("/placement", response_model=MasteryOut)
async def placement(
    req: PlacementRequest, db: AsyncSession = Depends(get_db)
) -> MasteryOut:
    """Seed per-skill priors from a learner's self-assessment.

    BKT starts every learner at one global `p_init` (0.3) for every skill, so a
    first session picks problems as if the learner knew nothing about anything.
    A placement survey replaces that flat prior with a per-skill one, which is
    what makes the first assignment fit the person rather than the default.

    **Only seeds skills with no observations.** A self-report is weaker evidence
    than a graded attempt, so it must never overwrite what the learner has
    actually demonstrated — otherwise retaking the survey would silently erase
    real history. Skills already observed are returned unchanged.

    Nothing is written to `observations`: this is a prior, not an attempt, and
    logging it as one would corrupt the sequence a knowledge-tracing model
    trains on.
    """
    unknown = sorted(set(req.responses.values()) - set(PLACEMENT_PRIORS))
    if unknown:
        raise HTTPException(
            400, f"unknown confidence {unknown}; expected {sorted(PLACEMENT_PRIORS)}"
        )

    for skill, level in req.responses.items():
        row = (
            await db.execute(
                select(SkillMastery).where(
                    SkillMastery.learner_id == req.learner_id,
                    SkillMastery.skill == skill,
                )
            )
        ).scalar_one_or_none()

        if row is None:
            db.add(
                SkillMastery(
                    learner_id=req.learner_id,
                    skill=skill,
                    mastery=PLACEMENT_PRIORS[level],
                    observations=0,
                )
            )
        elif not row.observations:
            # Seeded before but never attempted — a re-take may still move it.
            row.mastery = PLACEMENT_PRIORS[level]

    await db.commit()
    skills = await _load(db, req.learner_id)
    return MasteryOut(
        learner_id=req.learner_id,
        skills=skills,
        ability=_tracer_from(skills).get_ability(),
    )


@app.get("/observations/{learner_id}", response_model=list[ObservationOut])
async def observations(
    learner_id: str,
    after_seq: int = 0,
    limit: int = 1000,
    db: AsyncSession = Depends(get_db),
) -> list[ObservationOut]:
    """Replayable interaction history, oldest first.

    `after_seq` lets a training job page through without rescanning: pass the
    last `seq` you saw. Sequence ordering is the point — a knowledge-tracing
    model consumes the order, not a bag of events.
    """
    rows = (
        await db.execute(
            select(Observation)
            .where(Observation.learner_id == learner_id, Observation.id > after_seq)
            .order_by(Observation.id)
            .limit(min(limit, 5000))
        )
    ).scalars()

    return [
        ObservationOut(
            seq=r.id,
            skill=r.skill,
            correct=r.correct,
            problem_id=r.problem_id,
            observed_at=r.observed_at,
            mastery_after=r.mastery_after,
        )
        for r in rows
    ]


@app.post("/zpd")
async def zpd(req: ZPDRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """Filter candidate problems to the learner's zone of proximal development."""
    tracer = _tracer_from(await _load(db, req.learner_id))
    eligible = [
        p
        for p in req.candidates
        if tracer.in_zpd(p.get("difficulty", 1), p.get("skills", []))
    ]
    if not eligible:
        eligible = req.candidates
    return {"problems": eligible[: req.n], "ability": tracer.get_ability()}
