"""Placement seeds BKT priors from a learner's self-assessment.

The property that matters: a self-report is weaker evidence than a graded
attempt, so it must never overwrite demonstrated history. Everything else here
is guarding that one rule from different angles.
"""

from __future__ import annotations

import os

os.environ.setdefault("SAHAI_TRACER_DSN", "sqlite+aiosqlite:///:memory:")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.main import PLACEMENT_PRIORS, Base, app, engine  # noqa: E402


@pytest_asyncio.fixture
async def client():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_placement_seeds_per_skill_priors(client):
    r = await client.post(
        "/placement",
        json={"learner_id": "ada", "responses": {"arrays": "confident", "trees": "none"}},
    )
    assert r.status_code == 200
    skills = r.json()["skills"]
    assert skills["arrays"] == PLACEMENT_PRIORS["confident"]
    assert skills["trees"] == PLACEMENT_PRIORS["none"]
    # The whole point: the two skills no longer start from the same flat prior.
    assert skills["arrays"] > skills["trees"]


async def test_placement_never_overwrites_observed_mastery(client):
    """Retaking the survey must not erase what the learner has demonstrated."""
    await client.post(
        "/placement", json={"learner_id": "ada", "responses": {"arrays": "none"}}
    )
    for _ in range(4):
        await client.post(
            "/observe",
            json={"learner_id": "ada", "skills": ["arrays"], "correct": True},
        )
    earned = (await client.get("/mastery/ada")).json()["skills"]["arrays"]
    assert earned > PLACEMENT_PRIORS["none"]

    # Re-taking it, now claiming ignorance, must not drag the posterior back.
    r = await client.post(
        "/placement", json={"learner_id": "ada", "responses": {"arrays": "none"}}
    )
    assert r.json()["skills"]["arrays"] == earned


async def test_placement_may_be_revised_before_any_attempt(client):
    """Seeded but never attempted is still just an opinion — let it change."""
    await client.post(
        "/placement", json={"learner_id": "ada", "responses": {"heaps": "none"}}
    )
    r = await client.post(
        "/placement", json={"learner_id": "ada", "responses": {"heaps": "practiced"}}
    )
    assert r.json()["skills"]["heaps"] == PLACEMENT_PRIORS["practiced"]


async def test_placement_writes_no_observations(client):
    """A prior is not an attempt. Logging it as one would corrupt the sequence
    a knowledge-tracing model trains on."""
    await client.post(
        "/placement",
        json={"learner_id": "ada", "responses": {"arrays": "confident"}},
    )
    rows = (await client.get("/observations/ada")).json()
    assert rows == []


async def test_placement_rejects_unknown_confidence(client):
    r = await client.post(
        "/placement", json={"learner_id": "ada", "responses": {"arrays": "expert"}}
    )
    assert r.status_code == 400
    assert "expert" in r.json()["detail"]


async def test_priors_stay_inside_the_zpd_band(client):
    """A confident self-report should place work at the edge of what the
    learner claims, not past it. The ZPD sampler offers problems in
    [0.3, 0.7]; a prior above that would skip the skill entirely."""
    assert max(PLACEMENT_PRIORS.values()) <= 0.7
    assert min(PLACEMENT_PRIORS.values()) > 0.0
