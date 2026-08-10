"""Tracer service tests — run against SQLite in-memory, not Postgres.

The point of these is the behaviour the research code could not have: mastery
that survives, is per-learner, and moves in the right direction.
"""

from __future__ import annotations

import os

os.environ["SAHAI_TRACER_DSN"] = "sqlite+aiosqlite:///:memory:"

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.main import Base, app, engine  # noqa: E402


@pytest_asyncio.fixture
async def client():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    r = await client.get("/health")
    assert r.json()["status"] == "ok"


async def test_unknown_learner_starts_at_prior(client):
    r = await client.get("/mastery/new-learner")
    body = r.json()
    assert body["skills"] == {}
    assert body["ability"] == pytest.approx(0.3)  # p_init


async def test_correct_answers_raise_mastery(client):
    before = (await client.get("/mastery/alice")).json()["ability"]
    for _ in range(5):
        await client.post("/observe", json={
            "learner_id": "alice", "skills": ["hash_maps"], "correct": True,
        })
    after = (await client.get("/mastery/alice")).json()["skills"]["hash_maps"]
    assert after > before


async def test_wrong_answers_lower_mastery(client):
    await client.post("/observe", json={
        "learner_id": "bob", "skills": ["recursion"], "correct": True,
    })
    high = (await client.get("/mastery/bob")).json()["skills"]["recursion"]
    for _ in range(3):
        await client.post("/observe", json={
            "learner_id": "bob", "skills": ["recursion"], "correct": False,
        })
    low = (await client.get("/mastery/bob")).json()["skills"]["recursion"]
    assert low < high


async def test_learners_are_isolated(client):
    """The bug the in-memory tracer made impossible to even express."""
    for _ in range(5):
        await client.post("/observe", json={
            "learner_id": "carol", "skills": ["trees"], "correct": True,
        })
    carol = (await client.get("/mastery/carol")).json()["skills"]["trees"]
    dave = (await client.get("/mastery/dave")).json()["skills"]
    assert carol > 0.3
    assert dave == {}


async def test_mastery_persists_across_requests(client):
    await client.post("/observe", json={
        "learner_id": "erin", "skills": ["sorting"], "correct": True,
    })
    first = (await client.get("/mastery/erin")).json()["skills"]["sorting"]
    second = (await client.get("/mastery/erin")).json()["skills"]["sorting"]
    assert first == second


async def test_zpd_filters_to_the_learnable_band(client):
    """Mastered skills should drop out of the candidate set."""
    for _ in range(30):
        await client.post("/observe", json={
            "learner_id": "frank", "skills": ["arrays"], "correct": True,
        })
    r = await client.post("/zpd", json={
        "learner_id": "frank",
        "candidates": [
            {"id": "p1", "difficulty": 1, "skills": ["arrays"]},
            {"id": "p2", "difficulty": 2, "skills": ["graphs"]},
        ],
        "n": 4,
    })
    picked = {p["id"] for p in r.json()["problems"]}
    assert "p2" in picked
    assert "p1" not in picked


async def test_observe_requires_a_skill(client):
    r = await client.post("/observe", json={
        "learner_id": "x", "skills": [], "correct": True,
    })
    assert r.status_code == 400


# --- observation log (data for knowledge tracing) ---------------------------

async def test_every_observation_is_recorded(client):
    for correct in (True, False, True):
        await client.post("/observe", json={
            "learner_id": "log-1", "skills": ["arrays"],
            "correct": correct, "problem_id": "mbpp-11",
        })
    rows = (await client.get("/observations/log-1")).json()
    assert [r["correct"] for r in rows] == [True, False, True]
    assert all(r["problem_id"] == "mbpp-11" for r in rows)


async def test_one_row_per_skill(client):
    await client.post("/observe", json={
        "learner_id": "log-2", "skills": ["arrays", "sorting"], "correct": True,
    })
    rows = (await client.get("/observations/log-2")).json()
    assert {r["skill"] for r in rows} == {"arrays", "sorting"}


async def test_observations_are_ordered_and_pageable(client):
    for i in range(5):
        await client.post("/observe", json={
            "learner_id": "log-3", "skills": ["trees"], "correct": i % 2 == 0,
        })
    rows = (await client.get("/observations/log-3")).json()
    seqs = [r["seq"] for r in rows]
    assert seqs == sorted(seqs), "sequence order is what a tracing model consumes"

    later = (await client.get(f"/observations/log-3?after_seq={seqs[1]}")).json()
    assert [r["seq"] for r in later] == seqs[2:]


async def test_log_is_append_only(client):
    """Mastery is overwritten in place; the history must not be."""
    for _ in range(4):
        await client.post("/observe", json={
            "learner_id": "log-4", "skills": ["dfs"], "correct": True,
        })
    assert len((await client.get("/observations/log-4")).json()) == 4


async def test_mastery_is_reproducible_from_the_log(client):
    """The whole justification for the table: SkillMastery is a cache that can
    be rebuilt by replaying observations, so nothing is lost by keeping only
    this and recomputing."""
    from sahai_core import BKTTracer

    outcomes = [True, False, True, True, False]
    for c in outcomes:
        await client.post("/observe", json={
            "learner_id": "log-5", "skills": ["recursion"], "correct": c,
        })

    stored = (await client.get("/mastery/log-5")).json()["skills"]["recursion"]

    replay = BKTTracer()
    for row in (await client.get("/observations/log-5")).json():
        replay.update(row["skill"], row["correct"])

    assert replay.get_mastery("recursion") == pytest.approx(stored)


async def test_learners_have_separate_logs(client):
    await client.post("/observe", json={
        "learner_id": "log-6a", "skills": ["stacks"], "correct": True,
    })
    assert (await client.get("/observations/log-6b")).json() == []
