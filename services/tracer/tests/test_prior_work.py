"""Imported prior work: replaced on re-import, and holding no source code."""

from __future__ import annotations

import os

os.environ.setdefault("SAHAI_TRACER_DSN", "sqlite+aiosqlite:///:memory:")

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.main import Base, PriorWork, app, engine  # noqa: E402


@pytest_asyncio.fixture
async def client():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _items():
    return [
        {"slug": "two-sum", "title": "Two Sum", "skill": "hash_maps",
         "technique": "a hash map", "solved_on": "2026-08-24", "dated": True},
        {"slug": "coin-change", "title": "Coin Change", "skill": "dynamic_programming",
         "technique": "a DP table", "solved_on": "2026-05-01", "dated": False},
    ]


async def test_stores_and_returns_prior_work(client):
    r = await client.post("/prior_work", json={"learner_id": "ada", "items": _items()})
    assert r.status_code == 200
    assert r.json()["stored"] == 2

    got = (await client.get("/prior_work/ada")).json()["items"]
    assert {i["slug"] for i in got} == {"two-sum", "coin-change"}


async def test_re_importing_replaces_rather_than_accumulating(client):
    """Found live, not here: `db.delete(row)` per row defers to flush time and
    the ORM orders INSERTs first, so the second import hit
    `uq_learner_prior_slug` on a row it was about to remove. A bulk DELETE
    statement executes immediately, which is what makes replace mean replace.
    """
    for _ in range(3):
        r = await client.post("/prior_work", json={"learner_id": "ada", "items": _items()})
        assert r.status_code == 200, r.text

    got = (await client.get("/prior_work/ada")).json()["items"]
    assert len(got) == 2, f"accumulated duplicates: {got}"


async def test_a_shorter_re_import_drops_what_is_gone(client):
    await client.post("/prior_work", json={"learner_id": "ada", "items": _items()})
    await client.post("/prior_work", json={"learner_id": "ada", "items": _items()[:1]})
    got = (await client.get("/prior_work/ada")).json()["items"]
    assert [i["slug"] for i in got] == ["two-sum"]


async def test_filters_to_the_skills_asked_for(client):
    await client.post("/prior_work", json={"learner_id": "ada", "items": _items()})
    got = (await client.get("/prior_work/ada?skills=hash_maps")).json()["items"]
    assert [i["slug"] for i in got] == ["two-sum"]


async def test_most_recent_first_and_limited(client):
    await client.post("/prior_work", json={"learner_id": "ada", "items": _items()})
    got = (await client.get("/prior_work/ada?limit=1")).json()["items"]
    assert [i["slug"] for i in got] == ["two-sum"], "older work came first"


async def test_one_learners_import_does_not_touch_anothers(client):
    await client.post("/prior_work", json={"learner_id": "ada", "items": _items()})
    await client.post("/prior_work", json={"learner_id": "bob", "items": _items()[:1]})
    assert len((await client.get("/prior_work/ada")).json()["items"]) == 2


async def test_an_unknown_learner_is_empty_not_an_error(client):
    r = await client.get("/prior_work/nobody")
    assert r.status_code == 200
    assert r.json()["items"] == []


async def test_the_table_has_nowhere_to_put_source_code():
    """The boundary that makes reading a solutions repo safe at all. A column
    here would be one migration away from the tutor's prompt."""
    columns = set(PriorWork.__table__.columns.keys())
    assert columns == {
        "id", "learner_id", "slug", "title", "skill", "technique",
        "solved_on", "dated",
    }


async def test_prior_work_is_not_an_observation():
    """These are unverified claims from a third-party repo, not attempts this
    system graded. Mixing them would corrupt the sequence DKT trains on."""
    from app.main import Observation

    assert PriorWork.__tablename__ != Observation.__tablename__
