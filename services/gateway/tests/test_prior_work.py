"""Prior work reaches the tutor as an analogy, never as an answer.

The design question this settles: the tutor's entire reward exists to stop
solutions reaching the learner, so importing a repo of solved problems looks
like handing it exactly what it must not have. The resolution is that the code
never crosses the boundary — the importer parses it and keeps only a problem
title and a technique name. These tests hold that boundary in place.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

import app.main as m
from app.main import app

client = TestClient(app)

ITEMS = [
    {"slug": "two-sum", "title": "Two Sum", "skill": "hash_maps",
     "technique": "a hash map", "solved_on": "2026-08-24", "dated": True},
    {"slug": "contains-duplicate", "title": "Contains Duplicate", "skill": "hash_maps",
     "technique": "a set for lookups", "solved_on": "2025-01-02", "dated": False},
]


# --- the boundary ---


def test_the_request_model_has_nowhere_to_put_code():
    """Not "we strip it" — there is no field. A `code` field here would put the
    answer one careless prompt change away from the tutor."""
    from app.main import PriorWorkIn

    assert set(PriorWorkIn.model_fields) == {"items"}


def test_unknown_fields_are_dropped_before_they_reach_storage(monkeypatch, auth_headers):
    """`items` is a list of free dicts, so the allow-list is what actually
    holds the line."""
    sent = {}

    async def fake_post(url, json=None, headers=None, **kw):
        sent["items"] = json["items"]
        return _Resp(200, {"stored": len(json["items"])})

    monkeypatch.setattr(m.app.state, "http", _Http(post=fake_post), raising=False)
    sneaky = dict(ITEMS[0], code="def twoSum(): ...", solution="secret", source="x")
    r = client.post("/v1/me/prior_work", json={"items": [sneaky]}, headers=auth_headers)
    assert r.status_code == 200
    stored = sent["items"][0]
    assert set(stored) <= {"slug", "title", "skill", "technique", "solved_on", "dated"}
    for banned in ("code", "solution", "source"):
        assert banned not in stored


def test_the_learner_written_to_is_the_authenticated_one(monkeypatch, auth):
    """The importer cannot write to somebody else's record by naming them."""
    headers, learner_id = auth
    sent = {}

    async def fake_post(url, json=None, headers=None, **kw):
        sent.update(json)
        return _Resp(200, {"stored": 0})

    monkeypatch.setattr(m.app.state, "http", _Http(post=fake_post), raising=False)
    client.post(
        "/v1/me/prior_work",
        json={"items": [], "learner_id": "victim"},
        headers=headers,
    )
    assert sent["learner_id"] == learner_id


def test_prior_work_requires_a_token():
    assert client.post("/v1/me/prior_work", json={"items": []}).status_code == 401


# --- what the tutor is told ---


def test_the_current_problem_is_never_referenced():
    """Everything here is already solved. Naming the technique for the problem
    in front of them is the one hint that ends the lesson."""
    import asyncio

    async def fake_get(url, params=None, headers=None, **kw):
        return _Resp(200, {"items": ITEMS})

    m.app.state.http = _Http(get=fake_get)
    out = asyncio.run(m._prior_work_for("me", ["hash_maps"], exclude_slug="two-sum"))
    assert [i["slug"] for i in out] == ["contains-duplicate"]


def test_no_skills_means_no_lookup():
    import asyncio

    called = []

    async def fake_get(url, params=None, headers=None, **kw):
        called.append(url)
        return _Resp(200, {"items": ITEMS})

    m.app.state.http = _Http(get=fake_get)
    assert asyncio.run(m._prior_work_for("me", [])) == []
    assert not called


def test_a_tracer_outage_costs_an_analogy_not_the_turn():
    """Same rule as `_mastery_for`: personalisation improves a turn, it is not
    a precondition for one."""
    import asyncio

    async def boom(url, params=None, headers=None, **kw):
        raise RuntimeError("tracer down")

    m.app.state.http = _Http(get=boom)
    assert asyncio.run(m._prior_work_for("me", ["hash_maps"])) == []


def test_the_block_names_the_problem_and_the_technique_and_no_code():
    lines = m._prior_work_lines(ITEMS, today=date(2026, 9, 14))
    text = "\n".join(lines)
    assert "Two Sum" in text
    assert "a hash map" in text
    assert "def " not in text and "return" not in text


def test_the_tutor_is_told_to_ask_not_to_tell():
    """"You used a hash map in Two Sum" delivered as a statement is a hint.
    Asked back as a question it is the analogy doing the teaching."""
    header = m._prior_work_lines(ITEMS[:1])[0]
    assert "ask them to connect it" in header
    assert "do not connect it for them" in header


def test_undated_work_is_not_given_an_invented_date():
    """The importer cannot date bulk-imported commits, and a tutor confidently
    saying "three weeks ago" about something it cannot place is worse than
    vague."""
    text = "\n".join(m._prior_work_lines([ITEMS[1]], today=date(2026, 9, 14)))
    assert text.endswith("before.")
    assert "2025" not in text


@pytest.mark.parametrize("solved_on,expected", [
    ("2026-09-12", "this week"),
    ("2026-08-24", "3 weeks ago"),
    ("2026-02-01", "7 months ago"),
])
def test_recency_reads_the_way_a_person_would_say_it(solved_on, expected):
    item = dict(ITEMS[0], solved_on=solved_on, dated=True)
    text = "\n".join(m._prior_work_lines([item], today=date(2026, 9, 14)))
    assert expected in text


def test_nothing_to_say_adds_nothing_to_the_prompt():
    assert m._prior_work_lines([]) == []


# --- tiny httpx stand-ins ---


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Http:
    def __init__(self, get=None, post=None):
        self._get, self._post = get, post

    async def get(self, *a, **kw):
        return await self._get(*a, **kw)

    async def post(self, *a, **kw):
        return await self._post(*a, **kw)
