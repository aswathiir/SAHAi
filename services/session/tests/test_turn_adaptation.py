"""The system prompt the tutor actually receives, driven through the endpoint.

The rest of this suite inspects source, which catches a missing call but not a
wrong one. This runs the real handler against a real (in-memory) database with
the tutor stubbed, and asserts on the exact message list that would have gone
over the wire — the only place where "the signals were read from the learner's
own turn" can be checked rather than assumed.
"""

from __future__ import annotations

import os

os.environ.setdefault("SAHAI_SESSION_DSN", "sqlite+aiosqlite:///:memory:")

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

import app.main as m  # noqa: E402
from app.main import Base, app, engine  # noqa: E402

OWNER = {"x-learner-id": "ada"}


@pytest_asyncio.fixture
async def client(monkeypatch):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    sent: list[list[dict]] = []

    async def fake_tutor(messages):
        sent.append(messages)
        return {"text": "What have you tried so far?", "complete": True}

    monkeypatch.setattr(m, "_call_tutor", fake_tutor)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        c.sent = sent
        yield c


async def _start(client) -> str:
    r = await client.post(
        "/sessions",
        json={"learner_id": "ada", "problem_id": "p1", "problem_title": "Sum a list"},
        headers=OWNER,
    )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _system(client) -> str:
    return client.sent[-1][0]["content"]


async def test_asking_for_the_answer_reaches_the_tutor_as_a_refusal(client):
    session_id = await _start(client)
    r = await client.post(
        f"/sessions/{session_id}/turns",
        json={"content": "just tell me the answer"},
        headers=OWNER,
    )
    assert r.status_code == 200, r.text

    system = _system(client)
    assert "THIS TURN:" in system
    assert "hand over the answer" in system
    # The rule it reinforces must still come first.
    assert system.index("NEVER write code") < system.index("THIS TURN:")


async def test_pasted_code_asks_for_a_reply_about_that_code(client):
    session_id = await _start(client)
    await client.post(
        f"/sessions/{session_id}/turns",
        json={"content": "def f(xs):\n    return sum(xs)\nfails on an empty list"},
        headers=OWNER,
    )
    assert "Respond to their code" in _system(client)


async def test_hinglish_in_asks_for_hinglish_back(client):
    session_id = await _start(client)
    await client.post(
        f"/sessions/{session_id}/turns",
        json={"content": "Mujhe samajh nahi aa raha hai ye kaise kaam karta hai"},
        headers=OWNER,
    )
    system = _system(client)
    assert "Reply in Hinglish" in system
    assert "They are stuck" in system


async def test_an_ordinary_turn_adds_nothing(client):
    session_id = await _start(client)
    await client.post(
        f"/sessions/{session_id}/turns",
        json={"content": "Should I use a set here?"},
        headers=OWNER,
    )
    assert "THIS TURN:" not in _system(client)


async def test_guidance_tracks_the_latest_turn_not_the_first(client):
    """Signals are per-message. A learner who was stuck two turns ago and has
    since pasted code must get the code reply, not the scaffold again."""
    session_id = await _start(client)
    await client.post(
        f"/sessions/{session_id}/turns",
        json={"content": "I don't understand any of this"},
        headers=OWNER,
    )
    assert "They are stuck" in _system(client)

    await client.post(
        f"/sessions/{session_id}/turns",
        json={"content": "ok here:\ndef f(xs):\n    return sum(xs)"},
        headers=OWNER,
    )
    system = _system(client)
    assert "Respond to their code" in system
    assert "They are stuck" not in system


async def test_the_learner_context_still_arrives_alongside_it(client):
    """Profile and turn are different questions and must not crowd each
    other out — who they are, and what they just said."""
    session_id = await _start(client)
    await client.post(
        f"/sessions/{session_id}/turns",
        json={
            "content": "just tell me the answer",
            "learner_context": "WHAT THIS LEARNER ALREADY KNOWS:\n- New to arrays.",
        },
        headers=OWNER,
    )
    system = _system(client)
    assert "New to arrays" in system
    assert system.index("New to arrays") < system.index("THIS TURN:")
