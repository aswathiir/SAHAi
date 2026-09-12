"""End-to-end: gateway-less but complete tutoring loop.

Runs session + tutor + executor + tracer together in one process, with httpx
routing between their real ASGI apps. No mocks of our own code — the only thing
substituted is the tutor's model backend, which uses the stub.

This is the test that proves the services actually compose.
"""

from __future__ import annotations

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from executor_app.main import app as executor_app  # noqa: E402
from tracer_app.main import app as tracer_app  # noqa: E402
from tutor_app.main import app as tutor_app  # noqa: E402

from app.main import Base, app as session_app, engine  # noqa: E402


class RouterTransport(httpx.AsyncBaseTransport):
    """Dispatch by host to the right in-process ASGI app."""

    def __init__(self, routes: dict[str, httpx.AsyncBaseTransport]):
        self.routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self.routes.get(request.url.host)
        if transport is None:
            raise RuntimeError(f"no route for {request.url}")
        return await transport.handle_async_request(request)


SOLUTION = "def first_repeated_char(s):\n    seen = set()\n    for c in s:\n        if c in seen:\n            return c\n        seen.add(c)\n    return None"
WRONG = "def first_repeated_char(s):\n    return s[0]"
TESTS = [
    {"input": {"s": "abcabc"}, "expected": "a"},
    # The answer here is NOT the first character, so a `return s[0]` stub fails.
    {"input": {"s": "abcb"}, "expected": "b"},
]


@pytest_asyncio.fixture
async def stack():
    from tracer_app.main import Base as TracerBase, engine as tracer_engine

    async with engine.begin() as c:
        await c.run_sync(Base.metadata.drop_all)
        await c.run_sync(Base.metadata.create_all)
    async with tracer_engine.begin() as c:
        await c.run_sync(TracerBase.metadata.drop_all)
        await c.run_sync(TracerBase.metadata.create_all)

    router = RouterTransport({
        "tutor": ASGITransport(app=tutor_app),
        "tracer": ASGITransport(app=tracer_app),
        "executor": ASGITransport(app=executor_app),
    })

    # Bring the tutor backend up the way its lifespan would.
    from tutor_app.main import _state
    from tutor_app.backends import build_backend
    _state["backend"] = build_backend()

    session_app.state.http = AsyncClient(transport=router, timeout=30.0)
    async with AsyncClient(
        transport=ASGITransport(app=session_app), base_url="http://session"
    ) as client:
        yield client
    await session_app.state.http.aclose()


def _as(learner: str) -> dict[str, str]:
    """Session endpoints check that the caller owns the session, so every
    session-scoped call needs the owner's identity — the same stub header the
    gateway forwards."""
    return {"x-learner-id": learner}


async def test_full_loop_solved(stack):
    """Start a session, converse, submit a correct solution, mastery rises."""
    r = await stack.post("/sessions", json={
        "learner_id": "e2e-user", "problem_id": "mbpp-11",
        "problem_title": "First repeated character",
    })
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # Two exchanges with the tutor.
    for msg in ["I don't understand how to start.", "Maybe I track what I've seen?"]:
        r = await stack.post(f"/sessions/{sid}/turns", json={"content": msg}, headers=_as("e2e-user"))
        assert r.status_code == 200
        assert r.json()["tutor_reply"]

    body = r.json()
    assert [t["role"] for t in body["turns"]] == [
        "student", "tutor", "student", "tutor",
    ]

    # Submit a correct solution.
    r = await stack.post(f"/sessions/{sid}/submit", json={
        "code": SOLUTION, "function_name": "first_repeated_char",
        "test_cases": TESTS, "skills": ["strings", "hash_maps"],
    }, headers=_as("e2e-user"))
    assert r.status_code == 200
    result = r.json()
    assert result["solved"] is True
    assert result["pass_rate"] == 1.0

    # The loop closed: execution outcome advanced the learner model.
    tracer_client = AsyncClient(
        transport=ASGITransport(app=tracer_app), base_url="http://tracer"
    )
    mastery = (await tracer_client.get("/mastery/e2e-user")).json()
    await tracer_client.aclose()
    assert mastery["skills"]["strings"] > 0.3
    assert mastery["skills"]["hash_maps"] > 0.3


async def test_full_loop_failed_submission(stack):
    r = await stack.post("/sessions", json={
        "learner_id": "e2e-fail", "problem_id": "mbpp-11", "problem_title": "x",
    })
    sid = r.json()["session_id"]
    await stack.post(f"/sessions/{sid}/turns", json={"content": "help"}, headers=_as("e2e-fail"))

    r = await stack.post(f"/sessions/{sid}/submit", json={
        "code": WRONG, "function_name": "first_repeated_char",
        "test_cases": TESTS, "skills": ["strings"],
    }, headers=_as("e2e-fail"))
    body = r.json()
    assert body["solved"] is False
    assert body["pass_rate"] < 1.0


async def test_tutor_never_leaks_code_through_the_session(stack):
    """The pedagogy score is computed on the transcript the learner saw."""
    r = await stack.post("/sessions", json={
        "learner_id": "e2e-ped", "problem_id": "p", "problem_title": "t",
    })
    sid = r.json()["session_id"]
    for _ in range(3):
        await stack.post(f"/sessions/{sid}/turns", json={"content": "how do I do this?"}, headers=_as("e2e-ped"))

    r = await stack.post(f"/sessions/{sid}/submit", json={
        "code": SOLUTION, "function_name": "first_repeated_char",
        "test_cases": TESTS, "skills": ["strings"],
    }, headers=_as("e2e-ped"))
    assert r.json()["pedagogy_score"] == 1.0


async def test_session_survives_reload(stack):
    """Transcript is persisted, not held in memory."""
    r = await stack.post("/sessions", json={
        "learner_id": "e2e-persist", "problem_id": "p", "problem_title": "t",
    })
    sid = r.json()["session_id"]
    await stack.post(f"/sessions/{sid}/turns", json={"content": "first message"}, headers=_as("e2e-persist"))

    fetched = (await stack.get(f"/sessions/{sid}", headers=_as("e2e-persist"))).json()
    assert fetched["turns"][0]["content"] == "first message"
    assert len(fetched["turns"]) == 2


async def test_termination_phrase_moves_session_state(stack):
    r = await stack.post("/sessions", json={
        "learner_id": "e2e-term", "problem_id": "p", "problem_title": "t",
    })
    sid = r.json()["session_id"]
    r = await stack.post(f"/sessions/{sid}/turns",
                         json={"content": "I think I can solve it now"}, headers=_as("e2e-term"))
    assert r.json()["status"] == "ready_to_submit"


async def test_unknown_session_is_404(stack):
    assert (
        await stack.get("/sessions/does-not-exist", headers=_as("e2e-user"))
    ).status_code == 404


async def test_session_requires_an_identity(stack):
    """Session endpoints check the caller owns the session, so a request with
    no identity cannot be answered at all."""
    assert (await stack.get("/sessions/anything")).status_code == 401


async def test_another_learner_cannot_read_your_session(stack):
    """404, not 403 — whether a session exists is not something a non-owner
    should be able to probe."""
    r = await stack.post("/sessions", json={
        "learner_id": "owner-a", "problem_id": "p", "problem_title": "t",
    })
    sid = r.json()["session_id"]
    assert (await stack.get(f"/sessions/{sid}", headers=_as("owner-a"))).status_code == 200
    # owner-b on purpose: this is the case the check exists for.
    assert (await stack.get(f"/sessions/{sid}", headers=_as("owner-b"))).status_code == 404


async def test_response_transcript_matches_database(stack):
    """Regression: the turns returned must not lag the persisted transcript.

    expire_on_commit=False keeps the SessionRow in the identity map, so a
    re-query returns the stale `turns` collection unless populate_existing is
    set. The symptom is a response one exchange behind the database — every
    status code is 200 and nothing errors.
    """
    r = await stack.post("/sessions", json={
        "learner_id": "e2e-stale", "problem_id": "p", "problem_title": "t",
    })
    sid = r.json()["session_id"]

    for i in range(3):
        returned = (await stack.post(
            f"/sessions/{sid}/turns", json={"content": f"message {i}"},
            headers=_as("e2e-stale"),
        )).json()["turns"]
        persisted = (await stack.get(f"/sessions/{sid}", headers=_as("e2e-stale"))).json()["turns"]
        assert returned == persisted, f"exchange {i}: response lags the database"
        assert len(returned) == (i + 1) * 2
