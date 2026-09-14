"""Identity is verified at the gateway, not claimed by the client.

The hole these replace: `x-learner-id` was a header the browser set to whatever
it liked and `_learner()` returned it unchanged, so every ownership check
downstream was resting on the client's honesty.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import identity
from app.main import app

client = TestClient(app)

PROTECTED = [
    ("get", "/v1/me/progress"),
    ("get", "/v1/me/mastery"),
    ("get", "/v1/me/sessions"),
    ("get", "/v1/diagnostic"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_no_credential_is_rejected(method, path):
    assert getattr(client, method)(path).status_code == 401


@pytest.mark.parametrize("method,path", PROTECTED)
def test_a_claimed_learner_id_is_rejected(method, path):
    """This is the exact request that used to work."""
    r = getattr(client, method)(path, headers={"x-learner-id": "victim"})
    assert r.status_code == 401


@pytest.mark.parametrize("method,path", PROTECTED)
def test_a_wrong_token_is_rejected(method, path):
    r = getattr(client, method)(path, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_a_real_token_is_accepted(auth_headers):
    assert client.get("/v1/diagnostic", headers=auth_headers).status_code == 200


def test_malformed_authorization_headers_are_rejected():
    for value in ("", "Bearer", "Bearer    ", "Basic abc", "abc", "bearer"):
        r = client.get("/v1/me/progress", headers={"Authorization": value})
        assert r.status_code == 401, value


def test_the_scheme_is_case_insensitive(auth):
    headers, _ = auth
    token = headers["Authorization"].split()[1]
    assert client.get("/v1/diagnostic", headers={"Authorization": f"bearer {token}"}).status_code == 200


# --- registration ---


def test_register_returns_the_token_once_and_an_id_it_chose(auth):
    headers, learner_id = auth
    assert learner_id.startswith(identity.LEARNER_ID_PREFIX)
    me = client.get("/v1/auth/me", headers=headers).json()
    assert me["learner_id"] == learner_id


def test_two_learners_get_different_ids_and_tokens():
    a = client.post("/v1/auth/register", json={"display_name": "A"}).json()
    b = client.post("/v1/auth/register", json={"display_name": "B"}).json()
    assert a["learner_id"] != b["learner_id"]
    assert a["token"] != b["token"]


def test_a_caller_cannot_name_its_own_learner_id():
    """Choosing an id would let a caller claim one that already holds someone's
    mastery history — the same hole in a different place. The *field* must not
    exist on the request model, not merely be ignored."""
    from app.main import RegisterIn

    assert set(RegisterIn.model_fields) == {"display_name"}

    body = client.post(
        "/v1/auth/register",
        json={"display_name": "sneaky", "learner_id": "victim"},
    ).json()
    assert body["learner_id"] != "victim"


def test_no_endpoint_hands_a_token_back(auth_headers):
    """Only a SHA-256 is stored, and there is deliberately no way to re-read
    the secret: an endpoint that can is an endpoint that can leak one."""
    body = client.get("/v1/auth/me", headers=auth_headers).json()
    assert "token" not in body
    assert "token_hash" not in body


def test_the_stored_value_is_not_the_token(auth):
    headers, _ = auth
    token = headers["Authorization"].split()[1]
    assert identity.hash_token(token) != token
    assert len(identity.hash_token(token)) == 64


def test_tokens_are_long_enough_to_be_unguessable():
    token = client.post("/v1/auth/register", json={"display_name": "x"}).json()["token"]
    # 32 random bytes, base64url-encoded. Short of that it is a password, and
    # a fast hash would be the wrong way to store a password.
    assert identity.TOKEN_BYTES >= 32
    assert len(token) >= 40


def test_an_empty_display_name_is_refused():
    assert client.post("/v1/auth/register", json={"display_name": ""}).status_code == 422


def test_registration_is_rate_limited_per_address(monkeypatch):
    """It cannot be limited per learner — it is what creates one. Unlimited, it
    is an unauthenticated row-insert loop."""
    import app.main as m

    monkeypatch.setattr(m, "REGISTRATIONS_PER_HOUR", 3)
    monkeypatch.setattr(m, "_registrations", m.defaultdict(m.deque))
    codes = [
        client.post("/v1/auth/register", json={"display_name": f"n{i}"}).status_code
        for i in range(5)
    ]
    assert codes.count(201) == 3
    assert codes[-1] == 429


# --- what this does and does not promise ---


def test_rate_limiting_keys_on_the_verified_learner():
    """It used to key on the client-supplied header, so changing one string
    reset the window."""
    import inspect

    import app.main as m

    src = inspect.getsource(m.start_session)
    assert "learner = await _learner(" in src
    assert src.index("_learner(") < src.index("_rate_limit(")


def test_the_learner_forwarded_inward_is_the_verified_one(auth):
    """Session and tracer still trust `x-learner-id`; they are only reachable
    from inside the Docker network, and the gateway is what fills it in."""
    import inspect

    import app.main as m

    src = inspect.getsource(m._trace)
    assert "x-learner-id" in src

    src = inspect.getsource(m.add_turn)
    assert "_trace(learner)" in src
    assert "authorization" in inspect.signature(m.add_turn).parameters
