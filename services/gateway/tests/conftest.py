"""Test-wide setup for the gateway.

`app.identity` builds its engine at import time from `SAHAI_GATEWAY_DSN`, so the
DSN has to be set before any test module does `from app.main import app`.
conftest is loaded first, which is the only reliable place for it.

A temp file rather than `:memory:`: each async SQLite connection gets its own
private in-memory database, so the identity table would exist on the connection
that created it and nowhere else.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_DB = Path(tempfile.mkdtemp(prefix="sahai-gw-")) / "identity.db"
os.environ["SAHAI_GATEWAY_DSN"] = f"sqlite+aiosqlite:///{_DB}"
# Registration is rate-limited per client address, and every test shares one.
os.environ.setdefault("SAHAI_REGISTRATIONS_PER_HOUR", "10000")

import pytest  # noqa: E402

from app import identity  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _identity_schema():
    import asyncio

    asyncio.run(identity.create_tables())


@pytest.fixture
def auth(request):
    """Headers for a freshly registered learner, one per test.

    Per test rather than shared: several tests here assert that one learner
    cannot see another's work, and a shared identity would let those pass for
    the wrong reason.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        body = client.post(
            "/v1/auth/register", json={"display_name": request.node.name[:80]}
        ).json()
    return {"Authorization": f"Bearer {body['token']}"}, body["learner_id"]


@pytest.fixture
def auth_headers(auth):
    return auth[0]
