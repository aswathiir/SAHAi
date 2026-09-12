"""The request path's deadline ladder must stay ordered, outermost first.

    gateway 270s  ->  session 240s  ->  tutor generate 200s

Each hop has to outlast the one it calls. When that order inverts the outer
caller abandons a request the inner one is about to answer: the learner sees a
500 and the work is discarded rather than returned. That is not hypothetical —
a real turn was lost this way at exactly 240s, because the tutor had no
generation cap at all and ran past session's budget.

Read from source rather than imported: the services need torch and a database
driver the research venv does not have, and this is an arithmetic invariant
between three literals, not a runtime behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _constant(relative_path: str, name: str) -> float:
    source = (ROOT / relative_path).read_text()
    match = re.search(rf"^{name}\s*=\s*(?:float\(os\.getenv\([^,]+,\s*\")?([0-9.]+)",
                      source, re.MULTILINE)
    assert match, f"{name} not found in {relative_path}"
    return float(match.group(1))


def test_ladder_is_ordered_outermost_first():
    gateway = _constant("services/gateway/app/main.py", "GATEWAY_TIMEOUT_S")
    session = _constant("services/session/app/main.py", "SESSION_TIMEOUT_S")
    generate = _constant("services/tutor/app/backends.py", "GENERATE_MAX_SECONDS")

    assert gateway > session, (
        f"gateway ({gateway}s) must outlast session ({session}s), or it abandons "
        "requests session is still answering"
    )
    assert session > generate, (
        f"session ({session}s) must outlast the tutor's generation cap "
        f"({generate}s), or a reply the model produced is thrown away"
    )


def test_session_and_gateway_agree_on_the_session_budget():
    """The gateway documents session's timeout to justify its own. If session
    changes and the gateway's copy does not, the ladder silently loses its
    margin."""
    assert _constant("services/gateway/app/main.py", "SESSION_TIMEOUT_S") == _constant(
        "services/session/app/main.py", "SESSION_TIMEOUT_S"
    )


def test_margins_are_wide_enough_to_absorb_a_hop():
    """Each gap must cover the network hop and request handling around the
    inner call, not just the inner call itself."""
    gateway = _constant("services/gateway/app/main.py", "GATEWAY_TIMEOUT_S")
    session = _constant("services/session/app/main.py", "SESSION_TIMEOUT_S")
    generate = _constant("services/tutor/app/backends.py", "GENERATE_MAX_SECONDS")

    assert gateway - session >= 20
    assert session - generate >= 20
