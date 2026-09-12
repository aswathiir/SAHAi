"""Shape of the tutor system prompt.

Personalisation is appended, never prepended: the never-give-the-answer rules
are what the whole design rests on, and a context block ahead of them would
push them away from the generation point.
"""

from __future__ import annotations

def test_system_prompt_appends_learner_context_after_the_rules():
    """Context goes last so it can never displace the never-give-the-answer
    rules — those are what the whole design rests on."""
    from app.main import SessionRow, _system_prompt

    row = SessionRow(id="s", learner_id="me", problem_id="p", problem_title="Sum a list")

    plain = _system_prompt(row)
    assert "NEVER write code" in plain
    assert plain.endswith("Problem: Sum a list")

    with_ctx = _system_prompt(row, "WHAT THIS LEARNER ALREADY KNOWS:\n- New to arrays.")
    assert with_ctx.startswith(plain)
    assert with_ctx.index("NEVER write code") < with_ctx.index("New to arrays")


def test_system_prompt_unchanged_when_context_is_empty():
    """A tracer outage must cost a tailored hint, not change the prompt."""
    from app.main import SessionRow, _system_prompt

    row = SessionRow(id="s", learner_id="me", problem_id="p", problem_title="Sum a list")
    assert _system_prompt(row, "") == _system_prompt(row)


def test_a_turn_is_atomic():
    """Both halves of an exchange commit together, or neither does.

    The student's turn used to be committed before the tutor was called, so a
    timeout left a dialogue holding a student turn with no reply. A retry then
    appended a *second* student turn, and consecutive same-role turns break the
    alternation that prompt assembly, the pedagogy judge and the chat template
    all assume. Verified live: a real 240s timeout left exactly that state.
    """
    import inspect

    from app.main import add_turn

    src = inspect.getsource(add_turn)
    call = src.index("_call_tutor")
    # No commit may precede the tutor call inside the turn handler.
    assert "await db.commit()" not in src[:call].split("turn limit reached")[-1]
    assert src.index('role="student"') < call or "pending_student" in src


def test_dangling_student_turn_is_rejected_not_compounded():
    """A session left mid-exchange by a failed request must say so, rather
    than quietly stacking a second student turn on top."""
    import inspect

    from app.main import add_turn

    src = inspect.getsource(add_turn)
    assert 'row.turns[-1].role == "student"' in src
    assert "409" in src


def test_session_response_carries_the_problem_it_is_about():
    """Resuming needs it: the page cannot reattach to a conversation without
    knowing which problem it concerns. It was absent, so `session.problem_id`
    read as undefined and the resume fell back to the first problem in the
    bank — reattaching the learner to the wrong conversation."""
    from app.main import SessionOut

    assert "problem_id" in SessionOut.model_fields
    assert "problem_title" in SessionOut.model_fields


def test_every_session_response_populates_it():
    """A field nothing fills is worse than no field — it reads as a valid
    `None` rather than a missing feature."""
    import inspect
    import re

    import app.main as m

    src = inspect.getsource(m)
    sites = list(re.finditer(r"return SessionOut\(", src))
    assert sites, "no SessionOut constructions found"
    for site in sites:
        # Counting occurrences module-wide does not work: ResumableOut sets the
        # same field, so an earlier version of this test compared 3 against 5
        # and failed on correct code.
        block = src[site.start() : site.start() + 300]
        line = src[: site.start()].count("\n") + 1
        assert "problem_id=row.problem_id" in block, (
            f"SessionOut built at line {line} without problem_id"
        )


def test_session_endpoints_check_ownership():
    """`add_turn`, `submit` and `get_session` took a session id and no identity
    at all, so any learner id could read or write any session.

    The UI never did that deliberately, but nothing downstream would have
    refused it — which is what let a mid-conversation learner switch on the
    tutor page append one learner's turns to another's transcript. Identity is
    still the gateway's stub header rather than a verified token; checking
    ownership against it is not authentication, but it closes the gap between
    "the UI would not do that" and "the service would not allow it".
    """
    import inspect

    import app.main as m

    for name in ("add_turn", "submit", "get_session"):
        src = inspect.getsource(getattr(m, name))
        assert "_load_owned(" in src, f"{name} loads a session without checking the owner"
        assert "x_learner_id" in src, f"{name} does not take an identity"


def test_mismatched_owner_looks_like_a_missing_session():
    """404 rather than 403: whether a session exists is not something a
    non-owner should be able to probe."""
    import inspect

    from app.main import _load_owned

    src = inspect.getsource(_load_owned)
    # The raise, not the prose — the docstring explains the 404-over-403 choice
    # and an earlier version of this test matched its own explanation.
    assert "HTTPException(404" in src
    assert "HTTPException(403" not in src
