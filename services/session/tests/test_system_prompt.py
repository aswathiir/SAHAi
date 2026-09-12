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
