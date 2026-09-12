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
