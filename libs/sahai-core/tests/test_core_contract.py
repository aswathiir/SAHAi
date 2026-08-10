"""sahai-core contract tests.

The library's value is that every service can depend on it safely, so these
tests guard that property as much as the logic.
"""

import subprocess
import sys

from sahai_core import (
    AttemptResult,
    BKTTracer,
    Dialogue,
    ExecutionOutcome,
    RuleBasedJudge,
    SolveScorer,
    TestResult,
)


def _res(*outcomes):
    return AttemptResult([TestResult(o) for o in outcomes])


def test_core_imports_no_heavy_dependencies():
    """A service importing core must not inherit torch, a DB driver, or a web
    framework — that is the whole point of the split."""
    code = (
        "import sahai_core, sys;"
        "bad={'torch','transformers','peft','fastapi','sqlalchemy','datasets'} & set(sys.modules);"
        "print(sorted(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", f"core leaked: {out.stdout}"


def test_core_cannot_execute_code():
    """Scoring is separate from execution; core must expose no runner."""
    import sahai_core

    assert not hasattr(sahai_core, "CodeVerifier")
    assert not any("subprocess" in n for n in dir(sahai_core))


def test_solve_score_is_all_or_nothing():
    P, F = ExecutionOutcome.PASSED, ExecutionOutcome.FAILED
    attempts = [_res(P, P), _res(P, F), _res(F, F), _res(P, P)]
    assert SolveScorer.score(attempts) == 0.5           # 2 of 4 fully passed
    assert SolveScorer.partial_credit(attempts) == 0.625  # diagnostic only


def test_solve_score_empty_is_zero():
    assert SolveScorer.score([]) == 0.0


def test_pedagogy_is_graded_not_binary():
    """The change that made GRPO learn at all: partial credit keeps a group's
    reward variance non-zero."""
    judge = RuleBasedJudge()
    good = Dialogue("p"); good.add("student", "help"); good.add("tutor", "What structure gives O(1) lookup?")
    mid = Dialogue("p"); mid.add("student", "help"); mid.add("tutor", "Use a hash map.")
    bad = Dialogue("p"); bad.add("student", "help"); bad.add("tutor", "```python\ndef f(): return 1\n```")
    scores = {judge.evaluate(d) for d in (good, mid, bad)}
    assert len(scores) == 3, f"expected three distinct scores, got {scores}"


def test_tracer_mastery_moves_with_evidence():
    t = BKTTracer()
    start = t.get_mastery("arrays")
    for _ in range(5):
        t.update("arrays", True)
    assert t.get_mastery("arrays") > start


def test_empty_dialogue_scores_zero():
    """Silence is not good teaching — see tests/test_pedagogy.py for the story."""
    assert RuleBasedJudge().evaluate(Dialogue("p")) == 0.0
