"""Dataset layer: skill tagging and unsolvable problems.

Two defects found in the 2026-10-05 bottom-up audit.

Skills came from keyword-matching the problem *description*, so `general` --
the fallthrough when nothing matched -- was 22% of the bank, and BKT was
tracking per-learner mastery on a label that means nothing. The reference
solution was never consulted even though an AST technique detector already
existed and was already tested.

Problems whose reference solution fails its own tests are worse than noise for
GRPO: every rollout in the group scores `r_sol = 0` whatever the tutor said, so
the group has no reward variance, every z-scored advantage in it is zero, and
it contributes no gradient while costing a full generation budget.
"""

from __future__ import annotations

import pathlib

from sahai.core.dataset import _ast_skills, _tag_skills, drop_unsolvable


class FakeProblem:
    def __init__(self, pid, solution, tests=None, fn="f"):
        from sahai.core.data import TestCase

        self.id = pid
        self.solution = solution
        self.function_name = fn
        self.test_cases = tests if tests is not None else [TestCase(input={"a": [3, 1]}, expected=3)]
        self.title = "t"
        self.description = "d"


class TestSkillTagging:
    def test_keyword_tags_still_apply(self):
        assert "strings" in _tag_skills("Reverse the characters in a string.", "")

    def test_the_solution_contributes_tags_the_wording_does_not(self):
        """The measured case: 'recursion' appears in 12 solutions and the word
        'recursive' in 3 descriptions."""
        code = "def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\n"
        assert "recursion" in _tag_skills("Compute the nth term.", code)
        assert "recursion" not in _tag_skills("Compute the nth term.", "")

    def test_tags_are_the_union_not_a_replacement(self):
        """AST finds no technique in 74% of MBPP solutions, so it sharpens the
        keyword pass rather than replacing it."""
        code = "import heapq\ndef f(a):\n    return heapq.nlargest(2, a)\n"
        out = _tag_skills("Find the largest elements in a list.", code)
        assert "arrays" in out      # from the wording
        assert "heaps" in out       # from the solution

    def test_general_is_only_reached_when_both_fail(self):
        assert _tag_skills("Do the thing.", "def f():\n    return 1\n") == ["general"]

    def test_tags_are_sorted_and_unique(self):
        out = _tag_skills("count the unique elements in the list", "def f(a):\n    return len(set(a))\n")
        assert out == sorted(out) and len(out) == len(set(out))

    def test_an_unparseable_solution_loses_only_its_ast_tags(self):
        assert _tag_skills("Reverse a string.", "def f(:") == ["strings"]

    def test_ast_skills_is_safe_on_junk(self):
        assert _ast_skills("not python at all {{{") == set()
        assert _ast_skills("") == set()


class TestDropUnsolvable:
    def test_a_passing_reference_is_kept(self):
        p = FakeProblem("ok", "def f(a):\n    return max(a)\n")
        kept, dropped = drop_unsolvable([p])
        assert [x.id for x in kept] == ["ok"] and dropped == []

    def test_a_failing_reference_is_dropped(self):
        p = FakeProblem("bad", "def f(a):\n    return min(a)\n")
        kept, dropped = drop_unsolvable([p])
        assert kept == [] and [x.id for x in dropped] == ["bad"]

    def test_a_reference_that_raises_is_dropped_not_propagated(self):
        """An exploding reference is still unsolvable; it must not take the
        dataset load down with it."""
        p = FakeProblem("boom", "def f(a):\n    raise ValueError('x')\n")
        kept, dropped = drop_unsolvable([p])
        assert kept == [] and len(dropped) == 1

    def test_both_lists_are_returned(self):
        """Which problems were discarded is worth logging."""
        good = FakeProblem("g", "def f(a):\n    return max(a)\n")
        bad = FakeProblem("b", "def f(a):\n    return min(a)\n")
        kept, dropped = drop_unsolvable([good, bad])
        assert len(kept) == 1 and len(dropped) == 1


class TestLoaderContract:
    def test_validation_is_off_by_default(self):
        """Every number this project reported was measured without it; turning
        it on silently would make new runs incomparable with old ones."""
        src = pathlib.Path("sahai/core/dataset.py").read_text()
        assert "validate: bool = False" in src

    def test_the_loader_passes_the_solution_to_the_tagger(self):
        src = pathlib.Path("sahai/core/dataset.py").read_text()
        assert '_tag_skills(row["text"], row["code"])' in src

    def test_dropping_is_logged_rather_than_silent(self):
        src = pathlib.Path("sahai/core/dataset.py").read_text()
        block = src[src.index("if validate:"):]
        assert "logger.warning" in block[:600]
