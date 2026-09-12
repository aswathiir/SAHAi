"""Importing a solutions repo into skill priors.

The two things that could quietly go wrong: crediting bulk-imported work as if
it were recent, and letting a small repo's strongest skill read as mastery.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from import_neetcode import (  # noqa: E402
    HALF_LIFE_WEEKS,
    LEVEL_THRESHOLDS,
    tag,
    to_levels,
    weigh,
)

TODAY = date(2026, 9, 12)


def test_neetcode_names_map_to_their_technique():
    """NeetCode names problems after the story, not the method — the generic
    keyword tagger returns nothing useful for any of these."""
    assert "dynamic_programming" in tag("house-robber")
    assert "graphs" in tag("course-schedule")
    assert "stacks" in tag("daily-temperatures")
    assert "binary_search" in tag("eating-bananas")


def test_general_is_dropped_not_treated_as_a_skill():
    """`_tag_skills` returns ['general'] when it matches nothing. That is a
    fallback, not something a learner can be good at, and counting it would
    seed a prior from pure ignorance."""
    assert tag("some-unrecognisable-problem-name") == []


def test_older_work_counts_for_less():
    recent = {"binary-search": (TODAY - timedelta(weeks=1), False)}
    old = {"binary-search": (TODAY - timedelta(weeks=int(HALF_LIFE_WEEKS)), False)}
    assert weigh(recent, TODAY)["binary_search"] > weigh(old, TODAY)["binary_search"]


def test_half_life_halves_the_weight():
    now = weigh({"binary-search": (TODAY, False)}, TODAY)["binary_search"]
    later = weigh(
        {"binary-search": (TODAY - timedelta(weeks=int(HALF_LIFE_WEEKS)), False)}, TODAY
    )["binary_search"]
    assert abs(later - now / 2) < 0.02


def test_bulk_imported_work_is_discounted():
    """A bulk commit records when the repo was created, not when the problem
    was solved, so the true age is *at least* that. Crediting it in full would
    overestimate how recent — and so how well retained — the work is."""
    dated = weigh({"binary-search": (TODAY, False)}, TODAY)["binary_search"]
    bulk = weigh({"binary-search": (TODAY, True)}, TODAY)["binary_search"]
    assert bulk < dated


def test_levels_are_absolute_not_relative_to_the_learner():
    """A relative scale would call a four-problem repo's best skill 'confident',
    which is a placement flattering its own input."""
    tiny = {"heaps": 2.0}
    assert to_levels(tiny, ["heaps"])["heaps"] != "confident"
    big = {"heaps": LEVEL_THRESHOLDS[0][0] + 1}
    assert to_levels(big, ["heaps"])["heaps"] == "confident"


def test_unseen_skill_is_none():
    assert to_levels({}, ["trees"])["trees"] == "none"


def test_levels_are_ordered_by_evidence():
    weights = {"a": 50.0, "b": 5.0, "c": 1.0, "d": 0.0}
    levels = to_levels(weights, ["a", "b", "c", "d"])
    assert levels == {"a": "confident", "b": "practiced", "c": "seen", "d": "none"}
