"""Mastery -> tutor prompt.

The tutor previously gave every learner the same hint for the same problem no
matter what they had demonstrated. These pin the translation and, more
importantly, the two ways it could do harm: leaking skills the caller named,
and failing a turn when the tracer is down.
"""

from __future__ import annotations

from app.main import MASTERY_NEW, MASTERY_SOLID, _learner_context


def test_new_skill_asks_for_the_idea_first():
    out = _learner_context(["hash_maps"], {"hash_maps": 0.1})
    assert "hash_maps" in out
    assert "Build the idea" in out


def test_solid_skill_is_not_re_explained():
    out = _learner_context(["arrays"], {"arrays": 0.9})
    assert "Do not re-explain" in out


def test_mid_band_says_nothing():
    """Between the cut points is exactly where the tutor should already be
    pitching, so guidance would only spend tokens agreeing with itself."""
    assert _learner_context(["arrays"], {"arrays": 0.5}) == ""


def test_unseen_skill_counts_as_new_not_middling():
    """No posterior means no evidence. Inventing a mid value would tell the
    tutor to assume knowledge the learner has never shown."""
    assert "Build the idea" in _learner_context(["trees"], {})


def test_only_mentions_skills_the_problem_touches():
    """The full profile is mostly irrelevant to any one problem, and this goes
    into a prompt whose own rule is 2-3 sentences per reply."""
    out = _learner_context(["arrays"], {"arrays": 0.1, "trees": 0.95, "heaps": 0.99})
    assert "arrays" in out
    assert "trees" not in out and "heaps" not in out


def test_no_skills_yields_no_context():
    assert _learner_context([], {"arrays": 0.1}) == ""


def test_every_placement_level_produces_guidance():
    """The bands must cover the range placement actually emits.

    They were first set to the ZPD cut points (0.3/0.7). But placement is
    capped inside [0.15, 0.65] so a confident learner still gets offered work,
    so three of its four levels landed in the silent mid band: a learner who
    had just imported 320 solved problems got no guidance at all.
    """
    placement_values = {"none": 0.15, "seen": 0.30, "practiced": 0.50, "confident": 0.65}
    spoken = {
        level: bool(_learner_context(["arrays"], {"arrays": value}))
        for level, value in placement_values.items()
    }
    assert spoken["none"] and spoken["seen"], "weak levels must ask for the idea first"
    assert spoken["confident"], "a confident level must stop the tutor re-explaining"
    assert not spoken["practiced"], "mid-band is where the tutor should already be"


def test_bands_are_not_silently_widened_to_nothing():
    assert 0 < MASTERY_NEW < MASTERY_SOLID < 1


def test_both_bands_can_appear_together():
    out = _learner_context(
        ["arrays", "hash_maps"], {"arrays": 0.95, "hash_maps": 0.05}
    )
    assert "Build the idea" in out and "Do not re-explain" in out
