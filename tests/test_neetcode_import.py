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


# --- step 2: reading the solution source ---

from import_neetcode import (  # noqa: E402
    TECHNIQUE_SKILLS,
    headline_technique,
    prior_work_items,
    techniques_used,
    title_for,
)

TWO_SUM = """
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
"""

KTH_LARGEST = """
import heapq
def findKthLargest(nums, k):
    h = []
    for n in nums:
        heapq.heappush(h, n)
        if len(h) > k:
            heapq.heappop(h)
    return h[0]
"""

SINGLE_NUMBER = """
def singleNumber(nums):
    r = 0
    for n in nums:
        r ^= n
    return r
"""


def test_the_source_says_what_the_name_cannot():
    """"kth-largest-element-in-an-array" is a heap problem or a sorting problem
    depending on what the learner actually wrote. Only the source knows."""
    assert "a heap" in techniques_used(KTH_LARGEST)
    assert "hash map" in techniques_used(TWO_SUM)


def test_augmented_assignment_counts_as_bit_manipulation():
    """The canonical xor solution is `r ^= n`, which is an AugAssign and has no
    BinOp node at all — checking only BinOp missed exactly the problems this
    tag exists for."""
    assert "bit manipulation" in techniques_used(SINGLE_NUMBER)


def test_unparseable_source_is_skipped_not_guessed_at():
    assert techniques_used("def f(:\n    pass") == set()
    assert headline_technique(set()) is None


def test_the_headline_is_the_most_specific_technique():
    """A solution that memoises is better described by that than by the
    recursion underneath it."""
    assert headline_technique({"recursion", "memoisation"}) == "memoisation"
    assert headline_technique({"sorting", "a heap"}) == "a heap"


def test_the_source_adds_skills_the_name_alone_would_miss():
    from import_neetcode import tag

    assert "bit_manipulation" not in tag("single-number")
    assert "bit_manipulation" in tag("single-number", techniques_used(SINGLE_NUMBER))


def test_every_technique_maps_to_a_real_skill():
    """A phrase with no skill behind it would be shown to the tutor and counted
    towards nothing."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sahai.core.dataset import SKILL_KEYWORDS

    for technique, skill in TECHNIQUE_SKILLS.items():
        assert skill in SKILL_KEYWORDS, f"{technique} -> unknown skill {skill}"


def test_prior_work_carries_names_and_techniques_and_no_code():
    """The boundary the whole design rests on: the source is parsed in this
    module and never leaves it."""
    solved = {"two-sum": (date(2026, 8, 24), False)}
    items = prior_work_items(solved, {"two-sum": techniques_used(TWO_SUM)})
    assert items == [{
        "slug": "two-sum", "title": "Two Sum", "skill": "hash_maps",
        "technique": "hash map", "solved_on": "2026-08-24", "dated": True,
    }]
    blob = repr(items)
    assert "def " not in blob and "return" not in blob and "seen" not in blob


def test_a_problem_with_no_recognisable_technique_is_skipped():
    """"You solved this somehow" gives a tutor nothing to build an analogy
    from."""
    solved = {"mystery": (date(2026, 8, 24), False)}
    assert prior_work_items(solved, {"mystery": set()}) == []


def test_bulk_imported_work_is_marked_undated():
    """The commit dates the repo's creation, not the solve. A tutor saying
    "three weeks ago" about it would be inventing a fact."""
    solved = {"two-sum": (date(2026, 8, 24), True)}
    items = prior_work_items(solved, {"two-sum": techniques_used(TWO_SUM)})
    assert items[0]["dated"] is False


def test_titles_read_like_problem_names_not_url_fragments():
    assert title_for("two-sum") == "Two Sum"
    assert title_for("binary-tree-level-order-traversal") == "Binary Tree Level Order Traversal"
    assert title_for("remove-nth-node-from-end-of-list") == "Remove Nth Node from End of List"
    assert title_for("lru-cache") == "LRU Cache"
