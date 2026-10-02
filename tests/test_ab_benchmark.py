"""The statistics that decide what the A/B benchmark may conclude.

Worth pinning precisely. This project spent six runs comparing 20-problem
means that could not have resolved anything smaller than a 40-point swing, and
drew conclusions from the differences anyway. The arithmetic below is what
stops that happening again, so it is tested rather than trusted.
"""

from __future__ import annotations

from sahai.eval.ab_benchmark import mcnemar, min_detectable


class TestMcNemar:
    def test_one_sided_win_is_significant(self):
        a = [0] * 10 + [1] * 50
        b = [1] * 10 + [1] * 50
        m = mcnemar(a, b)
        assert m["b01"] == 10 and m["b10"] == 0
        assert m["p_value"] < 0.05

    def test_even_split_is_not(self):
        a = [0] * 10 + [1] * 10 + [0] * 40
        b = [1] * 10 + [0] * 10 + [0] * 40
        m = mcnemar(a, b)
        assert m["b01"] == 10 and m["b10"] == 10
        assert m["p_value"] == 1.0

    def test_known_exact_binomial_value(self):
        """6 vs 1 discordant: two-sided exact binomial is 0.125 exactly."""
        a = [0] * 6 + [1] + [0] * 53
        b = [1] * 6 + [0] + [0] * 53
        assert mcnemar(a, b)["p_value"] == 0.125

    def test_concordant_problems_are_ignored(self):
        """Padding with problems both arms solve must not change the p-value.

        This is the whole reason for pairing: problems neither arm can do, and
        problems both arms can do, carry no information about which is better.
        An unpaired test dilutes the signal with exactly those.
        """
        a = [0, 0, 0, 1]
        b = [1, 1, 1, 1]
        base = mcnemar(a, b)["p_value"]
        padded = mcnemar(a + [1] * 500, b + [1] * 500)["p_value"]
        assert base == padded

    def test_no_disagreement_is_reported_not_claimed_significant(self):
        m = mcnemar([1, 0, 1, 0], [1, 0, 1, 0])
        assert m["discordant"] == 0
        assert m["p_value"] == 1.0
        assert "note" in m

    def test_direction_is_not_symmetric_in_the_counts(self):
        a, b = [0, 0, 0], [1, 1, 1]
        fwd, rev = mcnemar(a, b), mcnemar(b, a)
        assert fwd["b01"] == rev["b10"] == 3
        assert fwd["p_value"] == rev["p_value"]  # two-sided, so p matches


class TestMinDetectable:
    def test_more_problems_detect_smaller_differences(self):
        vals = [min_detectable(n, 0.15) for n in (20, 60, 100, 200, 400)]
        assert vals == sorted(vals, reverse=True)

    def test_twenty_problems_could_not_have_resolved_this_project(self):
        """The historical evals. Nothing under ~40 points was ever detectable,
        and every observed run-to-run difference was far smaller than that."""
        assert min_detectable(20, 0.15) > 0.30

    def test_sixty_problems_still_cannot_resolve_ten_points(self):
        """Why the v24 result is not evidence either way."""
        assert min_detectable(60, 0.15) > 0.10
