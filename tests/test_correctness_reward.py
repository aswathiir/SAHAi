"""The correctness term, and the guard that keeps it switched off.

Built because nothing in the reward asked whether the tutor's claims were
true: all three `r_ped` checks are negative, and leakage asks only whether the
answer escaped. Measured, before being enabled, on 120 real benchmark
dialogues -- it fires on 10% of them and its correlation with solving flips
sign between arms, so `mu_correct` stays 0.0 until a detector with real
coverage exists. These tests pin both the scoring and that default.
"""

from __future__ import annotations

import pathlib

from sahai.core.dialogue import Dialogue
from sahai.reward.correctness import claimed_techniques, evaluate
from sahai.reward.combined import SAHAIReward
from sahai.settings import RewardSettings, Settings


class FakeProblem:
    def __init__(self, solution: str):
        self.solution = solution
        self.title = "t"
        self.description = "d"


def dlg(*turns):
    d = Dialogue(problem_id="p")
    for role, text in turns:
        d.add(role, text)
    return d


HEAP = "import heapq\ndef f(a):\n    return heapq.nlargest(3, a)\n"
SORT = "def f(a):\n    return sorted(a)[:3]\n"


class TestClaimDetection:
    def test_a_recommendation_counts(self):
        assert claimed_techniques("You could use a hash map here.") == {"hash_maps"}

    def test_a_passing_mention_does_not(self):
        """Otherwise any sentence naming a concept becomes a claim."""
        assert claimed_techniques("Hash maps are a data structure.") == set()

    def test_a_bare_question_is_not_a_claim(self):
        assert claimed_techniques("What should the first step be?") == set()

    def test_empty_text_is_safe(self):
        assert claimed_techniques("") == set()


class TestScoring:
    def test_a_right_claim_scores_positive(self):
        out = evaluate(dlg(("tutor", "Try using a heap for this.")), FakeProblem(HEAP))
        assert out.score == 1.0 and out.correct == ["heaps"]

    def test_an_uncorroborated_claim_is_not_punished(self):
        """Most problems admit several approaches: 2 of 11 measured passing
        solutions used a different technique than their reference. Scoring
        those negative pays the tutor to recommend only what the dataset
        happens to contain."""
        out = evaluate(dlg(("tutor", "You should use a heap here.")), FakeProblem(SORT))
        assert out.score == 0.0
        assert out.uncorroborated == ["heaps"]
        assert out.claims == 1

    def test_silence_is_neutral_not_safe(self):
        """A penalty-only term would make saying nothing optimal, and the
        policy is already drifting that way (574 -> 393 chars over v24)."""
        out = evaluate(dlg(("tutor", "What do you notice about the input?")),
                       FakeProblem(SORT))
        assert out.score == 0.0 and out.claims == 0

    def test_mixed_claims_average(self):
        out = evaluate(
            dlg(("tutor", "Try sorting the list."), ("tutor", "You could use a heap.")),
            FakeProblem(SORT),
        )
        assert out.score == 0.5 and out.claims == 2

    def test_only_tutor_turns_are_scored(self):
        """The student guessing wrong is not the tutor being wrong."""
        out = evaluate(dlg(("student", "Should I use a heap?")), FakeProblem(SORT))
        assert out.claims == 0

    def test_unparseable_solution_scores_zero_rather_than_failing(self):
        out = evaluate(dlg(("tutor", "Use a heap.")), FakeProblem("def ("))
        assert out.score == 0.0

    def test_score_is_bounded_and_never_negative(self):
        """[0,1], not [-1,1]: absence from one reference is not refutation."""
        for sol in (HEAP, SORT):
            out = evaluate(
                dlg(("tutor", "Use a heap and sorting and binary search.")),
                FakeProblem(sol),
            )
            assert 0.0 <= out.score <= 1.0


class TestRewardIntegration:
    def test_default_weight_is_zero(self):
        """Measured inert on 120 dialogues before being wired in. Switching it
        on without a coverage measurement is the v16 mistake."""
        assert RewardSettings().mu_correct == 0.0
        assert Settings.kaggle().reward.mu_correct == 0.0

    def test_zero_weight_leaves_the_reward_identical(self):
        r = SAHAIReward(RewardSettings())
        a = r.compute(0.5, 1.0, 0.0, 0.3, correctness=0.0).r_sahai
        b = r.compute(0.5, 1.0, 0.0, 0.3, correctness=1.0).r_sahai
        assert a == b, "with mu_correct=0 the term must not move the reward"

    def test_a_nonzero_weight_moves_it_in_the_right_direction(self):
        s = RewardSettings()
        s.mu_correct = 0.5
        r = SAHAIReward(s)
        right = r.compute(0.5, 1.0, 0.0, 0.3, correctness=1.0).r_sahai
        quiet = r.compute(0.5, 1.0, 0.0, 0.3, correctness=0.0).r_sahai
        assert right > quiet

    def test_the_component_is_recorded_even_when_unweighted(self):
        """So a run reports the term's coverage instead of hiding it."""
        c = SAHAIReward(RewardSettings()).compute(0.5, 1.0, 0.0, 0.3, correctness=1.0)
        assert c.correctness == 1.0

    def test_the_hard_penalty_still_short_circuits(self):
        s = RewardSettings()
        s.mu_correct = 0.5
        r = SAHAIReward(s)
        assert r.compute(1.0, 0.0, 0.0, 0.0, correctness=1.0).r_sahai == -s.lambda_ped


class TestTrainerRecordsCoverage:
    """grpo.py by source: it imports torch, which this venv does not have."""

    @staticmethod
    def _src():
        return pathlib.Path("sahai/training/grpo.py").read_text()

    def test_correctness_is_computed_per_rollout(self):
        assert "evaluate_correctness(rollout.dialogue, rollout.problem)" in self._src()

    def test_coverage_reaches_the_metrics(self):
        src = self._src()
        assert "correctness_coverage" in src
        assert "mean_correctness" in src

    def test_claims_reach_the_rollout_dump(self):
        assert '"correctness_claims": r.correctness_claims' in self._src()
