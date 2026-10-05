"""The LLM hint judge: parsing, scoring, and the ways it must fail safe.

The model call is stubbed. What is tested is everything around it, which is
where a judge quietly goes wrong: a generation that does not parse, a verdict
read backwards, a penalty-only score that makes silence optimal.
"""

from __future__ import annotations

import pathlib

from sahai.core.dialogue import Dialogue
from sahai.reward.hint_judge import HintJudge, JudgeOutcome, parse_verdict


class FakeProblem:
    description = "Find the largest element."
    solution = "def f(a):\n    return max(a)\n"
    title = "t"


def dlg(*turns):
    d = Dialogue(problem_id="p")
    for role, text in turns:
        d.add(role, text)
    return d


class ScriptedJudge(HintJudge):
    """Returns verdicts from a list instead of calling a model."""

    def __init__(self, verdicts):
        super().__init__(model=object(), tokenizer=object())
        self._scripted = list(verdicts)
        self.seen: list[str] = []

    def judge_turn(self, turn, problem):
        self.seen.append(turn)
        return self._scripted.pop(0) if self._scripted else "NEUTRAL"


class TestParsing:
    def test_plain_verdicts(self):
        assert parse_verdict("HELPFUL") == "HELPFUL"
        assert parse_verdict("WRONG") == "WRONG"
        assert parse_verdict("NEUTRAL") == "NEUTRAL"

    def test_the_first_prompts_labels_still_parse(self):
        """Verdicts stored by the 2026-10-03 validation must stay readable."""
        assert parse_verdict("CORRECT") == "CORRECT"
        assert parse_verdict("MISLEADING") == "MISLEADING"

    def test_surrounding_noise_is_tolerated(self):
        assert parse_verdict("  helpful.\n") == "HELPFUL"
        assert parse_verdict("Answer: WRONG") == "WRONG"

    def test_a_negative_verdict_wins_when_both_words_appear(self):
        """A model emitting 'WRONG, not helpful' must not read as HELPFUL."""
        assert parse_verdict("MISLEADING, not correct") == "MISLEADING"
        assert parse_verdict("WRONG, not helpful") == "WRONG"

    def test_junk_defaults_to_neutral_not_misleading(self):
        """An unparseable generation must cost coverage, never invent a
        judgement against the tutor."""
        assert parse_verdict("I think the tutor") == "NEUTRAL"
        assert parse_verdict("") == "NEUTRAL"
        assert parse_verdict(None) == "NEUTRAL"


class TestScoring:
    def test_all_correct_scores_one(self):
        j = ScriptedJudge(["HELPFUL", "HELPFUL"])
        out = j.evaluate(dlg(("tutor", "a"), ("tutor", "b")), FakeProblem())
        assert out.score == 1.0 and out.judged == 2

    def test_all_misleading_scores_minus_one(self):
        j = ScriptedJudge(["WRONG"])
        assert j.evaluate(dlg(("tutor", "a")), FakeProblem()).score == -1.0

    def test_neutral_only_is_zero_not_negative(self):
        """Silence must be neutral. A penalty-only term makes saying nothing
        optimal, and the policy is already drifting that way."""
        j = ScriptedJudge(["NEUTRAL", "NEUTRAL"])
        out = j.evaluate(dlg(("tutor", "What do you think?"), ("tutor", "Go on")),
                         FakeProblem())
        assert out.score == 0.0 and out.judged == 0

    def test_neutral_turns_do_not_dilute_the_score(self):
        """One correct claim among three questions is still a correct claim."""
        j = ScriptedJudge(["NEUTRAL", "HELPFUL", "NEUTRAL"])
        out = j.evaluate(dlg(("tutor", "a"), ("tutor", "b"), ("tutor", "c")),
                         FakeProblem())
        assert out.score == 1.0 and out.judged == 1

    def test_mixed_averages(self):
        j = ScriptedJudge(["HELPFUL", "WRONG"])
        assert j.evaluate(dlg(("tutor", "a"), ("tutor", "b")), FakeProblem()).score == 0.0

    def test_only_tutor_turns_are_judged(self):
        j = ScriptedJudge(["WRONG"])
        out = j.evaluate(dlg(("student", "Should I use a heap?"), ("tutor", "x")),
                         FakeProblem())
        assert j.seen == ["x"]
        assert out.judged == 1

    def test_blank_turns_are_skipped(self):
        j = ScriptedJudge(["CORRECT"])
        j.evaluate(dlg(("tutor", "   "), ("tutor", "real")), FakeProblem())
        assert j.seen == ["real"]

    def test_score_is_bounded(self):
        for script in (["HELPFUL"] * 5, ["WRONG"] * 5, ["HELPFUL", "WRONG"] * 3):
            j = ScriptedJudge(script)
            d = dlg(*[("tutor", str(i)) for i in range(len(script))])
            assert -1.0 <= j.evaluate(d, FakeProblem()).score <= 1.0


class TestUnavailableJudge:
    def test_no_model_is_neutral_not_a_crash(self):
        """A missing judge must cost the term, never the run."""
        j = HintJudge(model=None, tokenizer=None)
        assert not j.available
        out = j.evaluate(dlg(("tutor", "Use a heap.")), FakeProblem())
        assert out.score == 0.0 and out.judged == 0


class TestGrounding:
    def test_the_prompt_contains_the_reference_solution(self):
        """The whole design. A judge asked 'is this good tutoring?' is one more
        opinion about text the policy already controls; a judge asked 'does
        this contradict this known-correct solution?' is anchored."""
        from sahai.reward.hint_judge import PROMPT

        assert "{solution}" in PROMPT
        assert "{description}" in PROMPT and "{turn}" in PROMPT

    def test_the_prompt_asks_for_one_word(self):
        from sahai.reward.hint_judge import PROMPT

        assert "one word" in PROMPT.lower()

    def test_the_prompt_does_not_lead_with_a_yes_no_question(self):
        """Regression. The first version asked "does this message contain a
        statement that is factually wrong?" and listed MISLEADING first. On
        120 real dialogues it answered MISLEADING for 109 of 109 turns --
        a constant, variance 0.0000, no gradient. The same judge on the same
        turns, asked to pick a label instead, returns a spread."""
        from sahai.reward.hint_judge import PROMPT

        assert "Does the tutor" not in PROMPT
        assert "factually wrong about" not in PROMPT

    def test_neutral_is_offered_before_the_negative_label(self):
        """Option order moved the verdict on its own."""
        from sahai.reward.hint_judge import PROMPT

        assert PROMPT.index("NEUTRAL -") < PROMPT.index("WRONG -")

    def test_the_prompt_says_the_reference_is_not_the_only_answer(self):
        """18% of measured passing solutions used a different technique than
        their reference. A judge anchored on it calls those wrong."""
        from sahai.reward.hint_judge import PROMPT

        assert "not the only one" in PROMPT
        assert "would also work is HELPFUL" in PROMPT

    def test_the_prompt_states_a_prior(self):
        """Without it the judge labels everything non-neutral."""
        from sahai.reward.hint_judge import PROMPT

        assert "Most tutor messages are NEUTRAL" in PROMPT

    def test_generation_is_greedy_and_short(self):
        src = pathlib.Path("sahai/reward/hint_judge.py").read_text()
        assert "do_sample=False" in src
        assert "max_new_tokens=self.max_new_tokens" in src


class TestTrainerIntegration:
    """grpo.py by source: it imports torch, which this venv does not have."""

    @staticmethod
    def _src():
        return pathlib.Path("sahai/training/grpo.py").read_text()

    def test_the_trainer_accepts_a_judge(self):
        assert "hint_judge=None" in self._src()

    def test_the_judge_replaces_the_vocabulary_detector_when_present(self):
        """Both return [-1, 1] with the same meaning. The vocabulary detector
        fires on 10% of dialogues, the 7B judge on 38-72%."""
        src = self._src()
        assert "if self.hint_judge is not None and self.hint_judge.available:" in src
        assert "else:" in src
        assert "evaluate_correctness(rollout.dialogue, rollout.problem)" in src

    def test_an_absent_judge_falls_back_rather_than_failing(self):
        """A judge that will not load must cost the term, never the run."""
        from sahai.reward.hint_judge import HintJudge

        assert HintJudge(None, None).available is False


class TestJudgeSettings:
    def test_mu_correct_is_still_off(self):
        """Enabling it is a training decision with a measured cost, not a
        default. The project's record on new reward terms is that they get
        gamed, so the configuration is set explicitly for the run that tests
        it."""
        from sahai.settings import Settings

        assert Settings.kaggle().reward.mu_correct == 0.0

    def test_judge_defaults_to_empty_so_the_fallback_is_the_ast_detector(self):
        from sahai.settings import Settings

        assert Settings.kaggle().reward.hint_judge_model == ""

    def test_four_bit_is_the_default_for_the_judge(self):
        """7B in bfloat16 is ~15.2 GB against a T4's 15.6 and leaves no room
        for activations; 4-bit is ~4.6 GB."""
        from sahai.settings import Settings

        assert Settings.kaggle().reward.hint_judge_4bit is True
