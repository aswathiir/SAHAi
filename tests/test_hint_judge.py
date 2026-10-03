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
        assert parse_verdict("CORRECT") == "CORRECT"
        assert parse_verdict("MISLEADING") == "MISLEADING"
        assert parse_verdict("NEUTRAL") == "NEUTRAL"

    def test_surrounding_noise_is_tolerated(self):
        assert parse_verdict("  correct.\n") == "CORRECT"
        assert parse_verdict("Answer: MISLEADING") == "MISLEADING"

    def test_misleading_wins_when_both_words_appear(self):
        """A model emitting 'MISLEADING, not correct' must not read as CORRECT."""
        assert parse_verdict("MISLEADING, not correct") == "MISLEADING"

    def test_junk_defaults_to_neutral_not_misleading(self):
        """An unparseable generation must cost coverage, never invent a
        judgement against the tutor."""
        assert parse_verdict("I think the tutor") == "NEUTRAL"
        assert parse_verdict("") == "NEUTRAL"
        assert parse_verdict(None) == "NEUTRAL"


class TestScoring:
    def test_all_correct_scores_one(self):
        j = ScriptedJudge(["CORRECT", "CORRECT"])
        out = j.evaluate(dlg(("tutor", "a"), ("tutor", "b")), FakeProblem())
        assert out.score == 1.0 and out.judged == 2

    def test_all_misleading_scores_minus_one(self):
        j = ScriptedJudge(["MISLEADING"])
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
        j = ScriptedJudge(["NEUTRAL", "CORRECT", "NEUTRAL"])
        out = j.evaluate(dlg(("tutor", "a"), ("tutor", "b"), ("tutor", "c")),
                         FakeProblem())
        assert out.score == 1.0 and out.judged == 1

    def test_mixed_averages(self):
        j = ScriptedJudge(["CORRECT", "MISLEADING"])
        assert j.evaluate(dlg(("tutor", "a"), ("tutor", "b")), FakeProblem()).score == 0.0

    def test_only_tutor_turns_are_judged(self):
        j = ScriptedJudge(["MISLEADING"])
        out = j.evaluate(dlg(("student", "Should I use a heap?"), ("tutor", "x")),
                         FakeProblem())
        assert j.seen == ["x"]
        assert out.judged == 1

    def test_blank_turns_are_skipped(self):
        j = ScriptedJudge(["CORRECT"])
        j.evaluate(dlg(("tutor", "   "), ("tutor", "real")), FakeProblem())
        assert j.seen == ["real"]

    def test_score_is_bounded(self):
        for script in (["CORRECT"] * 5, ["MISLEADING"] * 5, ["CORRECT", "MISLEADING"] * 3):
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

    def test_neutral_is_offered_before_misleading(self):
        """Option order moved the verdict on its own."""
        from sahai.reward.hint_judge import PROMPT

        assert PROMPT.index("NEUTRAL -") < PROMPT.index("MISLEADING -")

    def test_the_prompt_states_a_prior(self):
        """Without it the judge labels everything non-neutral."""
        from sahai.reward.hint_judge import PROMPT

        assert "Most tutor messages are NEUTRAL" in PROMPT

    def test_generation_is_greedy_and_short(self):
        src = pathlib.Path("sahai/reward/hint_judge.py").read_text()
        assert "do_sample=False" in src
        assert "max_new_tokens=self.max_new_tokens" in src
