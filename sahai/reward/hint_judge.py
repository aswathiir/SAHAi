"""Is this tutor turn factually wrong about this problem?

The technique-vocabulary detector in `correctness.py` is sound but has almost
nothing to read: measured on 120 benchmark dialogues it fires on 10% of them,
because the trained tutor asks questions in 65% of its turns and names one of
the fifteen known techniques 16 times in 150 turns. Coverage, not design, is
what stopped it being trainable.

A model-based judge has coverage over any claim in any wording. That is its
only advantage, and it buys that advantage at a real price.

**The judge sees the reference solution.** This is the part that matters. A
judge asked "is this good tutoring?" is one more opinion about the tutor's
text, which is exactly the channel the policy already knows how to satisfy --
`r_ped` reached 1.000 on vacuous Hinglish. A judge asked "does this statement
contradict this known-correct solution?" is anchored to something the tutor
cannot edit. Same principle as `correctness.py`, wider aperture.

**Judges are gameable, and this one is not exempt.** A policy trained against
it will find whatever it rewards, including phrasings that read as confident
and correct to a small model. That is not a reason to skip it; it is a reason
to measure its coverage and its agreement with outcomes *before* training on
it, and to keep measuring after. `mu_correct` stays 0.0 until those numbers
exist, exactly as it did for the vocabulary detector.

**One reference is not the only correct answer.** The prompt constrains WRONG
rather than encouraging HELPFUL, and the labels were renamed to make it harder to slip back:
"CORRECT" invites the judge to check agreement with the solution shown, while
"HELPFUL" asks whether the advice leads somewhere that works. Measured on 23
passing student solutions, 2 of the 11 with a detectable technique used a
different one than their reference -- `opposite_Signs` is `(x ^ y) < 0` in the
dataset and `x * y < 0` in a solution that passes every test. A judge anchored
on the reference calls the second wrong.

Phrasing this as three sentences urging HELPFUL collapsed the 1.5B judge the
other way -- every turn HELPFUL, variance 0.0000, exactly as useless as the
first prompt's every-turn-MISLEADING. The caveat is now a single clause
narrowing when WRONG applies, which leaves the NEUTRAL prior intact.

**It is a judgement, not a fact.** `correctness.py` compares against an AST
and is wrong only where its vocabulary is thin. This is a 1.5B-class model
reading another 1.5B-class model, and its verdicts carry that model's errors.
Reported agreement figures are the honest measure of how much to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahai.core.data import Problem
    from sahai.core.dialogue import Dialogue

# Single words, checked by prefix, so a judge that adds punctuation or a stray
# token still parses. Ordered: MISLEADING is checked first because a model that
# emits "MISLEADING, not correct" must not be read as CORRECT.
#
# The prompt below is the second one written. The first asked "does this
# message contain a statement that is factually wrong?", listed MISLEADING
# first among the options, and stated no prior. Measured on 120 dialogues it
# returned MISLEADING for 109 of 109 tutor turns -- a constant, with variance
# 0.0000 and therefore no gradient at all. The bias was the prompt, not the
# model: the same 1.5B judge on the same turns, asked to pick a label rather
# than answer a leading yes/no question, returns a spread. Three things fixed
# it, and all three matter: no leading question, NEUTRAL listed first, and an
# explicit prior that most messages are neutral.
VERDICTS = ("MISLEADING", "WRONG", "HELPFUL", "CORRECT", "NEUTRAL")
# MISLEADING/CORRECT are the first prompt's labels, kept so a stored
# verdict from the 2026-10-03 validation still parses.
_GOOD = {"HELPFUL", "CORRECT"}
_BAD = {"WRONG", "MISLEADING"}

PROMPT = """Classify ONE tutor message.

PROBLEM:
{description}

ONE SOLUTION THAT WORKS (there may be others):
{solution}

TUTOR MESSAGE:
{turn}

Which label fits the tutor message best?

NEUTRAL - asks a question, encourages, or says nothing technical
HELPFUL - makes a technical claim that would lead to a working solution
WRONG - makes a technical claim that is false, or would lead to a solution \
that does not pass

Most tutor messages are NEUTRAL. Only use HELPFUL or WRONG when the message \
makes a definite technical claim.

Use WRONG only for a claim that is false or that leads to a solution failing \
the tests. A different approach that would also work is not WRONG.

Reply with exactly one word: NEUTRAL, HELPFUL, or WRONG."""


@dataclass
class JudgeOutcome:
    """Verdicts and the score they produce, kept together.

    The per-turn verdicts are retained because the score alone is unreadable
    afterwards, and the reason this term exists at all is that `r_ped` scored
    1.0 on empty text and the number hid it.
    """

    score: float
    verdicts: list[str] = field(default_factory=list)

    @property
    def judged(self) -> int:
        """Turns that produced a non-NEUTRAL verdict, i.e. real coverage."""
        return sum(1 for v in self.verdicts if v in _GOOD | _BAD)


def parse_verdict(text: str) -> str:
    """Map a raw generation to one of VERDICTS, defaulting to NEUTRAL.

    Defaulting to NEUTRAL rather than guessing is deliberate: NEUTRAL scores
    0.0, so an unparseable generation costs coverage instead of inventing a
    judgement. The alternative, treating junk as MISLEADING, would punish the
    tutor for the judge's failures.
    """
    up = (text or "").strip().upper()
    for v in VERDICTS:
        if v in up:
            return v
    return "NEUTRAL"


class HintJudge:
    """Scores a dialogue's tutor turns against the problem's known solution."""

    def __init__(self, model=None, tokenizer=None, max_new_tokens: int = 6):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def judge_turn(self, turn: str, problem: Problem) -> str:
        if not self.available:
            return "NEUTRAL"
        import torch

        prompt = PROMPT.format(
            description=problem.description,
            solution=problem.solution,
            turn=turn.strip(),
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return parse_verdict(self.tokenizer.decode(gen, skip_special_tokens=True))

    def evaluate(self, dialogue: Dialogue, problem: Problem) -> JudgeOutcome:
        """Score in [-1, 1]: (correct - misleading) / judged, 0.0 if none.

        Symmetric for the same reason as `correctness.py`: a penalty-only term
        makes silence optimal, and the policy is already drifting toward
        silence without being paid to.
        """
        verdicts: list[str] = []
        for turn in dialogue.turns:
            if turn.role != "tutor" or not turn.content.strip():
                continue
            verdicts.append(self.judge_turn(turn.content, problem))

        good = sum(1 for v in verdicts if v in _GOOD)
        bad = sum(1 for v in verdicts if v in _BAD)
        total = good + bad
        score = 0.0 if total == 0 else (good - bad) / total
        return JudgeOutcome(score=score, verdicts=verdicts)
