"""Is what the tutor said actually true?

Nothing in the reward asked this before. `r_ped` is three checks and all of
them are negative -- no dangling promises, no code blocks, no solution patterns
-- and leakage asks whether the answer escaped. A tutor that confidently
recommends Heron's formula for a problem that does not need it satisfies every
one of them and scores `r_ped = 1.0`, which is what the 2026-10-02 benchmark
caught it doing (findings #15, #16).

**This term is dangerous and the danger is specific.** Adding an incentive to
the channel the policy already controls is the move that failed in v16: making
`_tutor_asks_questions` a per-turn fraction gave an inert check a gradient, the
policy optimised it directly, and held-out solve fell 16.3% -> 6.2%. Two design
choices exist to avoid repeating that.

**Ground truth lives outside the tutor's text.** The claim "this problem needs
a heap" is checked against `techniques_used(problem.solution)`, which is an AST
pass over the reference solution. The tutor cannot write its way into being
right; it can only be right or wrong about a fact it does not control. That is
the property `r_ped` lacks and the reason this is not simply a fourth rule.

**Silence is neutral, not safe.** Scoring this as a penalty would make saying
nothing optimal, and the policy is already drifting that way -- mean tutor turn
length fell 574 -> 393 characters over v24, and the benchmark found the
quietest third of tutoring outscored the most verbose third 0.350 to 0.200. A
tutor rewarded for silence is not a tutor. So the score is symmetric: no claims
scores 0.0, correct claims score positive, wrong claims negative. The gradient
points at speaking *correctly*, which is the behaviour the project wants and
has never once asked for.

Known limits, stated rather than discovered later:

* Coverage is the fifteen techniques in `sahai_core.techniques`. The Heron's
  formula case that motivated this is **not** among them, so this term would
  not have caught it. It catches "use a heap" on a sorting problem, which is
  the same failure in a detectable form.
* Negation is not parsed. "This is not a binary search problem" reads as a
  binary search claim. Measured on the v24 rollouts below, negated mentions are
  rare, but they are miscounted when they occur.
* A technique absent from the reference solution is not always wrong -- most
  problems admit several approaches. This measures agreement with *one* known
  good solution, which is a proxy for correctness and not correctness itself.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahai.core.data import Problem
    from sahai.core.dialogue import Dialogue

try:  # pragma: no cover - import plumbing, same shape as core/dialogue.py
    from sahai_core.techniques import TECHNIQUE_SKILLS, techniques_used
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    _libs = _Path(__file__).resolve().parents[2] / "libs" / "sahai-core"
    if _libs.is_dir():
        sys.path.insert(0, str(_libs))
    try:
        from sahai_core.techniques import TECHNIQUE_SKILLS, techniques_used
    except ImportError:
        TECHNIQUE_SKILLS = {}

        def techniques_used(source):  # type: ignore[misc]
            return set()


@dataclass
class CorrectnessOutcome:
    """Why the score came out as it did, for the rollout dumps.

    The score alone is unreadable after the fact, and the whole reason this
    term exists is that a previous reward term scored 1.0 on vacuous text and
    nobody could see it from the number.
    """

    score: float
    correct: list[str]
    wrong: list[str]

    @property
    def claims(self) -> int:
        return len(self.correct) + len(self.wrong)


# Phrases that mark a recommendation rather than a passing mention. Without
# this, "arrays" in any sentence counts as advocating an array technique.
_RECOMMEND = re.compile(
    r"(?i)\b(use|using|try|consider|apply|need|needs|want|should|could|"
    r"maintain|keep|build|reach for|think about|how about|what about)\b"
)


def claimed_techniques(text: str) -> set[str]:
    """Techniques this turn appears to recommend.

    Requires a recommending verb somewhere in the sentence containing the
    phrase. "A heap would work here" counts; "heaps are a data structure"
    does not.
    """
    found: set[str] = set()
    if not text:
        return found
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if not _RECOMMEND.search(sentence):
            continue
        low = sentence.lower()
        for phrase, skill in TECHNIQUE_SKILLS.items():
            if phrase.lower() in low:
                found.add(skill)
    return found


def evaluate(dialogue: Dialogue, problem: Problem) -> CorrectnessOutcome:
    """Score the tutor's technical claims against the reference solution.

    Returns a value in [-1, 1]:

        (correct - wrong) / (correct + wrong)

    with 0.0 when the tutor made no checkable claim at all. Dividing by the
    number of claims rather than by the number of turns is deliberate: a tutor
    that makes one correct claim should not be scored down for the turns in
    which it asked a question instead.
    """
    # Parseability is checked separately and first. `techniques_used` catches
    # SyntaxError itself and returns an empty set, which is indistinguishable
    # from "this solution uses no detectable technique" -- and the difference
    # matters enormously. An empty set means every claim is scored wrong, so a
    # reference solution the AST cannot read would punish the tutor for saying
    # anything at all, on no evidence. No ground truth means no opinion.
    try:
        ast.parse(problem.solution)
    except (SyntaxError, ValueError, TypeError):
        return CorrectnessOutcome(0.0, [], [])

    try:
        actual = techniques_used(problem.solution)
    except Exception:  # noqa: BLE001 - defensive; the parse above is the real guard
        return CorrectnessOutcome(0.0, [], [])

    actual_skills = {TECHNIQUE_SKILLS[p] for p in actual if p in TECHNIQUE_SKILLS}

    correct: list[str] = []
    wrong: list[str] = []
    for turn in dialogue.turns:
        if turn.role != "tutor":
            continue
        for skill in claimed_techniques(turn.content):
            (correct if skill in actual_skills else wrong).append(skill)

    total = len(correct) + len(wrong)
    if total == 0:
        return CorrectnessOutcome(0.0, correct, wrong)
    return CorrectnessOutcome((len(correct) - len(wrong)) / total, correct, wrong)
