from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# One definition of the learner block, shared with the services so a policy
# trained on it meets the same wording when served. On Kaggle `sahai_core` is
# uploaded beside `sahai`; in a checkout it lives under libs/, which is what
# the fallback finds. If neither is present the block is simply empty and
# training proceeds exactly as it did before this existed.
try:  # pragma: no cover - import plumbing
    from sahai_core.learner_context import learner_context
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    _libs = _Path(__file__).resolve().parents[2] / "libs" / "sahai-core"
    if _libs.is_dir():
        sys.path.insert(0, str(_libs))
    try:
        from sahai_core.learner_context import learner_context
    except ImportError:
        def learner_context(skills, mastery):  # type: ignore[misc]
            return ""

if TYPE_CHECKING:
    from sahai.core.data import Problem


# Off by default, and deliberately so. The staged fixes (clipped surrogate,
# greedy partial-credit r_sol, difficulty-targeted ZPD) have never executed,
# and turning the prompt change on at the same time would repeat v14 — three
# variables changed at once, net negative, none of them attributable.
#
# Set SAHAI_LEARNER_CONTEXT=1 for the run that tests it, so the configuration
# is recorded in the environment rather than inferred from whether a directory
# happened to be uploaded.
LEARNER_CONTEXT_ENABLED = os.getenv("SAHAI_LEARNER_CONTEXT", "0") == "1"


@dataclass
class Turn:
    role: str
    content: str
    # False when generation stopped at max_new_tokens rather than on EOS. Without
    # this, truncation can only be guessed at from the text, which is unreliable.
    complete: bool = True


@dataclass
class Dialogue:
    problem_id: str
    turns: list[Turn] = field(default_factory=list)
    # The learner block the tutor was conditioned on when these turns were
    # generated. Recorded on the dialogue because the dialogue is the record
    # of what happened, and because GRPO has to score these turns under the
    # same prompt that produced them.
    #
    # Without this, `generate` built the system prompt with the block and
    # `compute_log_probs` rebuilt it without, so the importance ratio compared
    # two different conditionings: rho stopped being a ratio of one
    # distribution and the clipped surrogate was invalid. Dormant while
    # SAHAI_LEARNER_CONTEXT defaulted off, and silently wrong the moment the
    # experiment it exists for was run.
    learner_context: str = ""

    def add(self, role: str, content: str, complete: bool = True) -> None:
        self.turns.append(Turn(role=role, content=content, complete=complete))

    def to_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for turn in self.turns:
            role = "assistant" if turn.role == "tutor" else "user"
            messages.append({"role": role, "content": turn.content})
        return messages

    def tutor_text(self) -> str:
        return "\n".join(t.content for t in self.turns if t.role == "tutor")

    def student_text(self) -> str:
        return "\n".join(t.content for t in self.turns if t.role == "student")

    def __len__(self) -> int:
        return len(self.turns)


TERMINATION_PHRASES = [
    "i think i can solve it",
    "let me try",
    "i'll try now",
    "i got it",
    "i understand",
]


class DialogueEngine:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns

    def run(self, tutor, student, problem: Problem) -> Dialogue:
        dialogue = Dialogue(problem_id=problem.id)

        # Seed the opening turn from a persona template rather than sampling it.
        # A free-sampled opener made the 0.5B student answer the problem outright
        # (code in the first turn of 5/24 dialogues), and the tutor then replied
        # as if to a peer — the roles inverted from turn one.
        dialogue.add("student", student.opening_turn(problem))

        # The learner the tutor is teaching, as the tutor will see it at
        # serving. Built from the simulated student's own BKT state, which the
        # ZPD sampler is already moving, so the block genuinely varies across
        # rollouts instead of being a constant the policy can ignore.
        context = ""
        tracer = getattr(student, "tracer", None)
        if LEARNER_CONTEXT_ENABLED and tracer is not None and getattr(problem, "skills", None):
            try:
                context = learner_context(problem.skills, tracer.skills)
            except Exception:  # noqa: BLE001 - personalisation is not a precondition
                context = ""

        dialogue.learner_context = context

        for _ in range(self.max_turns):
            tutor_msg = tutor.generate(dialogue, problem, context)
            dialogue.add("tutor", tutor_msg, complete=tutor.last_generation_complete)

            if len(dialogue) >= self.max_turns * 2:
                break

            student_msg = student.respond(dialogue, problem)
            dialogue.add("student", student_msg, complete=student.last_generation_complete)

            if any(phrase in student_msg.lower() for phrase in TERMINATION_PHRASES):
                break

        return dialogue
