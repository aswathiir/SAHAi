from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahai.core.data import Problem


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

        for _ in range(self.max_turns):
            tutor_msg = tutor.generate(dialogue, problem)
            dialogue.add("tutor", tutor_msg, complete=tutor.last_generation_complete)

            if len(dialogue) >= self.max_turns * 2:
                break

            student_msg = student.respond(dialogue, problem)
            dialogue.add("student", student_msg, complete=student.last_generation_complete)

            if any(phrase in student_msg.lower() for phrase in TERMINATION_PHRASES):
                break

        return dialogue
