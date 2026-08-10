from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sahai_core.domain.dialogue import Dialogue

RUBRICS = [
    (
        "Evaluate this tutoring dialogue for pedagogical quality.\n"
        "ACCEPT if ALL conditions are met:\n"
        "1. The tutor NEVER gives the direct solution or code.\n"
        "2. The tutor uses scaffolded hints that build understanding.\n"
        "3. The tutor encourages the student to think independently.\n"
        "4. The tutor responds appropriately to student mistakes.\n"
        "REJECT if any condition is violated.\n"
        "Respond with exactly one word: ACCEPT or REJECT."
    ),
    (
        "Evaluate this tutoring dialogue for pedagogical quality.\n"
        "ACCEPT if ALL conditions are met:\n"
        "1. The tutor does not leak the answer through examples that mirror the solution.\n"
        "2. The tutor adapts hint specificity to student progress.\n"
        "3. The tutor maintains productive struggle without causing frustration.\n"
        "4. The tutor uses language matching the student's register.\n"
        "REJECT if any condition is violated.\n"
        "Respond with exactly one word: ACCEPT or REJECT."
    ),
]

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
SOLUTION_INDICATORS = [
    r"def\s+\w+\s*\(",
    r"return\s+\[",
    r"for\s+\w+\s+in\s+range",
    r"while\s+\w+\s*[<>=]",
]


class RuleBasedJudge:
    """Heuristic pedagogy judge — no LLM needed. Usable on constrained hardware."""

    def evaluate(self, dialogue: Dialogue) -> float:
        """Fraction of checks passed.

        All-or-nothing scoring collapses a GRPO group to a single reward value,
        which zeroes every advantage and kills the gradient. Grading keeps
        within-group variance so the update has something to learn from.
        """
        # A dialogue with no tutor turns must score zero, not 0.8. Every "no X"
        # check passes vacuously when there is nothing to inspect, so silence
        # would otherwise be rewarded as good teaching — and in training the
        # policy could learn exactly that.
        if not any(t.role == "tutor" for t in dialogue.turns):
            return 0.0

        checks = [
            self._no_code_blocks(dialogue),
            self._no_solution_patterns(dialogue),
            self._tutor_asks_questions(dialogue),
            self._reasonable_length(dialogue),
            self._no_dangling_promises(dialogue),
        ]
        return sum(checks) / len(checks)

    def _no_dangling_promises(self, dialogue: Dialogue) -> bool:
        """A tutor turn ending in ':' promises content it never delivers.

        The student then completes the promise instead of answering, which is
        how the two roles invert. The tutor is the policy, so this cannot be
        fixed by post-processing — it has to be priced into the reward.
        """
        for turn in dialogue.turns:
            if turn.role == "tutor" and turn.content.rstrip().endswith(":"):
                return False
        return True

    def _no_code_blocks(self, dialogue: Dialogue) -> bool:
        for turn in dialogue.turns:
            if turn.role != "tutor":
                continue
            # Bare "```" catches blocks left unclosed by a length-truncated turn.
            if CODE_BLOCK_PATTERN.search(turn.content) or "```" in turn.content:
                return False
        return True

    def _no_solution_patterns(self, dialogue: Dialogue) -> bool:
        for turn in dialogue.turns:
            if turn.role != "tutor":
                continue
            for pattern in SOLUTION_INDICATORS:
                if re.search(pattern, turn.content):
                    return False
        return True

    def _tutor_asks_questions(self, dialogue: Dialogue) -> bool:
        tutor_turns = [t for t in dialogue.turns if t.role == "tutor"]
        if not tutor_turns:
            return False
        questions = sum(1 for t in tutor_turns if "?" in t.content)
        return questions >= len(tutor_turns) * 0.3

    def _reasonable_length(self, dialogue: Dialogue) -> bool:
        for turn in dialogue.turns:
            if turn.role == "tutor" and len(turn.content.split()) > 200:
                return False
        return True


class PedagogyReward:
    def __init__(
        self,
        model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
        num_judges: int = 2,
        use_rules: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.rubrics = RUBRICS[:num_judges]
        self.rule_judge = RuleBasedJudge()
        self.use_rules = use_rules or (model is None)

    def _judge_once(self, dialogue: Dialogue, rubric: str) -> bool:
        import torch

        dialogue_text = "\n".join(
            f"{'Tutor' if t.role == 'tutor' else 'Student'}: {t.content}"
            for t in dialogue.turns
        )
        messages = [
            {"role": "system", "content": rubric},
            {"role": "user", "content": f"Dialogue:\n{dialogue_text}"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip().upper()
        return "ACCEPT" in response

    def evaluate(self, dialogue: Dialogue) -> float:
        if self.use_rules:
            return self.rule_judge.evaluate(dialogue)
        votes = [self._judge_once(dialogue, rubric) for rubric in self.rubrics]
        if not votes:
            return 0.0
        return sum(votes) / len(votes)
