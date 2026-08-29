from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sahai.core.dialogue import Dialogue

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
        """Mean of five checks, each scored over the tutor's turns.

        All-or-nothing scoring collapses a GRPO group to a single reward value,
        which zeroes every advantage and kills the gradient. Grading keeps
        within-group variance so the update has something to learn from.

        **The four "never do X" checks are scored per turn, not per dialogue.**
        They used to fail the whole dialogue if *any* single tutor turn
        violated them. That made the score fall purely as a function of length:
        measured over the 304 rollouts of the v11 run, mean r_ped was 0.720 at
        one tutor turn, 0.639 at two, 0.537 at three, 0.453 at four — a clean
        monotonic slide, because more turns simply meant more chances to trip a
        conjunctive check. GRPO reads that as "shorter dialogues teach better"
        and optimises for ending the conversation, which is an artifact of this
        function rather than anything about teaching.

        Scoring each check as the *fraction of tutor turns that pass* removes
        that: adding a turn no worse than the others now leaves the score
        unchanged. It also gives a finer-grained signal — three clean turns and
        one with code scores 0.75 on that check instead of 0.
        """
        # A dialogue with no tutor turns must score zero, not 0.8. Every "no X"
        # check passes vacuously when there is nothing to inspect, so a rollout
        # where the tutor stays silent would be rewarded as good teaching.
        tutor_turns = [t for t in dialogue.turns if t.role == "tutor"]
        if not tutor_turns:
            return 0.0

        checks = [
            self._no_code_blocks(tutor_turns),
            self._no_solution_patterns(tutor_turns),
            self._tutor_asks_questions(tutor_turns),
            self._reasonable_length(tutor_turns),
            self._no_dangling_promises(tutor_turns),
        ]
        return sum(checks) / len(checks)

    @staticmethod
    def _fraction(turns: list, predicate) -> float:
        """Share of tutor turns satisfying `predicate` — the length-neutral form."""
        return sum(1 for t in turns if predicate(t.content)) / len(turns)

    def _no_dangling_promises(self, tutor_turns: list) -> float:
        """A tutor turn ending in ':' promises content it never delivers.

        The student then completes the promise instead of answering, which is
        how the two roles invert. The tutor is the policy, so this cannot be
        fixed by post-processing — it has to be priced into the reward.
        """
        return self._fraction(tutor_turns, lambda c: not c.rstrip().endswith(":"))

    def _no_code_blocks(self, tutor_turns: list) -> float:
        # Bare "```" catches blocks left unclosed by a length-truncated turn.
        return self._fraction(
            tutor_turns,
            lambda c: not (CODE_BLOCK_PATTERN.search(c) or "```" in c),
        )

    def _no_solution_patterns(self, tutor_turns: list) -> float:
        return self._fraction(
            tutor_turns,
            lambda c: not any(re.search(p, c) for p in SOLUTION_INDICATORS),
        )

    def _tutor_asks_questions(self, tutor_turns: list) -> float:
        """Left as a ratio-with-threshold: it was never length-biased.

        Unlike the "never do X" checks, this one already normalises by turn
        count, so adding turns does not systematically hurt it.
        """
        questions = sum(1 for t in tutor_turns if "?" in t.content)
        return float(questions >= len(tutor_turns) * 0.3)

    def _reasonable_length(self, tutor_turns: list) -> float:
        return self._fraction(tutor_turns, lambda c: len(c.split()) <= 200)


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
