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
        """Mean of three checks, each scored over the tutor's turns.

        **All three ask the same question: did the tutor give the answer away?**
        Nothing here scores teaching style any more.

        Two checks were removed after the question-fraction change cost a run
        (held-out solve 16.3% -> 6.2%; see docs/04-findings.md entry 11). The
        line they failed is not "execution-verifiable" — execution lives in the
        leakage term — but this:

            keep a check when satisfying the rule and achieving the goal are
            the same act; drop it when the rule is a proxy that can be
            satisfied without the goal.

        "Do not write code" has no fake version: either the tutor emitted a
        fence or it did not, and not emitting one is exactly the behaviour
        wanted. "Ask questions" has an obvious fake version — a question mark —
        and the policy found it in a single run. "Under 200 words" is an
        arbitrary cutoff and was the vector for the original length bias.

        What is lost is real: nothing now rewards Socratic teaching, so `r_sol`
        carries that signal alone — a tutor is rewarded when the student solves
        the problem afterwards, which is the outcome the project actually cares
        about and the one measure that has never been gamed. That is a
        deliberate narrowing, not an oversight.

        Scoring stays per turn rather than all-or-nothing: a conjunctive check
        over the whole dialogue made the score fall purely with length
        (measured across the v11 run: 0.720 / 0.639 / 0.537 / 0.453 at one to
        four tutor turns), which GRPO reads as "shorter dialogues teach better".
        """
        # A dialogue with no tutor turns must score zero, not 1.0. Every check
        # here is a "no X" and passes vacuously when there is nothing to
        # inspect, so a silent tutor would otherwise score perfectly — and in
        # training the policy could learn exactly that.
        tutor_turns = [t for t in dialogue.turns if t.role == "tutor"]
        if not tutor_turns:
            return 0.0

        checks = [
            self._no_code_blocks(tutor_turns),
            self._no_solution_patterns(tutor_turns),
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
