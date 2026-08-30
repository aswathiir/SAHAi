"""Leakage — did the tutor give the answer away?

The original estimator compared tutor text against the *reference* solution by
token overlap. Measured on a real rollout, that scored **0.000** for a tutor
that wrote a complete, working solution, because it used a dict where the MBPP
reference used `enumerate`/`count`. It scored 1.000 only for a verbatim paste.

It was measuring plagiarism of one particular implementation, not leakage.

So the primary signal is now execution: extract any code the tutor wrote and run
it against the problem's own tests. Code that passes **is** the answer,
regardless of how it is written. That cannot be dodged by renaming variables or
choosing a different algorithm.

This module does no execution — running untrusted code is a privileged
operation that belongs to the caller (`CodeVerifier` in training,
`services/executor` in serving). Here we only score the outcome.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahai_core.domain.dialogue import Dialogue

FENCED_CODE = re.compile(r"```(?:\w+)?\n?(.*?)(?:```|$)", re.DOTALL)

# A solution token only counts as leakage if it appears in at most this many
# solutions across the whole problem bank.
#
# Rationale, measured on the 228 code-free dialogues of the v14 run. Plain
# overlap charged the tutor for ordinary teaching vocabulary — `count`, `sum`,
# `append`, `index` each appear in 14-19 of 198 MBPP solutions — so every extra
# sentence raised leakage and the reward paid the tutor to say less
# (corr(verbosity, leak) = +0.380). Restricting to the rare tail keeps the real
# signal (`first_repeated_char`, `reverse_words`, `prime_num` each appear in
# exactly one solution and are unambiguous give-aways) while letting the tutor
# discuss the problem freely:
#
#   max_df   mean leak   corr(verbosity)   dialogues with signal
#      1       0.0252        +0.060              9.6%
#      2       0.0406        +0.160             13.6%   <- chosen
#      3       0.0565        +0.207             18.0%
#      5       0.0537        +0.224             19.3%
#
# 1 almost eliminates the length penalty but fires so rarely it degenerates
# into execution-only leakage; 2 keeps prose-leak detection while more than
# halving the pressure toward silence.
RARE_TOKEN_MAX_DF = 2

# Words that carry no evidence of leakage: Python's own vocabulary.
PROGRAMMING_NOISE = {
    "def", "return", "for", "while", "if", "else", "elif", "in", "range",
    "len", "true", "false", "none", "import", "from", "class", "self",
    "print", "int", "str", "list", "dict", "set", "tuple",
}

# Code present but not verified as a working solution. Still a strong signal —
# the tutor is writing code at a student — but not proof the answer was given.
UNVERIFIED_CODE_LEAKAGE = 0.5


class LeakageEstimator:
    def __init__(
        self,
        token_weight: float = 0.7,
        ace_weight: float = 0.3,
        solution_corpus: list[str] | None = None,
        max_doc_freq: int = RARE_TOKEN_MAX_DF,
    ):
        self.token_weight = token_weight
        self.ace_weight = ace_weight
        self.max_doc_freq = max_doc_freq
        self._doc_freq: Counter[str] = Counter()
        if solution_corpus:
            self.fit(solution_corpus)

    def fit(self, solution_corpus: list[str]) -> None:
        """Count how many solutions each token appears in.

        Must be fitted on the **same** corpus for training and evaluation, or
        the two score leakage by different rules — the exact class of bug that
        made held-out leakage incomparable to training leakage for every run
        before this one.
        """
        self._doc_freq = Counter()
        for solution in solution_corpus:
            for token in self._meaningful_tokens(solution):
                self._doc_freq[token] += 1

    @property
    def is_fitted(self) -> bool:
        return bool(self._doc_freq)

    def _rare(self, tokens: set[str]) -> set[str]:
        """Keep only tokens distinctive enough to be evidence of a leak.

        Unfitted, this is the identity — every token counts, i.e. the old
        behaviour. That is deliberate: silently applying a filter with no
        corpus behind it would be worse than not filtering at all.
        """
        if not self.is_fitted:
            return tokens
        return {t for t in tokens if self._doc_freq.get(t, 0) <= self.max_doc_freq}

    # --- signals the caller may need to gather -----------------------------

    @staticmethod
    def extract_tutor_code(dialogue: Dialogue) -> list[str]:
        """Fenced code blocks written by the tutor.

        Pure. The caller decides whether to run these; that is the only way to
        tell a complete leak from a fragment.
        """
        blocks: list[str] = []
        for turn in dialogue.turns:
            if turn.role != "tutor":
                continue
            blocks.extend(
                block.strip()
                for block in FENCED_CODE.findall(turn.content)
                if block.strip()
            )
        return blocks

    @staticmethod
    def defined_functions(code: str) -> list[str]:
        """Names of functions defined in a code block."""
        return re.findall(r"^\s*def\s+(\w+)\s*\(", code, re.MULTILINE)

    @classmethod
    def runnable_variants(cls, code: str, expected_name: str) -> list[str]:
        """The block, plus aliased versions so the tests can call it.

        A tutor leaking the answer rarely uses the same function name as the
        reference — here it wrote `find_first_repeated_char` where the problem
        expected `first_repeated_char`. Without an alias the verifier cannot
        call it, the block appears not to solve anything, and a complete leak
        scores zero. Renaming a function does not make it less of a leak.
        """
        names = cls.defined_functions(code)
        if not names or expected_name in names:
            return [code]
        return [f"{code}\n\n{expected_name} = {name}\n" for name in names]

    # --- scoring ------------------------------------------------------------

    def token_match_leakage(
        self, dialogue: Dialogue, solution: str, problem_text: str = ""
    ) -> float:
        """Secondary signal: overlap with solution tokens.

        `problem_text` is subtracted because words from the question itself
        ("string", "character", the function name) are not leakage — a tutor
        cannot discuss the problem without using them. Omitting this gave the
        metric a high floor that made its absolute value meaningless.
        """
        solution_tokens = self._rare(
            self._meaningful_tokens(solution) - self._meaningful_tokens(problem_text)
        )
        if not solution_tokens:
            return 0.0

        tutor_tokens = set(re.findall(r"\b\w+\b", dialogue.tutor_text().lower()))
        return len(solution_tokens & tutor_tokens) / len(solution_tokens)

    def ace_leakage(self, solve_with: float, solve_without: float) -> float:
        """Assistance-corrected effect: solved without the hints mattering.

        From the literature review. Still not wired into training — the caller
        would have to run randomized hint ablations to supply these.
        """
        ace = solve_with - solve_without
        return 1.0 if solve_with > 0.8 and ace < 0.1 else 0.0

    def estimate(
        self,
        dialogue: Dialogue,
        solution: str,
        problem_text: str = "",
        tutor_code_solves: bool = False,
        solve_with: float | None = None,
        solve_without: float | None = None,
    ) -> float:
        """Leakage in [0, 1].

        `tutor_code_solves` is execution-verified by the caller: True when code
        the tutor wrote passes the problem's own tests. That is a complete leak
        by definition and short-circuits everything else.
        """
        if tutor_code_solves:
            return 1.0

        score = self.token_match_leakage(dialogue, solution, problem_text)

        # The tutor wrote code that did not pass — a fragment, or broken. Still
        # worse than any amount of talking about the problem.
        if self.extract_tutor_code(dialogue):
            score = max(score, UNVERIFIED_CODE_LEAKAGE)

        if solve_with is not None and solve_without is not None:
            ace = self.ace_leakage(solve_with, solve_without)
            score = self.token_weight * score + self.ace_weight * ace

        return min(score, 1.0)

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower())) - PROGRAMMING_NOISE
