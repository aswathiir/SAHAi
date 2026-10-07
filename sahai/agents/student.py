from __future__ import annotations

import ast
import os
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sahai.agents.tracer import BKTTracer
from sahai.core.generation import (
    drop_dangling_promise,
    hit_terminator,
    terminator_ids,
    trim_incomplete,
)
from sahai.settings import TracerSettings

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sahai.core.data import Problem
    from sahai.core.dialogue import Dialogue


MISCONCEPTIONS = {
    "arrays": ["confuses index with value", "off-by-one in loops"],
    "hash_maps": ["forgets to handle collisions", "uses list instead of dict"],
    "linked_lists": ["loses reference to head", "forgets null check"],
    "stacks": ["pops from empty stack", "confuses LIFO with FIFO"],
    "binary_search": ["wrong midpoint calculation", "infinite loop on boundaries"],
    "recursion": ["missing base case", "returns wrong value in recursive call"],
    "dynamic_programming": ["wrong state transition", "forgets base case initialization"],
    "trees": ["confuses preorder with inorder", "forgets leaf node check"],
    "sorting": ["wrong comparison operator", "doesn't handle equal elements"],
    "two_pointers": ["moves wrong pointer", "skips valid pairs"],
    "strings": ["off-by-one in slicing", "mutates string in-place"],
    "dfs": ["forgets visited check", "wrong backtracking"],
}

CODE_MIXING_TEMPLATES = {
    "hinglish": {
        "confused": "Mujhe samajh nahi aa raha hai {topic} kaise kaam karta hai.",
        "partial": "Main {topic} use kar raha hoon but {issue} ho raha hai.",
        "asking": "Kya hum {approach} use kar sakte hain yahan pe?",
        "understood": "Achha, toh {concept} ka matlab hai ki {explanation}. Let me try now.",
    },
    "tanglish": {
        "confused": "Enaku {topic} puriyala, epdhi work aagum?",
        "partial": "Naan {topic} use panniten but {issue} varuthu.",
        "asking": "Inga {approach} use panlama?",
        "understood": "Oh okay, {concept} na {explanation} dhaane. I'll try now.",
    },
    "english": {
        "confused": "I don't understand how {topic} works here.",
        "partial": "I'm using {topic} but I'm getting {issue}.",
        "asking": "Can we use {approach} here?",
        "understood": "Oh I see, so {concept} means {explanation}. Let me try now.",
    },
}


@dataclass
class StudentPersona:
    ability_level: int = 2
    misconceptions: list[str] = field(default_factory=list)
    code_mixing_ratio: float = 0.5
    language: str = "hinglish"
    persistence: float = 0.7

    def build_system_prompt(self, problem: Problem) -> str:
        skill_errors = []
        for skill in problem.skills:
            if skill in MISCONCEPTIONS:
                skill_errors.extend(MISCONCEPTIONS[skill])
        if self.misconceptions:
            skill_errors.extend(self.misconceptions)

        templates = CODE_MIXING_TEMPLATES.get(self.language, CODE_MIXING_TEMPLATES["english"])
        example = templates["confused"].format(topic="this concept")

        return (
            f"You are a weak student (ability {self.ability_level}/5) struggling with: {problem.title}.\n"
            f"Mistakes you make: {', '.join(skill_errors[:3]) if skill_errors else 'general confusion'}.\n\n"
            f"RULES:\n"
            f"- NEVER write code or solutions. You don't know how to solve this yet.\n"
            f"- Speak in {self.language}. Example: '{example}'\n"
            f"- Ask short questions. Show confusion. Keep responses to 1-2 sentences.\n"
            f"- When you finally understand the approach, say exactly: 'I think I can solve it now'\n"
        )



def _parse_salvaging_tail(body: str) -> tuple[ast.Module, str] | None:
    """Parse `body`, dropping trailing lines until it parses.

    A generation cut off mid-statement leaves a complete function followed by a
    fragment; discarding the whole reply over the fragment loses the part that
    works. Returns the tree and the text it was parsed from, so callers alias
    against exactly what parsed.

    Module level rather than a method because it touches no instance state.
    """
    try:
        return ast.parse(body), body
    except SyntaxError:
        pass
    lines = body.split("\n")
    for cut in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:cut])
        try:
            return ast.parse(candidate), candidate
        except SyntaxError:
            continue
    return None


class StudentSimulator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        persona: StudentPersona | None = None,
        tracer_settings: TracerSettings | None = None,
        max_new_tokens: int = 160,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.persona = persona or StudentPersona()
        self.tracer = BKTTracer(tracer_settings)
        self.max_new_tokens = max_new_tokens
        self._terminators = terminator_ids(tokenizer, model)
        self.last_generation_complete = True

    def opening_turn(self, problem: Problem) -> str:
        """Scripted first turn expressing confusion in the persona's language.

        The simulator is not the policy under training, so scripting it costs
        nothing in RL correctness and removes the failure where the student
        answers the problem outright and the tutor replies as a peer.
        """
        templates = CODE_MIXING_TEMPLATES.get(
            self.persona.language, CODE_MIXING_TEMPLATES["english"]
        )
        topic = problem.skills[0].replace("_", " ") if problem.skills else "this"
        # Vary the opener so a group of G rollouts does not start identically,
        # which would flatten the within-group reward spread GRPO depends on.
        key = random.choice(["confused", "asking"])
        if key == "asking":
            return templates["asking"].format(approach=f"{topic} logic")
        return templates["confused"].format(topic=topic)

    def respond(self, dialogue: Dialogue, problem: Problem) -> str:
        system = self.persona.build_system_prompt(problem)
        messages = [{"role": "system", "content": system}]
        messages.append(
            {"role": "user", "content": f"Problem: {problem.title}\n{problem.description}"}
        )
        for turn in dialogue.turns:
            role = "assistant" if turn.role == "student" else "user"
            messages.append({"role": role, "content": turn.content})
        messages.append({"role": "user", "content": "(Your turn as the student)"})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        complete = hit_terminator(generated, self._terminators)
        self.last_generation_complete = complete
        if not complete:
            # Fall back to the raw generation if trimming leaves nothing.
            text = trim_incomplete(text) or text
        # Safe on the simulator: it is not the policy, so no log-probs are taken
        # over this text.
        return drop_dangling_promise(text) or text

    def _solve_context(self, dialogue: Dialogue) -> list[dict[str, str]]:
        """How the tutoring reaches the solution attempt.

        `SAHAI_SOLVE_CONTEXT` selects the framing:

        * `full` (default, and what every run to date used) replays the whole
          transcript, mapping the student's own turns to `assistant` and the
          tutor's to `user`.
        * `hints` passes only the tutor's turns, as a single user message.
        * `quoted` passes every turn, the student's included, as a single user
          message, so the content is preserved and no turn sits in the model's
          own assistant history.

        Why this is a parameter rather than a fix. The 2026-10-02 benchmark
        measured, on 60 paired problems, that a tutored student solves 0.133
        against 0.283 for the same student with an empty dialogue
        (McNemar p=0.0117). Under `full`, the model about to write code has its
        own hedging in its assistant history: across 1002 student turns in the
        v24 rollouts, 20% express confusion and 28% claim an understanding the
        solve rate does not support. A chat model conditions heavily on its own
        prior assistant turns, so it is being asked to write a solution
        immediately after telling itself it does not understand the problem.

        That is a hypothesis with a mechanism, not a measurement. The
        alternatives -- context length, Hinglish code-mixing, the tutor's hints
        being actively misleading -- are not excluded by anything measured so
        far. `hints` exists so the two can be compared on the same problems
        instead of argued about, and the default is unchanged so existing
        results stay comparable until that comparison is run.
        """
        mode = os.getenv("SAHAI_SOLVE_CONTEXT", "full").lower()

        if mode == "quoted":
            # Every turn, including the student's own, as a single user message.
            #
            # This exists to separate two mechanisms that `hints` removes at the
            # same time. `hints` drops the student's turns, which removes both
            # the content and the fact that content sat in the model's own
            # assistant history. `quoted` keeps every character and removes only
            # the role, so the two can be told apart:
            #
            #   quoted scores like hints  -> the role is the mechanism, and
            #                                `hints` is discarding usable
            #                                context for no reason
            #   quoted scores like full   -> the volume is the mechanism, and
            #                                self-conditioning has nothing to
            #                                do with it
            #
            # The hedging explanation originally offered for the `full` deficit
            # is already ruled out: recovery does not track whether the student
            # said it was lost (Fisher p=0.44, and p=1.00 in the base arm),
            # while it does track how much the student wrote.
            if not dialogue.turns:
                return []
            lines = [
                f"{'You' if t.role == 'student' else 'Tutor'}: {t.content}"
                for t in dialogue.turns
            ]
            return [{
                "role": "user",
                "content": "Here is the tutoring session you just had:\n\n"
                           + "\n\n".join(lines),
            }]

        if mode == "hints":
            hints = [t.content for t in dialogue.turns if t.role == "tutor"]
            if not hints:
                return []
            body = "\n\n".join(f"Hint {i}: {h}" for i, h in enumerate(hints, 1))
            return [{"role": "user", "content": f"Your tutor told you:\n\n{body}"}]

        return [
            {
                "role": "assistant" if t.role == "student" else "user",
                "content": t.content,
            }
            for t in dialogue.turns
        ]

    def attempt_solution(self, dialogue: Dialogue, problem: Problem) -> str:
        """The student's post-tutoring solution attempt — decoded **greedily**.

        This is a measurement, not part of the environment. The dialogue above
        it is sampled, and that sampling is where the tutor's behaviour varies;
        sampling *here* as well only adds the student's dice to the tutor's
        reward.

        It used to sample at temperature 0.3 and `SolveReward` averaged four
        draws. That made `r_sol` a 4-sample Bernoulli estimate: at a true pass
        probability of 0.15 the sampling sd alone is 0.18, larger than any
        plausible tutor-induced difference. Measured over the rollouts of three
        runs, mean within-group variance was 0.0098 for `r_sol` against 0.0482
        for the deterministic pedagogy term — so GRPO's advantage, which is a
        z-score of exactly that within-group spread, was mostly reading noise
        on the one term the project cares about. See docs/04-findings.md #14.

        Greedy makes `r_sol` a deterministic function of the dialogue: all of
        its within-group variance now comes from what the tutor said. It is
        also ~4x cheaper, because one 512-token generation replaces four.
        """
        # Three defects, found by the 2026-10-02 benchmark and fixed here.
        #
        # 1. The prompt asserted tutoring had happened even when the dialogue
        #    was empty: "you just received tutoring", "based on the hints you
        #    received". The unaided arm of a benchmark is exactly the case with
        #    no hints, so the control was being told to use hints that did not
        #    exist. That disadvantaged the control, which makes the measured
        #    result conservative rather than inflated, but it is still wrong.
        # 2. Only `problem.title` reached the prompt, and title is
        #    `row["text"][:80]` — the description truncated at 80 characters,
        #    frequently mid-clause. The student was solving from a cut-off
        #    statement while the tutor saw the whole thing.
        # 3. See `_solve_context` below for the dialogue-framing defect, which
        #    is the one that plausibly explains the benchmark result.
        tutored = bool(dialogue.turns)
        opening = (
            "You are a student who has just been tutored on this problem."
            if tutored
            else "You are a student solving this problem on your own."
        )
        guidance = (
            "Use the hints you were given, and write a Python solution."
            if tutored
            else "Write a Python solution."
        )
        system = (
            f"{opening}\n"
            f"Your ability level is {self.persona.ability_level}/5.\n"
            f"Problem: {problem.description}\n"
            f"{guidance}\n"
            f"Only output the function implementation, nothing else.\n"
            f"Signature: {problem.function_signature}"
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(self._solve_context(dialogue))
        messages.append({"role": "user", "content": "Now write your solution:"})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        code = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return self._extract_code(code, problem.function_name)

    def _extract_code(self, text: str, function_name: str) -> str:
        """Pull the student's code out of its reply, keeping it when the name
        differs.

        This used to end with a literal string check:

            if f"def {function_name}" not in text:
                text = f"def {function_name}():\n    pass"

        which discards everything the student wrote the moment the name does
        not match exactly. Measured on the oracle arm, where a correct solution
        is in front of the student, that fired on 5 of 60 problems and every
        one was naming style rather than a refusal to answer -- MBPP uses
        mixedCase (`max_Prime_Factors`, `decimal_To_Binary`) and the student
        writes snake_case. Three of the five were otherwise correct and scored
        zero, which is 5 points of solve rate lost in every arm of every run.

        So the fallback is now an alias rather than a stub. An alias and not a
        rename, because a recursive solution calls itself by the name it
        defined, and renaming the `def` would break the recursion it is there
        to preserve.
        """
        body = text
        if "```python" in body:
            body = body.split("```python")[1].split("```")[0]
        elif "```" in body:
            parts = body.split("```")
            if len(parts) > 1:
                body = parts[1]
        body = body.strip()
        if not body:
            return f"def {function_name}():\n    pass"

        tree = _parse_salvaging_tail(body)
        if tree is None:
            # Nothing parseable. The stub is still the right answer here: it
            # fails the tests, which is what unparseable output deserves.
            return f"def {function_name}():\n    pass"
        body = tree[1]

        funcs = [
            n.name for n in tree[0].body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not funcs:
            return f"def {function_name}():\n    pass"
        if function_name in funcs:
            return body
        # Last, not first: the reference convention in this dataset is helpers
        # first and the answer last, and the student imitates what it was shown.
        return f"{body}\n\n{function_name} = {funcs[-1]}\n"

