from __future__ import annotations

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

    def attempt_solution(self, dialogue: Dialogue, problem: Problem) -> str:
        system = (
            f"You are a student who just received tutoring on: {problem.title}.\n"
            f"Your ability level is {self.persona.ability_level}/5.\n"
            f"Based on the hints you received, write a Python solution.\n"
            f"Only output the function implementation, nothing else.\n"
            f"Signature: {problem.function_signature}"
        )
        messages = [{"role": "system", "content": system}]
        for turn in dialogue.turns:
            role = "assistant" if turn.role == "student" else "user"
            messages.append({"role": role, "content": turn.content})
        messages.append({"role": "user", "content": "Now write your solution:"})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        code = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return self._extract_code(code, problem.function_name)

    def _extract_code(self, text: str, function_name: str) -> str:
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        if f"def {function_name}" not in text:
            text = f"def {function_name}():\n    pass"
        return text.strip()
