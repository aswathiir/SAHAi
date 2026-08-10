from __future__ import annotations

from typing import TYPE_CHECKING

import re

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sahai.core.data import Problem
    from sahai.core.dialogue import Dialogue

from sahai.core.generation import (
    hit_terminator,
    strip_code,
    terminator_ids,
    trim_incomplete,
)

TUTOR_SYSTEM_PROMPT = (
    "You are a tutor. You help students think, NOT give answers.\n\n"
    "STRICT RULES:\n"
    "- NEVER write code. No code blocks. No function definitions. No pseudocode.\n"
    "- NEVER show the solution or any part of it.\n"
    "- Ask ONE question per turn to guide the student's thinking.\n"
    "- Keep responses to 2-3 sentences maximum.\n"
    "- Match the student's language style.\n\n"
    "GOOD example: 'What data structure lets you check if you have seen a character before in O(1)?'\n"
    "BAD example: 'Here is the solution: def func(): ...'"
)


class TutorPolicy:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 192,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self._terminators = terminator_ids(tokenizer, model)
        self.last_generation_complete = True

    def _build_messages(self, dialogue: Dialogue, problem: Problem) -> list[dict[str, str]]:
        system = (
            f"{TUTOR_SYSTEM_PROMPT}\n\n"
            f"Problem: {problem.title}\n{problem.description}\n"
            f"(You know the solution but must NOT reveal it.)"
        )
        messages = [{"role": "system", "content": system}]
        for turn in dialogue.turns:
            role = "assistant" if turn.role == "tutor" else "user"
            messages.append({"role": role, "content": turn.content})
        return messages

    def generate(self, dialogue: Dialogue, problem: Problem) -> str:
        messages = self._build_messages(dialogue, problem)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
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
        # No dangling-promise trim here, unlike the student: this is the policy
        # under training, and editing its text would put tokens it never sampled
        # into the log-prob computation. The pedagogy reward penalizes the
        # behaviour instead.
        return text

    _strip_code = staticmethod(strip_code)

    def compute_log_probs(
        self, dialogue: Dialogue, problem: Problem
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self._build_messages(dialogue, problem)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        encoding = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = encoding["input_ids"]

        tutor_mask = self._build_tutor_mask(messages, input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=encoding["attention_mask"])
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        tutor_mask = tutor_mask[:, 1:]
        masked_log_probs = token_log_probs * tutor_mask

        return masked_log_probs, tutor_mask

    def _build_tutor_mask(
        self, messages: list[dict], input_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.float)
        seq_len = input_ids.shape[1]

        prev_len = 0
        for i in range(1, len(messages) + 1):
            partial = messages[:i]
            text = self.tokenizer.apply_chat_template(partial, tokenize=False)
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
            curr_len = min(tokens.shape[1], seq_len)

            if messages[i - 1]["role"] == "assistant":
                mask[0, prev_len:curr_len] = 1.0

            prev_len = curr_len

        return mask
