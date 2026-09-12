"""Tutor inference backends.

Three implementations behind one interface so the rest of the stack never knows
which is running:

  stub  — deterministic, CPU-only, no weights. Lets the full end-to-end system
          run on a laptop and in CI. Not a mock inside a test: a real backend
          the gateway can actually serve.
  hf    — transformers + a PEFT LoRA adapter. What the trainer produces.
  vllm  — throughput backend for real serving (not yet implemented).

The heavy imports are inside the backends, so `import app.backends` stays cheap
and the stub path never pulls in torch.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Generation:
    text: str
    complete: bool  # False when generation stopped at the token cap


class TutorBackend(ABC):
    @abstractmethod
    def generate(
        self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float
    ) -> Generation: ...

    @property
    @abstractmethod
    def describe(self) -> str: ...


class StubBackend(TutorBackend):
    """Deterministic Socratic prompts, chosen by hashing the conversation.

    Every response is a question and none contain code, so a stack running on
    the stub exercises the same pedagogy checks a real model would face.
    """

    _PROMPTS = (
        "What data structure would let you check whether you have seen a value before?",
        "Before writing anything, what should happen on an empty input?",
        "Can you walk me through your approach on the smallest example you can think of?",
        "What is the cost of the step you just described, and can it be cheaper?",
        "Which part of the problem statement are you least sure about?",
        "What would you expect the answer to be for a two-element input?",
    )

    def generate(
        self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float
    ) -> Generation:
        seed = "".join(m.get("content", "") for m in messages[-3:])
        idx = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(self._PROMPTS)
        return Generation(text=self._PROMPTS[idx], complete=True)

    @property
    def describe(self) -> str:
        return "stub"


# Wall-clock cap on one generation, in seconds. Must stay below the session
# service's HTTP budget (240s) or the caller times out first and discards a
# reply the model did produce. Env-tunable so a GPU deployment, where this is
# never the binding constraint, can raise it without a rebuild.
GENERATE_MAX_SECONDS = float(os.getenv("SAHAI_GENERATE_MAX_SECONDS", "200"))


class HFBackend(TutorBackend):
    """transformers + optional LoRA adapter produced by jobs/trainer."""

    def __init__(self, base_model: str, adapter_path: str | None, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from sahai_core.generation import terminator_ids

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map=device
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        self.model = model
        self._adapter = adapter_path
        self._base = base_model
        self._terminators = terminator_ids(self.tokenizer, model)

    def generate(
        self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float
    ) -> Generation:
        from sahai_core.generation import hit_terminator, trim_incomplete

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                # Stop before the caller gives up. On CPU (~1.8 tok/s measured)
                # 192 tokens can outrun session's 240s budget, and generation
                # that finishes after the client has disconnected is work done
                # for nobody: the learner saw a 500 and the tokens are dropped.
                # `max_time` ends generation cleanly at the cap, so a partial
                # reply still goes through trim_incomplete below and comes back
                # as an honest `complete=False` instead of nothing at all.
                max_time=GENERATE_MAX_SECONDS,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        complete = hit_terminator(generated, self._terminators)
        if not complete:
            decoded = trim_incomplete(decoded) or decoded
        return Generation(text=decoded, complete=complete)

    @property
    def describe(self) -> str:
        return f"hf:{self._base}" + (f"+{self._adapter}" if self._adapter else "")


def build_backend() -> TutorBackend:
    kind = os.getenv("SAHAI_TUTOR_BACKEND", "stub").lower()
    if kind == "stub":
        return StubBackend()
    if kind == "hf":
        return HFBackend(
            base_model=os.getenv("SAHAI_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            adapter_path=os.getenv("SAHAI_ADAPTER_PATH") or None,
            device=os.getenv("SAHAI_DEVICE", "cuda"),
        )
    if kind == "vllm":
        raise NotImplementedError("vllm backend not implemented yet")
    raise ValueError(f"unknown SAHAI_TUTOR_BACKEND: {kind!r}")
