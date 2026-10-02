"""The prompt the student writes its solution from.

Three defects lived here undetected until a benchmark put an untutored arm
next to a tutored one. None of them were visible while every run tutored every
attempt, which is why they survived six runs.

student.py imports torch, which this venv deliberately does not have (see
test_reward.py), so the prompt construction is exercised through a stub rather
than a real model.
"""

from __future__ import annotations

import os
import pathlib
import sys
import types

import pytest


@pytest.fixture
def student_cls(monkeypatch):
    """Import StudentSimulator with torch and the generation helpers stubbed."""
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _null_ctx()
    fake_torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    gen = types.ModuleType("sahai.core.generation")
    gen.drop_dangling_promise = lambda s, *a, **k: s
    gen.hit_terminator = lambda *a, **k: True
    gen.terminator_ids = lambda *a, **k: []
    gen.trim_incomplete = lambda s, *a, **k: s
    monkeypatch.setitem(sys.modules, "sahai.core.generation", gen)

    for mod in [m for m in sys.modules if m.startswith("sahai.agents.student")]:
        del sys.modules[mod]
    from sahai.agents.student import StudentPersona, StudentSimulator

    return StudentSimulator, StudentPersona


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last = messages
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


def build(student_cls, turns):
    StudentSimulator, StudentPersona = student_cls
    s = StudentSimulator.__new__(StudentSimulator)
    s.persona = StudentPersona(ability_level=2, code_mixing_ratio=0.3,
                               language="hinglish", persistence=0.7)
    s.tokenizer = FakeTok()

    from sahai.core.dialogue import Dialogue

    d = Dialogue(problem_id="p1")
    for role, content in turns:
        d.add(role, content)
    return s, d


class Problem:
    title = "Write a python function to count the number of pairs whose sum is equal"
    description = ("Write a python function to count the number of pairs whose sum "
                   "is equal to a given target value, considering each pair once.")
    function_signature = "def count_pairs(arr, target):"


class TestUntutoredPrompt:
    def test_does_not_claim_tutoring_that_did_not_happen(self, student_cls):
        """The control arm was told to use hints it had never been given."""
        s, d = build(student_cls, [])
        msgs = s._solve_context(d)
        assert msgs == [], "an empty dialogue contributes no turns"

    def test_system_prompt_adapts_to_an_empty_dialogue(self, student_cls, monkeypatch):
        src = pathlib.Path("sahai/agents/student.py").read_text()
        assert "solving this problem on your own" in src
        assert "tutored = bool(dialogue.turns)" in src


class TestDescription:
    def test_full_description_not_the_truncated_title(self):
        """`title` is `row["text"][:80]`, so the student was solving from a
        statement cut at 80 characters while the tutor saw all of it."""
        src = pathlib.Path("sahai/agents/student.py").read_text()
        block = src[src.index("def attempt_solution"):src.index("Now write your solution")]
        prompt_lines = [
            l for l in block.splitlines()
            if l.strip().startswith('f"') and "{problem." in l
        ]
        joined = "\n".join(prompt_lines)
        assert "problem.description" in joined, "full description must reach the prompt"
        assert "problem.title" not in joined, (
            "title is row['text'][:80]; using it means solving from a truncated "
            "statement"
        )


class TestSolveContextModes:
    def test_full_is_the_default_so_old_runs_stay_comparable(self, student_cls, monkeypatch):
        monkeypatch.delenv("SAHAI_SOLVE_CONTEXT", raising=False)
        s, d = build(student_cls, [("student", "I don't get it"), ("tutor", "What is a pair?")])
        msgs = s._solve_context(d)
        assert [m["role"] for m in msgs] == ["assistant", "user"]

    def test_full_puts_the_students_confusion_in_its_own_assistant_history(self, student_cls, monkeypatch):
        """The mechanism the benchmark result points at: the model is asked to
        write code immediately after telling itself it does not understand."""
        monkeypatch.delenv("SAHAI_SOLVE_CONTEXT", raising=False)
        s, d = build(student_cls, [("student", "I don't understand this at all")])
        msgs = s._solve_context(d)
        assert msgs[0]["role"] == "assistant"
        assert "don't understand" in msgs[0]["content"]

    def test_hints_mode_drops_the_students_own_turns(self, student_cls, monkeypatch):
        monkeypatch.setenv("SAHAI_SOLVE_CONTEXT", "hints")
        s, d = build(student_cls, [
            ("student", "I don't understand this at all"),
            ("tutor", "Think about what makes two numbers a pair."),
            ("student", "still lost"),
            ("tutor", "Try a nested loop first."),
        ])
        msgs = s._solve_context(d)
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        body = msgs[0]["content"]
        assert "don't understand" not in body and "still lost" not in body
        assert "nested loop" in body and "makes two numbers a pair" in body

    def test_hints_mode_numbers_the_hints(self, student_cls, monkeypatch):
        monkeypatch.setenv("SAHAI_SOLVE_CONTEXT", "hints")
        s, d = build(student_cls, [("tutor", "a"), ("tutor", "b")])
        body = s._solve_context(d)[0]["content"]
        assert "Hint 1: a" in body and "Hint 2: b" in body

    def test_hints_mode_with_no_tutor_turns_is_empty(self, student_cls, monkeypatch):
        monkeypatch.setenv("SAHAI_SOLVE_CONTEXT", "hints")
        s, d = build(student_cls, [("student", "hello")])
        assert s._solve_context(d) == []

    def test_unknown_mode_falls_back_to_full(self, student_cls, monkeypatch):
        monkeypatch.setenv("SAHAI_SOLVE_CONTEXT", "nonsense")
        s, d = build(student_cls, [("student", "x"), ("tutor", "y")])
        assert len(s._solve_context(d)) == 2


class TestBenchmarkKeepsEvidence:
    def test_dialogues_are_dumped(self):
        """The first run produced a significant result it could not explain,
        because only the scores were kept."""
        src = pathlib.Path("sahai/eval/ab_benchmark.py").read_text()
        assert "dialogues: list[dict]" in src
        assert "res.dialogues.append" in src

    def test_solve_context_is_recorded_in_the_output(self):
        src = pathlib.Path("sahai/eval/ab_benchmark.py").read_text()
        assert '"solve_context"' in src
