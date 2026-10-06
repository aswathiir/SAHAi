"""The student's code survives a name the tests do not use.

`_extract_code` ended with a literal string check and replaced the entire reply
with `def {name}(): pass` whenever `def {function_name}` was absent. Measured
on the oracle arm -- where a correct solution sits in front of the student --
that fired on 5 of 60 problems and all five were naming style, not refusal:
MBPP uses mixedCase (`max_Prime_Factors`, `count_Substrings`) and the student
writes snake_case. Three were otherwise correct and scored zero, which is 5
points of solve rate lost in every arm of every run recorded.

tutor.py and student.py import torch, which the test venv does not have, so
the method is exercised unbound.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def extract(monkeypatch):
    fake = types.ModuleType("torch")
    fake.no_grad = lambda: _null()
    fake.Tensor = object
    nn = types.ModuleType("torch.nn")
    fnn = types.ModuleType("torch.nn.functional")
    nn.functional = fnn
    fake.nn = nn
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fnn)
    gen = types.ModuleType("sahai.core.generation")
    gen.__getattr__ = lambda name: (lambda *a, **k: a[0] if a else None)
    monkeypatch.setitem(sys.modules, "sahai.core.generation", gen)
    for m in [k for k in sys.modules if k.startswith("sahai.agents.student")]:
        del sys.modules[m]
    from sahai.agents.student import StudentSimulator

    return lambda text, name: StudentSimulator._extract_code(None, text, name)


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


STUB = "def f():\n    pass"


class TestNameMismatch:
    def test_a_case_difference_no_longer_discards_the_solution(self, extract):
        """The measured case: wanted `max_Prime_Factors`, student wrote
        `max_prime_factors`."""
        out = extract("```python\ndef max_prime_factors(n):\n    return n\n```", "max_Prime_Factors")
        assert "def max_prime_factors" in out
        assert "max_Prime_Factors = max_prime_factors" in out

    def test_a_matching_name_is_returned_untouched(self, extract):
        out = extract("```python\ndef f(a):\n    return a\n```", "f")
        assert out.strip() == "def f(a):\n    return a"
        assert "=" not in out

    def test_the_last_function_is_aliased_not_the_first(self, extract):
        """MBPP references put helpers first and the answer last, and the
        student imitates what it was shown."""
        code = "```python\ndef helper(x):\n    return x\ndef answer(x):\n    return x + 1\n```"
        assert extract(code, "wanted").rstrip().endswith("wanted = answer")

    def test_aliasing_preserves_recursion(self, extract):
        """A rename would break a solution that calls itself."""
        code = "```python\ndef fact(n):\n    return 1 if n < 2 else n * fact(n - 1)\n```"
        out = extract(code, "factorial")
        ns: dict = {}
        exec(out, ns)
        assert ns["factorial"](5) == 120


class TestSalvage:
    def test_a_truncated_tail_does_not_lose_the_complete_function(self, extract):
        code = "```python\ndef f(a):\n    return max(a)\n\ndef g(b):\n    if b\n```"
        out = extract(code, "f")
        assert "def f(a)" in out
        ns: dict = {}
        exec(out, ns)
        assert ns["f"]([1, 9, 2]) == 9

    def test_unfenced_code_is_still_read(self, extract):
        out = extract("def f(a):\n    return a\n", "f")
        assert "def f(a)" in out

    def test_unparseable_output_falls_back_to_the_stub(self, extract):
        assert extract("```python\nthis is not python {{{\n```", "f") == STUB

    def test_empty_output_falls_back_to_the_stub(self, extract):
        assert extract("", "f") == STUB
        assert extract("```python\n\n```", "f") == STUB

    def test_prose_with_no_function_falls_back_to_the_stub(self, extract):
        assert extract("I think you should use a loop here.", "f") == STUB


class TestTheStubStillFails:
    def test_the_fallback_is_a_failing_function_not_a_passing_one(self, extract):
        """Unparseable output deserves to score zero; the stub is how."""
        out = extract("{{{", "f")
        ns: dict = {}
        exec(out, ns)
        assert ns["f"]() is None
