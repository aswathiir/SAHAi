"""Leakage scoring.

Written against a real failure: in the 2026-08 run the tutor wrote a complete
working solution and the estimator scored it 0.000, because it used a dict where
the MBPP reference used enumerate/count. It was measuring plagiarism of one
implementation, not leakage.
"""

import pytest
from sahai_core import Dialogue, LeakageEstimator

REFERENCE = (
    'def first_repeated_char(str1):\n'
    '  for index,c in enumerate(str1):\n'
    '    if str1[:index+1].count(c) > 1:\n'
    '      return c\n'
    '  return "None"'
)
PROBLEM_TEXT = "Write a python function to find the first repeated character in a given string."

LEAKED = """Here's how:
```python
def find_first_repeated_char(s):
    seen = {}
    for char in s:
        if char in seen:
            return char
        seen[char] = True
    return None
```"""

CLEAN = "What data structure lets you check whether you have seen a character before?"


def _dialogue(tutor_text):
    d = Dialogue("mbpp-11")
    d.add("student", "I am stuck")
    d.add("tutor", tutor_text)
    return d


@pytest.fixture
def est():
    return LeakageEstimator()


def test_working_solution_is_total_leakage(est):
    """The case the old metric scored 0.000."""
    score = est.estimate(_dialogue(LEAKED), REFERENCE, PROBLEM_TEXT, tutor_code_solves=True)
    assert score == 1.0


def test_code_that_does_not_solve_is_partial(est):
    score = est.estimate(_dialogue(LEAKED), REFERENCE, PROBLEM_TEXT, tutor_code_solves=False)
    assert 0.0 < score < 1.0


def test_clean_question_scores_zero(est):
    """Old metric gave 0.14 here — a floor that made the number meaningless."""
    assert est.estimate(_dialogue(CLEAN), REFERENCE, PROBLEM_TEXT) == 0.0


def test_problem_words_are_not_leakage(est):
    """A tutor must be able to name the thing without being penalised."""
    on_topic = "Think about each character in the string one at a time. What repeats?"
    assert est.estimate(_dialogue(on_topic), REFERENCE, PROBLEM_TEXT) == 0.0


def test_verbatim_paste_still_caught(est):
    d = _dialogue(f"```python\n{REFERENCE}\n```")
    assert est.estimate(d, REFERENCE, PROBLEM_TEXT, tutor_code_solves=True) == 1.0


# --- code extraction and aliasing ------------------------------------------

def test_extracts_only_tutor_code(est):
    d = Dialogue("p")
    d.add("student", "```python\nmy attempt\n```")
    d.add("tutor", "What does that return for an empty input?")
    assert est.extract_tutor_code(d) == []


def test_extracts_unclosed_block(est):
    """A turn truncated at the token cap leaves the fence open."""
    d = _dialogue("Try:\n```python\ndef f(x):\n    return x")
    assert len(est.extract_tutor_code(d)) == 1


def test_alias_added_when_the_name_differs(est):
    """Renaming a function does not make it less of a leak."""
    code = "def my_helper(s):\n    return s"
    variants = est.runnable_variants(code, "expected_name")
    assert len(variants) == 1
    assert "expected_name = my_helper" in variants[0]


def test_no_alias_when_the_name_matches(est):
    code = "def expected_name(s):\n    return s"
    assert est.runnable_variants(code, "expected_name") == [code]


def test_every_defined_function_gets_a_variant(est):
    code = "def helper(x):\n    return x\n\ndef main(x):\n    return helper(x)"
    assert len(est.runnable_variants(code, "wanted")) == 2
