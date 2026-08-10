from sahai.core.data import Problem, TestCase
from sahai.reward.solve import CodeVerifier


def _make_problem(function_name, solution, test_cases):
    return Problem(
        id="test",
        title="Test",
        description="Test problem",
        difficulty=1,
        skills=["test"],
        function_name=function_name,
        function_signature=f"def {function_name}():",
        test_cases=test_cases,
        solution=solution,
    )


def test_correct_solution():
    verifier = CodeVerifier(timeout=5)
    problem = _make_problem(
        "two_sum",
        "",
        [TestCase(input={"nums": [2, 7, 11, 15], "target": 9}, expected=[0, 1])],
    )
    code = (
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i"
    )
    score = verifier.verify(code, problem)
    assert score == 1.0


def test_wrong_solution():
    verifier = CodeVerifier(timeout=5)
    problem = _make_problem(
        "two_sum",
        "",
        [TestCase(input={"nums": [2, 7, 11, 15], "target": 9}, expected=[0, 1])],
    )
    code = "def two_sum(nums, target):\n    return [0, 0]"
    score = verifier.verify(code, problem)
    assert score == 0.0


def test_timeout():
    verifier = CodeVerifier(timeout=2)
    problem = _make_problem(
        "slow",
        "",
        [TestCase(input={"x": 1}, expected=1)],
    )
    code = "def slow(x):\n    import time\n    time.sleep(10)\n    return x"
    result = verifier.execute(code, problem.test_cases[0], "slow")
    assert not result.passed
    assert result.timed_out


def test_syntax_error():
    verifier = CodeVerifier(timeout=5)
    problem = _make_problem(
        "broken",
        "",
        [TestCase(input={"x": 1}, expected=1)],
    )
    code = "def broken(x)\n    return x"
    result = verifier.execute(code, problem.test_cases[0], "broken")
    assert not result.passed
    assert result.error is not None


def test_correct_solution_with_example_usage_print_still_passes():
    """Regression: found live on 2026-08-10 in an actual tutor transcript.

    Models routinely append "# Example usage\\nprint(...)" after a function.
    That extra print used to write to stdout before our own result line, so
    json.loads(proc.stdout) failed to parse and the verifier silently
    reported passed=False with no error. This deflated r_sol for every run to
    date wherever a student's attempted solution happened to include a demo
    print, and it also broke execution-verified leakage detection the same way.
    """
    verifier = CodeVerifier(timeout=5)
    problem = _make_problem(
        "find_first_repeated_char",
        solution="unused",
        test_cases=[
            TestCase(input={"s": "abcabc"}, expected="a"),
            TestCase(input={"s": "abcb"}, expected="b"),
        ],
    )
    code = (
        "def find_first_repeated_char(s):\n"
        "    seen = {}\n"
        "    for char in s:\n"
        "        if char in seen:\n"
        "            return char\n"
        "        seen[char] = True\n"
        "    return None\n"
        "\n"
        "# Example usage:\n"
        "input_string = 'hello'\n"
        "print(find_first_repeated_char(input_string))  # prints 'l'\n"
    )
    assert verifier.verify(code, problem) == 1.0


def test_multiple_prints_before_the_result_do_not_break_parsing():
    verifier = CodeVerifier(timeout=5)
    problem = _make_problem("f", "unused", [TestCase(input={"x": 3}, expected=6)])
    code = (
        "def f(x):\n"
        "    print('debug: computing')\n"
        "    print('debug: step 2')\n"
        "    return x * 2\n"
    )
    assert verifier.verify(code, problem) == 1.0


def test_missing_sentinel_reports_a_real_error_not_a_silent_zero():
    """A candidate that crashes before reaching our result line must not look
    identical to one that simply returned the wrong answer."""
    verifier = CodeVerifier(timeout=5)
    problem = _make_problem("f", "unused", [TestCase(input={"x": 1}, expected=1)])
    result = verifier.execute(
        "def f(x):\n    import os\n    os._exit(1)\n",
        problem.test_cases[0],
        "f",
    )
    assert result.passed is False
