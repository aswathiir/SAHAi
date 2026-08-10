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
