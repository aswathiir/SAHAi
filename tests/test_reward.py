from sahai.core.dialogue import Dialogue
from sahai.reward.combined import SAHAIReward
from sahai.reward.leakage import LeakageEstimator
from sahai.settings import RewardSettings


def test_sahai_reward_basic():
    settings = RewardSettings(lambda_ped=1.0, gamma_leak=0.5, hard_penalty=False)
    reward = SAHAIReward(settings)
    result = reward.compute(r_sol=0.8, r_ped=1.0, leakage=0.1, ability_baseline=0.3)
    expected = (0.8 - 0.3) + (1.0 - 1) * 1.0 - 0.5 * 0.1
    assert abs(result.r_sahai - expected) < 1e-6


def test_sahai_reward_hard_penalty():
    settings = RewardSettings(lambda_ped=1.0, gamma_leak=0.5, hard_penalty=True)
    reward = SAHAIReward(settings)
    result = reward.compute(r_sol=0.8, r_ped=0.0, leakage=0.1, ability_baseline=0.3)
    assert result.r_sahai == -1.0


def test_sahai_reward_batch():
    settings = RewardSettings(lambda_ped=1.0, gamma_leak=0.5, hard_penalty=False)
    reward = SAHAIReward(settings)
    results = reward.compute_batch(
        r_sols=[0.5, 0.8, 1.0],
        r_peds=[1.0, 0.0, 1.0],
        leakages=[0.2, 0.1, 0.0],
        ability_baseline=0.3,
    )
    assert len(results) == 3
    assert results[0].r_sol == 0.5
    assert results[2].leakage == 0.0


def test_leakage_token_match():
    estimator = LeakageEstimator()
    dialogue = Dialogue(problem_id="test")
    dialogue.add("tutor", "Use a dictionary to store seen values with their indices")
    dialogue.add("student", "Okay")
    solution = "seen = {}\nfor i, n in enumerate(nums):\n    if target - n in seen:\n        return [seen[target - n], i]\n    seen[n] = i"
    leak = estimator.token_match_leakage(dialogue, solution)
    assert 0.0 <= leak <= 1.0


def test_leakage_ace():
    estimator = LeakageEstimator()
    ace = estimator.ace_leakage(solve_with=0.9, solve_without=0.85)
    assert ace == 1.0

    ace_no_leak = estimator.ace_leakage(solve_with=0.3, solve_without=0.1)
    assert ace_no_leak == 0.0


def test_leakage_is_execution_verified():
    """The 2026-08 failure: a tutor wrote a complete working solution using a
    different algorithm than the reference and scored 0.000 token overlap.
    Execution is what makes leakage detectable regardless of how it is written.
    """
    estimator = LeakageEstimator()
    dialogue = Dialogue(problem_id="p")
    dialogue.add("student", "I am stuck")
    dialogue.add("tutor", "```python\ndef solve(s):\n    return s[0]\n```")

    reference = "def other_name(x):\n    return x[0]"
    problem_text = "Return the first character"

    assert estimator.estimate(dialogue, reference, problem_text,
                              tutor_code_solves=True) == 1.0
    # Same turn, code that does not pass: still penalised, but not maximally.
    assert 0.0 < estimator.estimate(dialogue, reference, problem_text) < 1.0


def test_clean_tutoring_has_no_leakage_floor():
    """Previously any on-topic word overlapped the solution and scored > 0."""
    estimator = LeakageEstimator()
    dialogue = Dialogue(problem_id="p")
    dialogue.add("student", "stuck")
    dialogue.add("tutor", "What happens to the first character when the string is empty?")
    assert estimator.estimate(
        dialogue,
        "def other_name(x):\n    return x[0]",
        problem_text="Return the first character of a string",
    ) == 0.0


def test_evaluator_scores_leakage_the_same_way_as_training():
    """Eval must pass problem_text and tutor_code_solves, like the trainer does.

    `Evaluator.evaluate_model` called `estimate(dialogue, solution)` and let both
    keyword arguments fall back to their defaults, so held-out leakage was scored
    by a different rule than training leakage in every run to date:
      - problem_text="" removed the problem-statement subtraction, inflating
        leak by 34% relative (0.1495 -> 0.2006, measured over 304 v14 dialogues)
      - tutor_code_solves=False meant a complete working solution written by the
        tutor fell back to token overlap instead of scoring 1.0
    The two errors pointed opposite ways and partly cancelled, which is why the
    headline number never looked obviously wrong.

    This test pins the call itself rather than a score, because the bug was an
    omitted argument, not a bad formula.
    """
    import inspect

    from sahai.eval import benchmark

    src = inspect.getsource(benchmark.Evaluator.evaluate_model)
    assert "problem_text=" in src, "eval dropped the problem-statement subtraction"
    assert "tutor_code_solves=" in src, "eval dropped execution-verified leakage"


def test_tutor_code_solves_is_shared_by_trainer_and_evaluator():
    """One implementation, imported by both — duplication has cost this repo
    two bugs already (the same fix had to be applied twice).

    grpo.py is checked by source rather than import: it pulls in torch, which
    the test venv deliberately does not have.
    """
    import pathlib

    from sahai.eval.benchmark import tutor_code_solves as eval_fn
    from sahai.reward.solve import tutor_code_solves as canonical

    assert eval_fn is canonical

    grpo_src = pathlib.Path("sahai/training/grpo.py").read_text()
    assert "from sahai.reward.solve import" in grpo_src
    assert "tutor_code_solves" in grpo_src
    # the old private implementation must be gone, not merely shadowed
    assert "for block in self.leakage_estimator.extract_tutor_code" not in grpo_src


def test_rare_token_filter_ignores_common_teaching_vocabulary():
    """Ordinary teaching words must not count as leakage.

    Plain overlap charged the tutor for saying `count`/`sum`/`index` — words in
    14-19 of 198 MBPP solutions — so every extra sentence raised leakage and the
    reward paid the tutor to stay silent (corr(verbosity, leak) = +0.380 over
    the v14 run's 228 code-free dialogues; +0.160 after this filter).
    """
    common = "def f(x):\n    count = 0\n    return count"
    corpus = [common] * 10 + ["def g(y):\n    return y"] * 10
    estimator = LeakageEstimator(solution_corpus=corpus)

    dialogue = Dialogue(problem_id="p")
    dialogue.add("student", "stuck")
    dialogue.add("tutor", "What would a running count give you here?")
    assert estimator.estimate(dialogue, common, problem_text="Count things") == 0.0


def test_rare_token_filter_still_catches_a_distinctive_giveaway():
    """A token unique to one solution is that problem's signature."""
    target = "def first_repeated_char(s):\n    return s[0]"
    corpus = [target] + ["def g(y):\n    return y"] * 20
    estimator = LeakageEstimator(solution_corpus=corpus)

    dialogue = Dialogue(problem_id="p")
    dialogue.add("student", "stuck")
    dialogue.add("tutor", "Just write first_repeated_char and return the answer.")
    assert estimator.estimate(dialogue, target, problem_text="Find a repeat") > 0.0


def test_unfitted_estimator_does_not_silently_filter():
    """With no corpus there is no basis for calling a token rare, so the
    estimator must fall back to the old behaviour rather than quietly
    filtering against an empty frequency table (which would score everything
    as a leak)."""
    estimator = LeakageEstimator()
    assert not estimator.is_fitted
    dialogue = Dialogue(problem_id="p")
    dialogue.add("student", "stuck")
    dialogue.add("tutor", "Think about using a running count.")
    scored = estimator.estimate(
        dialogue, "def f(x):\n    count = 0\n    return count", problem_text=""
    )
    assert 0.0 < scored <= 1.0


# --- r_sol: partial credit, and the strict reading kept alongside it ---


class _FixedStudent:
    """Returns one canned attempt, so `verify` decides the score."""

    def __init__(self, code):
        self.code = code
        self.calls = 0

    def attempt_solution(self, dialogue, problem):
        self.calls += 1
        return self.code


def _three_test_problem():
    from sahai.core.data import Problem, TestCase

    return Problem(
        id="p", title="Double", description="Return 2n.", difficulty=1,
        skills=["math"], function_name="d", function_signature="def d(n):",
        test_cases=[
            TestCase(input={"n": 1}, expected=2),
            TestCase(input={"n": 2}, expected=4),
            TestCase(input={"n": 3}, expected=6),
        ],
        solution="def d(n):\n    return 2 * n",
    )


def test_partial_credit_is_not_thrown_away():
    """An attempt passing 2 of 3 tests used to score the same zero as one that
    did not parse. The verifier already returned the fraction; `SolveReward`
    discarded it with `if score == 1.0`. On the one reward term that carries no
    other signal, that was the largest avoidable loss of resolution — 22 of 24
    rollouts on disk scored exactly 0.00."""
    from sahai.reward.solve import CodeVerifier, SolveReward

    problem = _three_test_problem()
    # Correct for n=1 and n=2, wrong for n=3.
    student = _FixedStudent("def d(n):\n    return 2 * n if n < 3 else 0")
    reward = SolveReward(CodeVerifier(timeout=5), num_samples=4)

    outcome = reward.compute(student, Dialogue(problem_id="p"), problem)
    assert abs(outcome.score - 2 / 3) < 1e-6, outcome.score
    assert outcome.solved == 0.0, "a partial pass is not a solve"


def test_solved_stays_all_or_nothing_for_comparability():
    from sahai.reward.solve import CodeVerifier, SolveReward

    problem = _three_test_problem()
    student = _FixedStudent("def d(n):\n    return 2 * n")
    reward = SolveReward(CodeVerifier(timeout=5), num_samples=4)

    outcome = reward.compute(student, Dialogue(problem_id="p"), problem)
    assert outcome.score == 1.0
    assert outcome.solved == 1.0


def test_a_greedy_student_is_asked_once_not_four_times():
    """`attempt_solution` decodes greedily, so repeated draws are identical.
    Paying for four of them was the bulk of the ~35 min/epoch reward phase."""
    from sahai.reward.solve import CodeVerifier, SolveReward

    student = _FixedStudent("def d(n):\n    return 2 * n")
    reward = SolveReward(CodeVerifier(timeout=5), num_samples=4)
    reward.compute(student, Dialogue(problem_id="p"), _three_test_problem())
    assert student.calls == 1, f"generated {student.calls} times for one dialogue"


def test_outcome_behaves_as_its_own_score():
    """Load-bearing: the Kaggle kernel is a separate upload from the package
    and formats this value with `:.2f` in its smoke-check cell. A plain
    dataclass would kill the run before training starts."""
    from sahai.reward.solve import SolveOutcome

    outcome = SolveOutcome(0.6, 0.0)
    assert f"{outcome:.2f}" == "0.60"
    assert outcome + 0.4 == 1.0
    assert outcome.score == 0.6 and outcome.solved == 0.0
