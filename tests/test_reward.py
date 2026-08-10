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
