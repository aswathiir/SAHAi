from sahai.core.dataset import _parse_assert, _tag_skills, _extract_function_name


def test_parse_assert_simple():
    result = _parse_assert("assert add(1, 2) == 3")
    assert result is not None
    func, inputs, expected = result
    assert func == "add"
    assert expected == 3


def test_parse_assert_list():
    result = _parse_assert("assert two_sum([2, 7], 9) == [0, 1]")
    assert result is not None
    func, inputs, expected = result
    assert func == "two_sum"
    assert expected == [0, 1]


def test_parse_assert_string():
    result = _parse_assert('assert reverse("hello") == "olleh"')
    assert result is not None
    _, _, expected = result
    assert expected == "olleh"


def test_parse_assert_invalid():
    assert _parse_assert("not an assert") is None
    assert _parse_assert("assert True") is None


def test_tag_skills():
    skills = _tag_skills("Write a function to sort an array")
    assert "sorting" in skills or "arrays" in skills

    skills = _tag_skills("Find the longest common subsequence")
    assert "dynamic_programming" in skills


def test_tag_skills_fallback():
    skills = _tag_skills("Do something unusual")
    assert skills == ["general"]


def test_extract_function_name():
    assert _extract_function_name("def foo(x):\n    return x") == "foo"
    assert _extract_function_name("def bar(a, b):\n    pass") == "bar"
    assert _extract_function_name("no function here") == "solution"


def test_eval_set_is_large_enough_to_resolve_a_run():
    """20 held-out problems could not distinguish the runs it was judging.

    Bootstrapping a 20-problem mean: a run whose true solve rate is 0.16
    reports anywhere in [0.00, 0.35], sd 0.082. The entire spread across six
    runs — 6.2% to 16.3% — is about 1.3 standard deviations, so every
    comparison rested on roughly two problems. This pins the size so it cannot
    drift back down without someone deciding to.
    """
    from sahai.settings import Settings

    assert Settings().eval_problems >= 60
    assert Settings.kaggle().eval_problems >= 60


def test_eval_budget_still_fits_the_kaggle_cap():
    """At the measured 89 s/problem, and 6.0 h of training for 10 epochs, the
    whole run has to stay inside Kaggle's 12 h session limit."""
    from sahai.settings import Settings

    s = Settings.kaggle()
    training_hours = 6.0                     # measured, 10 epochs
    eval_hours = s.eval_problems * 89 / 3600
    assert training_hours + eval_hours < 11.0, (
        f"training {training_hours:.1f}h + eval {eval_hours:.1f}h leaves no margin "
        "under the 12h cap"
    )


def _bank_with_mastery(mastery_by_skill, difficulty=1):
    """A bank plus a tracer whose mastery we control, to drive ZPD selection."""
    from sahai.agents.tracer import BKTTracer
    from sahai.core.data import Problem, ProblemBank, TestCase

    problems = [
        Problem(
            id=f"p{i}", title=f"t{i}", description="d", difficulty=difficulty,
            skills=[skill], function_name="f", function_signature="def f():",
            test_cases=[TestCase(input={}, expected=1)], solution="pass",
        )
        for i, skill in enumerate(mastery_by_skill)
    ]
    tracer = BKTTracer()
    for skill, value in mastery_by_skill.items():
        tracer.skills[skill] = value
    return ProblemBank(problems=problems), tracer


def _graded_bank(ability):
    """A bank spanning difficulty 1..5, and a tracer at a chosen ability."""
    from sahai.agents.tracer import BKTTracer
    from sahai.core.data import Problem, ProblemBank, TestCase

    problems = [
        Problem(
            id=f"d{d}", title=f"d{d}", description="d", difficulty=d,
            skills=[f"skill{d}"], function_name="f", function_signature="def f():",
            test_cases=[TestCase(input={}, expected=1)], solution="pass",
        )
        for d in range(1, 6)
    ]
    tracer = BKTTracer()
    # One skill per problem, all at `ability`, so get_ability() == ability and
    # nothing is excluded by the mastery ceiling.
    for d in range(1, 6):
        tracer.skills[f"skill{d}"] = ability
    return ProblemBank(problems=problems), tracer


def test_selection_survives_the_mastery_collapse_that_emptied_the_band():
    """The defect that cost the 2026-09-12 run seven of its ten epochs.

    `p_init` was 0.3 and `zpd_low` was 0.3, so the old band meant "skills never
    attempted". BKT then drives a failed skill to a fixed point of ~0.109 in
    three observations with no forgetting transition to lift it back, so once
    every skill had been tried the band was empty for the rest of the run and
    stayed empty. The log read "Only 0 problems in the ZPD band" at epochs 4,
    5, 6, 7 and 9.
    """
    collapsed = {f"s{i}": 0.109 for i in range(8)}
    bank, tracer = _bank_with_mastery(collapsed)

    assert all(m < 0.3 for m in collapsed.values()), (
        "fixture must sit below the old zpd_low or it does not reproduce the bug"
    )
    assert bank.zpd_candidates(tracer), "band empty again at the BKT fixed point"
    assert len(bank.zpd_sample(tracer, 4)) == 4


def test_selection_targets_difficulty_just_above_ability():
    bank, tracer = _graded_bank(0.5)  # target = 1 + 0.5*4 = 3.0
    assert tracer.target_difficulty(5) == 3.0
    assert {p.difficulty for p in bank.zpd_candidates(tracer)} == {2, 3, 4}

    bank, tracer = _graded_bank(0.0)  # target = 1.0
    assert {p.difficulty for p in bank.zpd_candidates(tracer)} == {1, 2}


def test_selection_does_not_freeze_on_one_difficulty_level():
    """Ranking by `abs(difficulty - target)` reintroduced the original bug.

    At an ability putting the target near 2.0, every difficulty-2 problem
    scores 0.0 and every difficulty-1 problem scores 1.0, so a sort draws
    difficulty 2 every epoch and the rest of the bank is never trained on.
    Simulated over the measured bank (1:186 2:59 3:36 4:9 5:6) that produced
    ten identical epochs. Draws are random within the window for this reason.
    """
    bank, tracer = _graded_bank(0.25)  # target = 2.0; window covers 1, 2, 3
    drawn = {p.difficulty for _ in range(40) for p in bank.zpd_sample(tracer, 1)}
    assert drawn == {1, 2, 3}, f"curriculum frozen on {drawn}"


def test_a_failing_skill_is_still_offered_work():
    """The old *floor* excluded exactly the learner who most needs practice.

    Mastery 0.11 means they are failing that skill. The response is an easier
    problem, which difficulty targeting supplies — not exclusion from the
    curriculum, which is what `zpd_low <= mastery` did.
    """
    bank, tracer = _bank_with_mastery({"failing": 0.11}, difficulty=1)
    assert tracer.in_zpd(1, ["failing"], bank.max_difficulty())


def test_a_mastered_skill_is_not_re_offered():
    bank, tracer = _bank_with_mastery({"mastered": 0.98}, difficulty=1)
    assert not tracer.in_zpd(1, ["mastered"], bank.max_difficulty())


def test_a_mastered_skill_is_never_drawn():
    bank, tracer = _bank_with_mastery(
        {"middling": 0.5, "mastered": 0.98, "failing": 0.05}
    )
    drawn = {p.skills[0] for _ in range(40) for p in bank.zpd_sample(tracer, 1)}
    assert "mastered" not in drawn
    assert drawn == {"middling", "failing"}


def test_repeated_epochs_at_one_ability_do_not_redraw_one_fixed_set():
    """gcd-by-recursion was the drawn problem in three of the run's last six
    epochs, because the fallback ordering was deterministic once the band was
    empty. Ties must break randomly or the curriculum freezes."""
    bank, tracer = _bank_with_mastery({f"s{i}": 0.5 for i in range(12)})
    seen = {p.id for _ in range(20) for p in bank.zpd_sample(tracer, 2)}
    assert len(seen) > 2, f"selection is frozen on {seen}"


def test_zpd_sample_still_returns_n_when_everything_fits():
    bank, tracer = _bank_with_mastery({f"s{i}": 0.5 for i in range(8)})
    assert len(bank.zpd_sample(tracer, 4)) == 4
    assert len(bank.zpd_candidates(tracer)) == 8


def test_zpd_sample_cannot_exceed_the_bank():
    bank, tracer = _bank_with_mastery({"only": 0.99})
    assert len(bank.zpd_sample(tracer, 4)) == 1


def test_eval_budget_still_fits_after_the_greedy_solve_change():
    """The reward phase dominates the epoch and is mostly student generation.

    Measured on the 2026-09-12 run: ~20 min rollout + ~35 min rewards + ~2.5
    min update per epoch = 9.1 h for 10 epochs, plus 0.5 h for a 20-problem
    eval. Four sampled solution attempts per rollout were the bulk of that 35
    min; one greedy attempt replaces them. This asserts the arithmetic still
    clears the 12 h cap with the 60-problem eval, so the two changes are not
    quietly in conflict.
    """
    from sahai.settings import Settings

    settings = Settings.kaggle()

    rollout_min, reward_min, update_min = 20.0, 35.0, 2.5
    # Generation is ~4/5 of the reward phase; one greedy draw replaces four.
    reward_min_after = reward_min * (0.2 + 0.8 / 4)
    epoch_h = (rollout_min + reward_min_after + update_min) / 60.0
    training_h = epoch_h * settings.training.epochs
    # 89 s/problem measured, and the eval student also drops to one draw.
    eval_h = settings.eval_problems * 89.0 * 0.4 / 3600.0

    assert training_h + eval_h < 11.0, (
        f"training {training_h:.1f}h + eval {eval_h:.1f}h leaves no margin "
        "under the 12h cap"
    )
    assert training_h < 9.1, "the greedy change should have bought time, not cost it"
