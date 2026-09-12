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
