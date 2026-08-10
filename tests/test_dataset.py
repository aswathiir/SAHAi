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
