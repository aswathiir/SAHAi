from sahai.core.generation import (
    drop_dangling_promise,
    hit_terminator,
    strip_code,
    trim_incomplete,
)


def test_trims_mid_sentence_fragment():
    """The exact failure from the training log: a turn cut off mid-clause got
    appended to the dialogue, and the next model completed it instead of
    replying."""
    text = (
        "3. **Character Counting**: We count occurrences of each character within the"
    )
    assert trim_incomplete(text) == ""


def test_keeps_complete_sentences_and_drops_the_fragment():
    text = "Good start. What structure gives O(1) lookup? Now consider what happens when"
    assert trim_incomplete(text) == "Good start. What structure gives O(1) lookup?"


def test_complete_text_is_unchanged():
    text = "What data structure lets you check if you have seen a character before?"
    assert trim_incomplete(text) == text


def test_trim_drops_unclosed_fence():
    text = "Try this:\n```python\nseen = set()\nfor c in s:"
    assert "```" not in trim_incomplete(text)


def test_hit_terminator():
    assert hit_terminator([1, 2, 151645], {151645}) is True
    assert hit_terminator([1, 2, 3], {151645}) is False
    # No known terminator ids — assume the generation ended cleanly.
    assert hit_terminator([1, 2, 3], set()) is True


def test_strip_code_handles_unclosed_fence():
    """The old paired-fence regex required a closing ``` and so left the entire
    block intact whenever generation was truncated mid-code."""
    text = "Here's the fix:\n```python\nseen_chars = set()\nfor char in s:\n    seen_chars.add(char"
    stripped = strip_code(text)
    assert "```" not in stripped
    assert "seen_chars" not in stripped


def test_strip_code_handles_closed_fence():
    text = "Try:\n```python\ndef f(x):\n    return x\n```\nWhat do you think?"
    stripped = strip_code(text)
    assert "def f" not in stripped
    assert "What do you think?" in stripped


def test_strip_code_removes_bare_assignments():
    text = "Consider this.\nseen = set()\ncounts.append(x)\nWhat does that give you?"
    stripped = strip_code(text)
    assert "seen = set()" not in stripped
    assert "counts.append" not in stripped
    assert "What does that give you?" in stripped


def test_drop_dangling_promise_cuts_trailing_colon():
    """Real student turn from the 2026-07-28 run; the tutor completed it."""
    text = "The output should look something like this:"
    assert drop_dangling_promise(text) == ""


def test_drop_dangling_promise_keeps_prior_sentence():
    text = "I am stuck on this. Here is how we can achieve this:"
    assert drop_dangling_promise(text) == "I am stuck on this."


def test_drop_dangling_promise_loops_past_nested_colons():
    """One pass can land on another colon-terminated clause."""
    text = "Step one.\nHere is the plan:\nAnd the code:"
    assert not drop_dangling_promise(text).endswith(":")


def test_drop_dangling_promise_leaves_clean_text():
    text = "What data structure gives you O(1) lookup?"
    assert drop_dangling_promise(text) == text


def test_drop_dangling_promise_terminates():
    """A pathological all-colon input must not spin."""
    assert drop_dangling_promise(":" * 50) == ""
