"""The tutor must answer *this* turn, not just this learner.

Each detector here exists because of a specific failure visible in the
training run's own transcripts — see `sahai_core.turn_signals` for which.
The false-positive cases matter more than the true-positive ones: a missing
signal costs a generic reply, a wrong signal misdirects the whole turn.
"""

from sahai_core import read_turn, system_prompt, turn_guidance
from sahai_core.turn_signals import TurnSignals


# --- stuck ---


def test_explicit_surrender_is_read_as_stuck():
    for message in (
        "I don't understand how any of this works",
        "no idea where to even begin",
        "Mujhe samajh nahi aa raha hai stacks kaise kaam karta hai",
        "kaise karu ye?",
    ):
        assert read_turn(message).stuck, message


def test_struggle_in_the_past_tense_is_not_stuck():
    """"I was stuck but then I saw it" is progress being reported, and telling
    the tutor to scaffold from scratch there wastes the turn."""
    for message in (
        "I was stuck but then I saw it — the base case was the issue",
        "The confusing bit was why we need two pointers, but I get it now",
        "that part is no longer confusing",
    ):
        assert not read_turn(message).stuck, message


# --- asking for the answer ---


def test_asking_for_the_solution_is_caught():
    for message in (
        "could you please provide Python code to implement this solution?",
        "just tell me the answer",
        "write the function for me",
        "code do na yaar",
    ):
        assert read_turn(message).asking_for_answer, message


def test_ordinary_requests_for_help_are_not_answer_requests():
    """These are what tutoring *is*. Refusing them would be a worse failure
    than the one the detector exists to prevent."""
    for message in (
        "Can you explain that part again?",
        "Tell me more about how heaps order things",
        "can you help me think about the edge cases?",
        "what should I be looking at next?",
    ):
        assert not read_turn(message).asking_for_answer, message


# --- posted code ---


def test_a_pasted_attempt_is_recognised():
    assert read_turn("def f(x):\n    return x * 2\nbut the third test fails").posted_code
    assert read_turn("```python\nfor i in range(n):\n    pass\n```").posted_code


def test_prose_about_code_is_not_code():
    """Case matters: `Return the largest` is English, `return x` is Python.
    Matching lowercase only is what keeps this apart."""
    for message in (
        "I need to return the largest value, right?",
        "Should I use a set here?",
        "My plan is to loop over the list and keep a running total",
    ):
        assert not read_turn(message).posted_code, message


# --- language ---


def test_hinglish_is_detected_in_both_scripts():
    assert read_turn("Kya hum strings logic use kar sakte hain yahan pe?").hinglish
    assert read_turn("मुझे समझ नहीं आ रहा").hinglish


def test_one_stray_marker_is_not_enough():
    """`hai` and `kar` turn up in English text; two markers in one short
    message effectively do not."""
    assert not read_turn("The value of hai is unclear to me").hinglish


# --- guidance ---


def test_guidance_is_empty_when_nothing_is_certain():
    """An approach being proposed is the most useful thing a tutor can respond
    to and is deliberately not detected — no cheap check separates proposing
    from guessing. Silence is the designed outcome, not a gap."""
    signals = read_turn("I think a hash map would let me check in O(1). Is that right?")
    assert not signals
    assert turn_guidance(signals) == ""


def test_guidance_names_moves_not_observations():
    guidance = turn_guidance(read_turn("just tell me the answer"))
    assert "Do not" in guidance
    assert "ask what they have tried" in guidance


def test_refusal_leads_when_several_signals_fire():
    """Handing over a solution is the one mistake that cannot be walked back
    inside a session, so it is stated first."""
    guidance = turn_guidance(
        TurnSignals(stuck=True, asking_for_answer=True, posted_code=True, hinglish=True)
    )
    lines = [line for line in guidance.splitlines() if line.startswith("-")]
    assert len(lines) == 4
    assert "hand over the answer" in lines[0]


# --- placement in the prompt ---


def test_turn_guidance_sits_last_and_rules_stay_first():
    prompt = system_prompt(
        "Return the sum of a list.",
        "WHAT THIS LEARNER ALREADY KNOWS:\n- New to arrays.",
        "THIS TURN:\n- They have written code.",
    )
    assert prompt.index("NEVER write code") < prompt.index("WHAT THIS LEARNER")
    assert prompt.index("WHAT THIS LEARNER") < prompt.index("THIS TURN")
    assert prompt.rstrip().endswith("They have written code.")


def test_an_empty_turn_changes_nothing():
    base = system_prompt("Return the sum of a list.")
    assert system_prompt("Return the sum of a list.", "", "") == base
    assert turn_guidance(read_turn("")) == ""
    assert turn_guidance(read_turn("   ")) == ""
