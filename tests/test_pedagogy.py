from sahai.core.dialogue import Dialogue
from sahai.reward.pedagogy import RuleBasedJudge


def _make_dialogue(tutor_msgs, student_msgs):
    d = Dialogue(problem_id="test")
    for s, t in zip(student_msgs, tutor_msgs):
        d.add("student", s)
        d.add("tutor", t)
    return d


def test_good_dialogue_accepted():
    judge = RuleBasedJudge()
    d = _make_dialogue(
        tutor_msgs=[
            "What data structure could help you look up values quickly?",
            "Good thinking. What would you store as the key?",
            "Exactly. Now what would you check for each element?",
        ],
        student_msgs=[
            "I don't know how to start.",
            "Maybe a dictionary?",
            "The number as key and index as value?",
        ],
    )
    assert judge.evaluate(d) == 1.0


def test_code_block_penalized():
    judge = RuleBasedJudge()
    d = _make_dialogue(
        tutor_msgs=[
            "Here's the solution:\n```python\ndef solve(x):\n    return x\n```",
        ],
        student_msgs=["Help me."],
    )
    # Fails code-block, solution-pattern and asks-question of 5 checks.
    assert judge.evaluate(d) == 2 / 5


def test_unclosed_code_block_penalized():
    """A turn truncated at max_new_tokens leaves the fence open."""
    judge = RuleBasedJudge()
    d = _make_dialogue(
        tutor_msgs=[
            "Here's the solution:\n```python\nseen = set()\nfor c in s:",
        ],
        student_msgs=["Help me."],
    )
    assert judge.evaluate(d) < 1.0


def test_solution_pattern_penalized():
    judge = RuleBasedJudge()
    d = _make_dialogue(
        tutor_msgs=[
            "You should write: def two_sum(nums, target): return [0,1]",
        ],
        student_msgs=["How do I solve this?"],
    )
    assert judge.evaluate(d) == 3 / 5


def test_no_questions_penalized():
    judge = RuleBasedJudge()
    d = _make_dialogue(
        tutor_msgs=[
            "Use a hash map.",
            "Store the values.",
            "Then look them up.",
        ],
        student_msgs=[
            "How do I start.",
            "Ok.",
            "Ok.",
        ],
    )
    assert judge.evaluate(d) == 4 / 5


def test_grading_separates_dialogues():
    """Partial credit is what keeps a GRPO group from collapsing to one value."""
    judge = RuleBasedJudge()
    good = _make_dialogue(
        tutor_msgs=["What structure gives O(1) lookup?"],
        student_msgs=["Help me."],
    )
    mediocre = _make_dialogue(
        tutor_msgs=["Use a hash map."],
        student_msgs=["Help me."],
    )
    bad = _make_dialogue(
        tutor_msgs=["```python\ndef f(): return 1\n```"],
        student_msgs=["Help me."],
    )
    assert judge.evaluate(good) > judge.evaluate(mediocre) > judge.evaluate(bad)


def test_dangling_promise_penalized():
    """A tutor turn ending in ':' promises code it never delivers; the student
    then completes the promise, which is how the roles invert."""
    judge = RuleBasedJudge()
    d = _make_dialogue(
        tutor_msgs=["Good thinking. Here is how you would structure it:"],
        student_msgs=["I don't get it."],
    )
    clean = _make_dialogue(
        tutor_msgs=["Good thinking. How would you structure it?"],
        student_msgs=["I don't get it."],
    )
    assert judge.evaluate(d) < judge.evaluate(clean)


def test_dangling_promise_only_checks_tutor():
    """The student is post-processed, so its colons must not cost the tutor."""
    judge = RuleBasedJudge()
    d = _make_dialogue(
        tutor_msgs=["What structure gives O(1) lookup?"],
        student_msgs=["Here is my attempt:"],
    )
    assert judge.evaluate(d) == 1.0


def test_empty_dialogue_scores_zero():
    """Found by hand-driving the live API: submitting with no conversation at
    all returned 0.8, because every "no X" check passes vacuously when there is
    nothing to inspect. In training that rewards a silent tutor."""
    judge = RuleBasedJudge()
    assert judge.evaluate(Dialogue(problem_id="p")) == 0.0


def test_student_only_dialogue_scores_zero():
    """Student turns alone are not teaching."""
    judge = RuleBasedJudge()
    d = Dialogue(problem_id="p")
    d.add("student", "I am stuck")
    d.add("student", "still stuck")
    assert judge.evaluate(d) == 0.0


def test_one_good_tutor_turn_still_scores_well():
    """The zero-guard must not punish short but real dialogues."""
    judge = RuleBasedJudge()
    d = Dialogue(problem_id="p")
    d.add("student", "I am stuck")
    d.add("tutor", "What structure gives you O(1) lookup?")
    assert judge.evaluate(d) == 1.0


def _uniform_dialogue(n_turns, tutor_msg):
    d = Dialogue(problem_id="p")
    for _ in range(n_turns):
        d.add("student", "I am stuck")
        d.add("tutor", tutor_msg)
    return d


def test_score_is_length_neutral():
    """Equally good dialogues must score the same regardless of length.

    The checks used to fail the whole dialogue if any single tutor turn
    violated them, so the score slid down purely with turn count — measured
    across the v11 run, mean r_ped went 0.720 / 0.639 / 0.537 / 0.453 for
    1 / 2 / 3 / 4 tutor turns. GRPO then optimises for ending the conversation
    early, which is an artifact of the scoring rather than better teaching.
    """
    judge = RuleBasedJudge()
    good = "What structure gives you O(1) lookup?"
    scores = {n: judge.evaluate(_uniform_dialogue(n, good)) for n in (1, 2, 4, 8)}
    assert len(set(scores.values())) == 1, f"length-dependent scores: {scores}"

    bad = "```python\ndef f(): return 1\n```"
    bad_scores = {n: judge.evaluate(_uniform_dialogue(n, bad)) for n in (1, 2, 4, 8)}
    assert len(set(bad_scores.values())) == 1, f"length-dependent scores: {bad_scores}"


def test_violations_scored_per_turn_not_all_or_nothing():
    """One bad turn among several good ones must cost partial credit, not all.

    Under all-or-nothing scoring a single slip erased the whole check, which
    both overstated the penalty and flattened the gradient between rollouts
    that differ by one turn.
    """
    judge = RuleBasedJudge()
    d = Dialogue(problem_id="p")
    for _ in range(3):
        d.add("student", "I am stuck")
        d.add("tutor", "What structure gives you O(1) lookup?")
    d.add("student", "still stuck")
    d.add("tutor", "```python\ndef f(): return 1\n```")

    score = judge.evaluate(d)
    all_good = judge.evaluate(_uniform_dialogue(4, "What gives O(1) lookup?"))
    all_bad = judge.evaluate(_uniform_dialogue(4, "```python\ndef f(): return 1\n```"))
    assert all_bad < score < all_good


def test_threshold_does_not_pay_to_question_every_turn():
    """The regression that cost a run, pinned at its actual mechanism.

    The check was briefly the *fraction* of tutor turns containing a question,
    to give it a gradient. Over the run that followed, mean r_ped climbed
    0.717 -> 0.992 while held-out solve fell 16.3% -> 6.2%, the worst since the
    original baseline.

    The exploit is not that a short question scores well — a single
    contentless question reaches r_ped 1.0 under *both* forms, because the
    other four checks all pass vacuously on a short clean turn. The difference
    is the pressure. In a four-turn dialogue:

        questions   threshold   fraction
            2          1.00       0.90
            4          1.00       1.00

    Under the threshold, converting the two remaining substantive turns into
    questions gains nothing. Under the fraction it is worth 0.10 of r_ped, and
    since reward carries `(r_ped - 1) * lambda`, closing that gap also cancels
    the pedagogy penalty outright. Every teaching turn left un-questioned was
    costing reward, so they stopped being teaching turns.

    This does not make the threshold good — it is inert, which is why it was
    changed. It makes it non-exploitable, which the fraction was not.
    """
    judge = RuleBasedJudge()

    def with_questions(n, total=4):
        d = Dialogue(problem_id="p")
        for i in range(total):
            d.add("student", "stuck")
            d.add(
                "tutor",
                "What structure helps here?" if i < n
                else "A hash map gives O(1) lookup on the key.",
            )
        return judge.evaluate(d)

    assert with_questions(2) == with_questions(4), (
        "questioning every turn must gain nothing over clearing the bar, or "
        "substantive turns are worth converting into question marks"
    )
    # And the check still has to do something: no questions must score lower.
    assert with_questions(0) < with_questions(2)
