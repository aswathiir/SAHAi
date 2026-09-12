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
    # Fails code-block and solution-pattern of the 3 remaining checks.
    assert judge.evaluate(d) == 1 / 3


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
    assert judge.evaluate(d) == 2 / 3


def test_style_is_no_longer_scored():
    """Questions and length were dropped after the question-fraction change
    cost a run (held-out solve 16.3% -> 6.2%, docs/04-findings.md entry 11).

    A turn that teaches plainly and one that asks a question now score the
    same, because the reward has no opinion about which is better teaching —
    and every opinion it did have was gamed within one run. `r_sol` decides,
    by whether the student solves the problem afterwards.
    """
    judge = RuleBasedJudge()
    asks = _make_dialogue(
        tutor_msgs=["What structure gives you O(1) lookup?"], student_msgs=["Help."]
    )
    tells = _make_dialogue(
        tutor_msgs=["A hash map gives O(1) lookup on the key."], student_msgs=["Help."]
    )
    exploit = _make_dialogue(
        tutor_msgs=["Hash maps? Kaise use kar sakta hu?"], student_msgs=["Help."]
    )
    assert judge.evaluate(asks) == judge.evaluate(tells) == judge.evaluate(exploit)


def test_length_is_no_longer_scored():
    """The 200-word cutoff was arbitrary and was the vector for the original
    length bias. A long substantive turn is not a pedagogy failure."""
    judge = RuleBasedJudge()
    long_turn = _make_dialogue(tutor_msgs=["word " * 250], student_msgs=["Help."])
    assert judge.evaluate(long_turn) == 1.0


def test_grading_separates_dialogues():
    """Partial credit is what keeps a GRPO group from collapsing to one value.

    The separation is now between *giving the answer away* and not, rather than
    between teaching styles: asking and telling score identically on purpose.
    """
    judge = RuleBasedJudge()
    clean = _make_dialogue(
        tutor_msgs=["What structure gives O(1) lookup?"], student_msgs=["Help me."]
    )
    partial = _make_dialogue(
        tutor_msgs=["Here is how you would structure it:"], student_msgs=["Help me."]
    )
    gives_it_away = _make_dialogue(
        tutor_msgs=["```python\ndef f(): return 1\n```"], student_msgs=["Help me."]
    )
    assert judge.evaluate(clean) > judge.evaluate(partial) > judge.evaluate(gives_it_away)


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


def test_questions_are_not_scored_at_all():
    """Stronger than the threshold this replaced: there is no question check,
    so no amount of question-shaping moves the score.

    The per-turn fraction of questions cost a run — mean r_ped climbed
    0.717 -> 0.992 while held-out solve fell 16.3% -> 6.2%, because reward
    carries `(r_ped - 1) * lambda` and driving r_ped to 1.0 cancels the
    pedagogy penalty outright. Reverting to the threshold removed the pressure
    but left the exploit reachable; removing the check removes both.
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

    assert len({with_questions(n) for n in range(5)}) == 1, (
        "question count must not move the score at all"
    )
