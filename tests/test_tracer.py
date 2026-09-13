from sahai.agents.tracer import BKTTracer
from sahai.settings import TracerSettings


def test_initial_mastery():
    tracer = BKTTracer(TracerSettings(p_init=0.3))
    assert tracer.get_mastery("arrays") == 0.3


def test_mastery_increases_on_correct():
    tracer = BKTTracer()
    initial = tracer.get_mastery("arrays")
    tracer.update("arrays", correct=True)
    assert tracer.get_mastery("arrays") > initial


def test_mastery_decreases_on_incorrect():
    tracer = BKTTracer(TracerSettings(p_init=0.6, p_learn=0.0))
    tracer.get_mastery("arrays")
    initial = tracer.skills["arrays"]
    tracer.update("arrays", correct=False)
    assert tracer.skills["arrays"] < initial + 0.01


def test_ability_average():
    tracer = BKTTracer(TracerSettings(p_init=0.5))
    tracer.get_mastery("arrays")
    tracer.get_mastery("sorting")
    assert abs(tracer.get_ability() - 0.5) < 1e-6


def test_zpd_is_a_difficulty_window_around_the_target():
    tracer = BKTTracer(TracerSettings(p_init=0.5, zpd_low=0.3, zpd_high=0.7))
    tracer.skills["arrays"] = 0.5          # target = 1 + 0.5*4 = 3.0
    assert tracer.target_difficulty(5) == 3.0
    assert tracer.in_zpd(2, ["arrays"], 5)
    assert tracer.in_zpd(4, ["arrays"], 5)
    assert not tracer.in_zpd(1, ["arrays"], 5)
    assert not tracer.in_zpd(5, ["arrays"], 5)


def test_zpd_excludes_mastery_above_the_ceiling_but_not_below_a_floor():
    """There is deliberately no lower mastery bound any more.

    `zpd_low` as a floor meant a learner failing a skill was removed from that
    skill's problems entirely, and since BKT sends a failed skill to ~0.109 and
    never lifts it, that removal was permanent. Difficulty targeting is what
    responds to a struggling learner now; the ceiling only stops work they have
    already demonstrated.
    """
    tracer = BKTTracer(TracerSettings(p_init=0.5, zpd_low=0.3, zpd_high=0.7))

    tracer.skills["arrays"] = 0.05
    assert tracer.in_zpd(1, ["arrays"], 5), "a failing learner was excluded again"

    tracer.skills["arrays"] = 0.9
    assert not tracer.in_zpd(5, ["arrays"], 5)


def test_reset():
    tracer = BKTTracer()
    tracer.update("arrays", True)
    assert len(tracer.skills) == 1
    tracer.reset()
    assert len(tracer.skills) == 0


def test_update_batch():
    tracer = BKTTracer()
    tracer.update_batch(["arrays", "sorting"], correct=True)
    assert "arrays" in tracer.skills
    assert "sorting" in tracer.skills
    assert tracer.get_mastery("arrays") > tracer.settings.p_init
