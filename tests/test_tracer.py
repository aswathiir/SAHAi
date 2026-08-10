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


def test_zpd():
    tracer = BKTTracer(TracerSettings(p_init=0.5, zpd_low=0.3, zpd_high=0.7))
    assert tracer.in_zpd(1, ["arrays"])
    tracer.skills["arrays"] = 0.1
    assert not tracer.in_zpd(1, ["arrays"])
    tracer.skills["arrays"] = 0.9
    assert not tracer.in_zpd(1, ["arrays"])


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
