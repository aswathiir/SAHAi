"""The learner block must reach the training prompt, and match serving.

It previously existed only at serving time, so the policy was optimised on
prompts that never contained it and then asked at inference to act on it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "libs" / "sahai-core"))

from sahai_core.learner_context import learner_context  # noqa: E402


def test_new_and_solid_are_named_as_moves_not_numbers():
    out = learner_context(["arrays", "hash_maps"], {"arrays": 0.10, "hash_maps": 0.80})
    assert "New to arrays" in out
    assert "Solid on hash maps" in out
    assert "0.10" not in out and "0.80" not in out


def test_middling_skills_are_omitted():
    """Between the thresholds is where a tutor should already be pitching."""
    assert learner_context(["arrays"], {"arrays": 0.5}) == ""


def test_unseen_skill_counts_as_new_not_middling():
    assert "New to trees" in learner_context(["trees"], {})


def test_only_the_current_problem_s_skills_appear():
    out = learner_context(["arrays"], {"arrays": 0.1, "graphs": 0.1})
    assert "graphs" not in out


def test_no_skills_means_no_block():
    assert learner_context([], {"arrays": 0.1}) == ""


def test_underscores_are_humanised():
    out = learner_context(["hash_maps"], {"hash_maps": 0.9})
    assert "hash maps" in out and "hash_maps" not in out
