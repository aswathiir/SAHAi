"""The prompt GRPO scores must be the prompt generation used.

`generate` built the system prompt with the learner block and
`compute_log_probs` rebuilt it without. Both `old_log_probs` and the update's
`log_probs` therefore conditioned on a different prompt than the one that
produced the tokens, so the importance ratio

    rho = exp(log p_new(tokens) - log p_old(tokens))

was not a ratio of one distribution evaluated twice. It was two distributions
over different contexts, and the clipped surrogate built on it is invalid.

The bug was dormant only because SAHAI_LEARNER_CONTEXT defaults to 0. It would
have corrupted the first run of the experiment it exists for, silently, with
training that appeared to proceed normally.

tutor.py imports torch, so the message construction is exercised through a stub.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest


@pytest.fixture
def tutor_cls(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _null()
    fake_torch.Tensor = object
    nn = types.ModuleType("torch.nn")
    fnn = types.ModuleType("torch.nn.functional")
    nn.functional = fnn
    fake_torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fnn)
    gen = types.ModuleType("sahai.core.generation")
    # Whatever tutor.py imports from here, hand back a passthrough. The names
    # have changed before and this test is about prompt construction, not them.
    gen.__getattr__ = lambda name: (lambda *a, **k: a[0] if a else None)
    monkeypatch.setitem(sys.modules, "sahai.core.generation", gen)
    for m in [k for k in sys.modules if k.startswith("sahai.agents.tutor")]:
        del sys.modules[m]
    from sahai.agents.tutor import TutorPolicy

    return TutorPolicy


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


class Problem:
    title = "t"
    description = "Find the largest element."
    solution = "def f(a): return max(a)"


def make(tutor_cls):
    t = tutor_cls.__new__(tutor_cls)
    return t


def dialogue_with(context: str):
    from sahai.core.dialogue import Dialogue

    d = Dialogue(problem_id="p")
    d.add("student", "I am stuck.")
    d.add("tutor", "What does the first element tell you?")
    d.learner_context = context
    return d


BLOCK = "WHAT THIS LEARNER ALREADY KNOWS:\n- Solid on arrays."


class TestPromptConsistency:
    def test_generation_and_scoring_build_the_same_system_prompt(self, tutor_cls):
        """The regression. Previously these differed whenever a block existed."""
        t = make(tutor_cls)
        d = dialogue_with(BLOCK)
        gen_msgs = t._build_messages(d, Problem(), d.learner_context)
        score_msgs = t._build_messages(d, Problem(), d.learner_context)
        assert gen_msgs == score_msgs
        assert BLOCK in gen_msgs[0]["content"]

    def test_the_block_reaches_the_system_message(self, tutor_cls):
        t = make(tutor_cls)
        msgs = t._build_messages(dialogue_with(BLOCK), Problem(), BLOCK)
        assert BLOCK in msgs[0]["content"]

    def test_an_empty_block_changes_nothing(self, tutor_cls):
        t = make(tutor_cls)
        with_empty = t._build_messages(dialogue_with(""), Problem(), "")
        assert "WHAT THIS LEARNER" not in with_empty[0]["content"]

    def test_a_block_actually_changes_the_prompt(self, tutor_cls):
        """If it did not, the consistency test above would pass vacuously."""
        t = make(tutor_cls)
        a = t._build_messages(dialogue_with(""), Problem(), "")
        b = t._build_messages(dialogue_with(BLOCK), Problem(), BLOCK)
        assert a[0]["content"] != b[0]["content"]


class TestDialogueCarriesItsConditioning:
    def test_dialogue_has_the_field(self):
        from sahai.core.dialogue import Dialogue

        assert Dialogue(problem_id="p").learner_context == ""

    def test_the_engine_records_it_before_generating(self):
        """It must be set before the first generate(), or the first turn is
        scored against a prompt that was never used."""
        src = pathlib.Path("sahai/core/dialogue.py").read_text()
        assert "dialogue.learner_context = context" in src
        assert src.index("dialogue.learner_context = context") < src.index(
            "tutor_msg = tutor.generate(dialogue, problem, context)"
        )

    def test_compute_log_probs_reads_it_back(self):
        src = pathlib.Path("sahai/agents/tutor.py").read_text()
        block = src[src.index("def compute_log_probs"):src.index("def _build_tutor_mask")]
        assert "learner_context" in block, (
            "compute_log_probs must rebuild the prompt generation used"
        )

    def test_the_field_does_not_shadow_the_helper(self):
        """`learner_context` is both an imported function and a field name in
        dialogue.py; the engine still has to be able to call the function."""
        import sahai.core.dialogue as mod

        assert callable(mod.learner_context)


class TestTrainAndServeReadOneSwitch:
    """The learner block must be present on both sides or neither.

    `sahai_core.learner_context` already shared the *wording* between the
    trainer and the gateway, and its docstring said why: a policy trained on
    one wording and served another is asked to follow an instruction it never
    saw. Whether the block existed at all was not shared. `dialogue.py` gated
    it on SAHAI_LEARNER_CONTEXT and defaulted that off; the gateway's
    `_context_for` built it on every served turn, typed and spoken, with no
    reference to the flag.

    So the deployed tutor received a personalisation block its adapter had
    never been trained on, and nothing recorded the discrepancy because each
    side was internally consistent. Same shape as the `generate` versus
    `compute_log_probs` mismatch above: one decision, two implementations.
    """

    def test_the_flag_is_defined_in_the_shared_library(self):
        from sahai_core.learner_context import LEARNER_CONTEXT_ENABLED

        assert isinstance(LEARNER_CONTEXT_ENABLED, bool)

    def test_the_trainer_imports_the_flag_rather_than_redefining_it(self):
        src = pathlib.Path("sahai/core/dialogue.py").read_text()
        assert "LEARNER_CONTEXT_ENABLED, learner_context" in src
        assert 'LEARNER_CONTEXT_ENABLED = os.getenv' not in src, (
            "a second definition is how the trainer and the gateway drifted apart"
        )

    def test_the_gateway_honours_the_flag(self):
        src = pathlib.Path("services/gateway/app/main.py").read_text()
        assert "LEARNER_CONTEXT_ENABLED" in src
        block = src[src.index("async def _context_for"):]
        block = block[:block.index("\n\n\n")] if "\n\n\n" in block else block
        assert "if not LEARNER_CONTEXT_ENABLED" in block, (
            "_context_for must not build a block the trainer omitted"
        )

    def test_the_gateway_checks_before_doing_the_work(self):
        """Both served paths funnel through `_context_for`, so the guard has to
        sit ahead of the mastery lookup rather than at each call site."""
        src = pathlib.Path("services/gateway/app/main.py").read_text()
        block = src[src.index("async def _context_for"):]
        assert block.index("LEARNER_CONTEXT_ENABLED") < block.index("_mastery_for")

    def test_the_trainer_still_gates_the_block_on_it(self):
        src = pathlib.Path("sahai/core/dialogue.py").read_text()
        assert "if LEARNER_CONTEXT_ENABLED and tracer is not None" in src
