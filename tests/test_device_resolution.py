"""Device selection, including Apple Silicon.

models.py imports torch, which this venv deliberately does not have (see
test_reward.py), so these read the source. Thin, but they pin the two
decisions that would silently waste a run: resolving `auto` in the right
order, and refusing 4-bit where bitsandbytes cannot work.
"""

from __future__ import annotations

import pathlib

import pytest

SRC = pathlib.Path("sahai/core/models.py").read_text()


class TestResolveDevice:
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        block = SRC[SRC.index("def resolve_device"):SRC.index("def _quantize_is_supported")]
        assert block.index("cuda.is_available") < block.index("mps.is_available"), (
            "cuda must be checked before mps"
        )
        assert block.index("mps.is_available") < block.rindex('"cpu"'), (
            "cpu is the last resort"
        )

    def test_an_explicit_device_is_honoured(self):
        """A Kaggle run that asks for cuda must fail loudly if it is missing,
        not quietly fall to a CPU that would take days."""
        block = SRC[SRC.index("def resolve_device"):SRC.index("def _quantize_is_supported")]
        assert 'if requested != "auto":' in block
        assert "return requested" in block

    def test_mps_is_guarded_for_older_torch(self):
        """torch.backends.mps does not exist on every build."""
        block = SRC[SRC.index("def resolve_device"):SRC.index("def _quantize_is_supported")]
        assert 'getattr(torch.backends, "mps"' in block


class TestQuantizationGuard:
    def test_four_bit_is_cuda_only(self):
        block = SRC[SRC.index("def _quantize_is_supported"):SRC.index("def load_model")]
        assert 'return device.startswith("cuda")' in block

    def test_unsupported_quantization_warns_and_continues(self):
        """Dropping to bfloat16 costs memory, not accuracy. Failing the load
        would cost the run."""
        block = SRC[SRC.index("def load_model"):SRC.index("def load_for_inference")]
        assert "logger.warning" in block
        assert "quantize_4bit = False" in block

    def test_the_guard_runs_before_the_quantized_branch(self):
        block = SRC[SRC.index("def load_model"):SRC.index("def load_for_inference")]
        assert block.index("_quantize_is_supported") < block.index("BitsAndBytesConfig")


class TestLoadersDefaultToAuto:
    @pytest.mark.parametrize("fn", ["load_model", "load_for_inference", "load_for_training"])
    def test_default_device_is_auto(self, fn):
        sig_start = SRC.index(f"def {fn}(")
        sig = SRC[sig_start:SRC.index(":", SRC.index(")", sig_start))]
        assert 'device: str = "auto"' in sig, f"{fn} should default to auto"
