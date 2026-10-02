from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from sahai.settings import ModelSettings


logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_str
    ]


def resolve_device(requested: str = "auto") -> str:
    """Pick a device, preferring the fastest available.

    `auto` resolves cuda -> mps -> cpu. Anything else is honoured as given, so
    a Kaggle run that says "cuda" still fails loudly if CUDA is missing rather
    than silently dropping to a CPU that would take days.

    mps is Apple Silicon's GPU. It is a real option for this project: both
    models are 1.5B, which is about 6 GB in bfloat16 against the 16 GB unified
    memory on an M4, so the whole benchmark fits without quantisation.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _quantize_is_supported(device: str) -> bool:
    """4-bit needs bitsandbytes, which needs CUDA.

    On mps the import succeeds and the load then fails deep inside with an
    unhelpful error, so refuse it here and say why. The cost of dropping to
    bfloat16 is memory, not correctness, and at 1.5B there is enough of it.
    """
    return device.startswith("cuda")


def load_model(
    name: str, dtype: str = "bfloat16", device: str = "auto", quantize_4bit: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"trust_remote_code": True}

    if quantize_4bit and not _quantize_is_supported(device):
        logger.warning(
            "4-bit quantization requested but unavailable on %s (bitsandbytes "
            "is CUDA-only); loading in %s instead. This costs memory, not "
            "accuracy, and at 1.5B there is room for it.",
            device, dtype,
        )
        quantize_4bit = False

    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_resolve_dtype(dtype),
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = _resolve_dtype(dtype)
        kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    return model, tokenizer


def load_for_inference(
    name: str, dtype: str = "bfloat16", device: str = "auto", quantize_4bit: bool = False
):
    model, tokenizer = load_model(name, dtype, device, quantize_4bit)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def load_for_training(name: str, settings: ModelSettings, device: str = "auto"):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, tokenizer = load_model(name, settings.dtype, device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer
