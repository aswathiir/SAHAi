from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from sahai.settings import ModelSettings


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_str
    ]


def load_model(
    name: str, dtype: str = "bfloat16", device: str = "cuda", quantize_4bit: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"trust_remote_code": True}

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
    name: str, dtype: str = "bfloat16", device: str = "cuda", quantize_4bit: bool = False
):
    model, tokenizer = load_model(name, dtype, device, quantize_4bit)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def load_for_training(name: str, settings: ModelSettings, device: str = "cuda"):
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
