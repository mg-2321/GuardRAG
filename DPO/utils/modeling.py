from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("transformers is required for training utilities") from exc

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover
    raise ImportError("peft is required for LoRA/QLoRA training") from exc


@dataclass(frozen=True)
class PolicyLoadConfig:
    model_name_or_path: str
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str | dict = "auto"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def load_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy_with_lora(cfg: PolicyLoadConfig):
    quant_config: Optional[BitsAndBytesConfig] = None
    if cfg.load_in_4bit or cfg.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit,
            bnb_4bit_compute_dtype=cfg.torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
        quantization_config=quant_config,
        trust_remote_code=False,
    )

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model

