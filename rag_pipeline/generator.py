"""
Generation interface for GuardRAG pipelines.

Author: Gayatri Malladi

Supports:
  - Local HuggingFace models (Llama, Mistral) via provider="local"
  - OpenAI API (GPT-4o)                        via provider="openai"
  - Anthropic API (Claude 3.5 Sonnet/Haiku)    via provider="anthropic"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    model_name_or_path: str
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    device: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # "local" uses HuggingFace; "openai" / "anthropic" call their APIs
    provider: str = "local"
    api_key: Optional[str] = None
    trust_remote_code: bool = False
    use_chat_template: bool = False
    device_map: Optional[object] = None
    cpu_offload: bool = False
    max_gpu_memory: Optional[str] = None
    max_cpu_memory: Optional[str] = None
    offload_folder: Optional[str] = None


class Generator:
    """
    Unified generation wrapper.

    For local models (provider="local") it wraps HuggingFace CausalLM.
    For API models it delegates to api_generator.OpenAIGenerator /
    api_generator.AnthropicGenerator.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

        if config.provider in ("openai", "anthropic"):
            # API-based path — no GPU / torch required
            from .api_generator import build_api_generator
            self._impl = build_api_generator(config)
            self._use_api = True
            print(f"Using {config.provider.upper()} API model: {config.model_name_or_path}")
            return

        # ── Local HuggingFace path ────────────────────────────────────────
        self._use_api = False
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"Loading model on {device}...")
        trust_remote = getattr(config, 'trust_remote_code', False)

        dtype_name = getattr(config, "torch_dtype", None)
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        requested_dtype = dtype_map.get(str(dtype_name).lower()) if dtype_name else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=trust_remote
        )

        if device == "cuda":
            device_map = getattr(config, "device_map", None) or "auto"
            model_kwargs: dict = {
                "device_map": device_map,
                "low_cpu_mem_usage": True,
                "trust_remote_code": trust_remote,
            }
            max_memory = {}
            if getattr(config, "max_gpu_memory", None):
                max_memory[0] = getattr(config, "max_gpu_memory")
            if getattr(config, "max_cpu_memory", None):
                max_memory["cpu"] = getattr(config, "max_cpu_memory")
            if max_memory:
                model_kwargs["max_memory"] = max_memory
            if config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                print("Loading model with 4-bit quantization (QLoRA)...")
                quant_kwargs = dict(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                if getattr(config, "cpu_offload", False):
                    quant_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
                    offload_folder = getattr(config, "offload_folder", None)
                    if not offload_folder:
                        safe_name = os.path.basename(str(config.model_name_or_path).rstrip("/")) or "model"
                        offload_folder = f"/gscratch/uwb/gayat23/GuardRAG/cache/offload/{safe_name}"
                    os.makedirs(offload_folder, exist_ok=True)
                    model_kwargs["offload_folder"] = offload_folder
                    model_kwargs["offload_state_dict"] = True
                    print(f"Enabling CPU offload via {offload_folder} ...")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    **quant_kwargs
                )
            elif getattr(config, "load_in_8bit", False):
                print("Loading model with 8-bit quantization...")
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                except Exception:
                    # Fallback for much older transformers installs
                    model_kwargs["load_in_8bit"] = True
            else:
                model_kwargs["torch_dtype"] = requested_dtype or torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                device_map=None,
                torch_dtype=requested_dtype or torch.float32,
            )
            if device != "cpu":
                self.model = self.model.to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "cuda":
            self.model.eval()

    def generate(self, prompt: str) -> str:
        if self._use_api:
            return self._impl.generate(prompt)

        # ── Local generation ─────────────────────────────────────────────
        import torch

        if getattr(self.config, "use_chat_template", False) and hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        # Long batch evals on shared 24 GB GPUs can fragment CUDA memory over time.
        # Releasing cached blocks before each generation is slower, but avoids
        # late-run OOMs without changing model outputs.
        if self.device == "cuda":
            torch.cuda.empty_cache()

        with torch.inference_mode():
            do_sample = self.config.temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Slice off the prompt tokens so we return only the new text
        new_tokens = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
