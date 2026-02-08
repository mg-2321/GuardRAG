"""
Generation interface for GuardRAG pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    model_name_or_path: str
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    device: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False


class Generator:
    """Thin wrapper around Hugging Face causal models."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"Loading model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        
        # Optimize model loading for GPU
        if device == "cuda":
            # Use float16 for faster inference and less memory
            # Use device_map="auto" for multi-GPU support
            # Use dtype for explicit dtype control (torch_dtype is deprecated)
            import torch
            
            # Prepare model loading kwargs
            model_kwargs = {
                'device_map': "auto",
                'low_cpu_mem_usage': True,
            }
            
            # Add quantization if requested (for large models like 70B)
            if config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                print("Loading model with 4-bit quantization (QLoRA)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs['quantization_config'] = quantization_config
            elif config.load_in_8bit:
                print("Loading model with 8-bit quantization...")
                model_kwargs['load_in_8bit'] = True
            else:
                # Standard float16 loading for smaller models
                model_kwargs['torch_dtype'] = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                **model_kwargs
            )
            # Compile model for faster inference (PyTorch 2.0+)
            # Note: torch.compile can take 5-10 minutes on first run, but speeds up subsequent inference
            # Commenting out for now to avoid long initial compilation time
            # Uncomment if you want to optimize for multiple runs
            # try:
            #     print("Compiling model with torch.compile (this may take 5-10 minutes)...")
            #     self.model = torch.compile(self.model, mode="reduce-overhead")
            #     print("Model compiled with torch.compile for optimization")
            # except Exception as e:
            #     print(f"torch.compile not available or failed: {e}, continuing without compilation")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                device_map=None,
                torch_dtype=torch.float32,
            )
            if device != "cpu":
                self.model = self.model.to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Enable optimizations
        if device == "cuda":
            self.model.eval()  # Set to evaluation mode

    def generate(self, prompt: str) -> str:
        # Tokenize and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use inference_mode for better performance (faster than no_grad)
        with torch.inference_mode():
            # If temperature is 0.0, use greedy decoding (do_sample=False)
            do_sample = self.config.temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

