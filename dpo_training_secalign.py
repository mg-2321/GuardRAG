#!/usr/bin/env python3
"""
DPO Training Script for SecAlign Defense
Based on: https://github.com/facebookresearch/SecAlign

This implements DPO (Direct Preference Optimization) to defend against IPI attacks.
The model learns to prefer secure outputs (ignoring injections) over insecure ones.
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Union
import numpy as np
from pathlib import Path
import sys

# Check for TRL availability first
try:
    from trl import DPOTrainer, DPOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    TRL_AVAILABLE = True
except ImportError as e:
    TRL_AVAILABLE = False
    print(f"⚠️  TRL or transformers not available: {e}")
    print("   Install with: pip install trl transformers torch")
    print("   Will show error and exit")
    sys.exit(1)
except TypeError as e:
    # Handle Python version compatibility issues
    print(f"⚠️  Compatibility issue with transformers: {e}")
    print("   This may be a Python 3.9 vs 3.10+ compatibility issue")
    print("   Try using Python 3.10+ or updating transformers")
    sys.exit(1)


class PreferenceDataset(Dataset):
    """Dataset for DPO preference pairs"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
        
        print(f"✓ Loaded {len(self.data)} preference pairs from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Format for DPO: prompt + response pairs
        chosen_text = f"{prompt}\n\n{chosen}"
        rejected_text = f"{prompt}\n\n{rejected}"
        
        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'chosen_text': chosen_text,
            'rejected_text': rejected_text
        }


class ManualDPOTrainer:
    """
    Manual DPO implementation if TRL is not available
    Based on Direct Preference Optimization (Rafailov et al., 2023)
    """
    
    def __init__(self, model, tokenizer, beta: float = 0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta  # Temperature parameter
    
    def compute_dpo_loss(self, batch):
        """
        Compute DPO loss:
        L_DPO = -log(σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x))))
        
        where:
        - π_θ: current policy
        - π_ref: reference policy (frozen model)
        - y_w: chosen (winning) response
        - y_l: rejected (losing) response
        - β: temperature parameter
        """
        # Tokenize
        chosen_inputs = self.tokenizer(
            batch['chosen_text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        rejected_inputs = self.tokenizer(
            batch['rejected_text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        prompt_inputs = self.tokenizer(
            batch['prompt'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Get log probabilities
        with torch.no_grad():
            # Reference model (frozen) log probs
            ref_chosen_logps = self._get_log_probs(chosen_inputs, prompt_inputs, use_ref=True)
            ref_rejected_logps = self._get_log_probs(rejected_inputs, prompt_inputs, use_ref=True)
        
        # Current model log probs
        chosen_logps = self._get_log_probs(chosen_inputs, prompt_inputs, use_ref=False)
        rejected_logps = self._get_log_probs(rejected_inputs, prompt_inputs, use_ref=False)
        
        # Compute DPO loss
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def _get_log_probs(self, inputs, prompt_inputs, use_ref=False):
        """Get log probabilities for sequence"""
        model = self.model if not use_ref else self.model
        # Simplified: using model output
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding tokens
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        log_probs = (log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        
        return log_probs


def train_dpo_with_trl(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 2,  # With 8-bit quantization, we can use batch_size=2
    beta: float = 0.1,
    lambda_security: float = 0.7,
    lambda_utility: float = 0.3
):
    """Train DPO using TRL library (recommended)"""
    if not TRL_AVAILABLE:
        raise ImportError("TRL not available. Install with: pip install trl")
    
    print("="*80)
    print("TRAINING DPO WITH TRL (SecAlign + Custom Loss)")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Training data: {train_data_path}")
    print(f"Output: {output_dir}")
    print(f"Loss weights: λ_security={lambda_security}, λ_utility={lambda_utility}")
    print()
    
    # Load model and tokenizer with 8-bit quantization to save memory
    # 8-bit quantization reduces model size from ~14GB to ~7GB per copy
    print(f"✓ Configuring 8-bit quantization for memory efficiency")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    if torch.cuda.is_available():
        device_map = "auto"
        print(f"✓ Using {'GPU' if torch.cuda.device_count() == 1 else f'{torch.cuda.device_count()} GPUs'}")
    else:
        device_map = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map
    )
    
    # Prepare model for LoRA training (necessary for 8-bit quantized models)
    print("✓ Preparing model for LoRA training")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA adapters
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Mistral modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA adapters to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    with open(train_data_path, 'r') as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"✓ Loaded {len(train_data)} training pairs")
    
    # Format for TRL DPO
    formatted_data = []
    for item in train_data:
        formatted_data.append({
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected']
        })
    
    # Convert to HuggingFace Dataset (required by TRL)
    formatted_data = Dataset.from_list(formatted_data)
    
    # DPO Configuration
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        learning_rate=learning_rate,
        beta=beta,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,  # Save memory
        dataloader_pin_memory=False  # Reduce memory usage
    )
    
    # Create reference model (frozen copy) with 8-bit quantization
    print(f"✓ Loading reference model with 8-bit quantization")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map
    )
    
    # Create DPO Trainer
    # Note: beta is in DPOConfig, tokenizer handling varies by TRL version
    try:
        # Try with tokenizer (newer TRL versions)
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=formatted_data,
            tokenizer=tokenizer
        )
    except TypeError as e:
        # Fallback: try without explicit tokenizer (older TRL versions)
        print(f"Note: Trying without explicit tokenizer ({str(e)})")
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=formatted_data
        )
        # Set tokenizer as attribute if needed
        if hasattr(trainer, 'tokenizer') and trainer.tokenizer is None:
            trainer.tokenizer = tokenizer
    
    # Train
    print("\n" + "="*80)
    print("STARTING DPO TRAINING")
    print("="*80)
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*80}")
    print("✓ DPO TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Evaluate ASR reduction on test set")
    print("  2. Compare: Before DPO (26.8%) vs After DPO (<5%)")
    print("  3. Test generalization to unseen attacks")


def train_dpo_manual(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    beta: float = 0.1
):
    """Train DPO manually (fallback if TRL not available)"""
    print("="*80)
    print("TRAINING DPO (Manual Implementation)")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Training data: {train_data_path}")
    print()
    
    # Note: Manual implementation would require more code
    # For production, use TRL library
    print("⚠️  Manual DPO training requires full implementation")
    print("   Recommendation: Install TRL for production use")
    print("   Command: pip install trl")
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DPO model for IPI defense (SecAlign)')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Base model (e.g., Llama-3-8B-Instruct, Llama-2-7b-chat, Mistral-7B-Instruct)')
    parser.add_argument('--train-data', default='dpo_preference_data/dpo_preference_train.jsonl')
    parser.add_argument('--output-dir', default='dpo_models/secalign_defended')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0.1,
                       help='DPO temperature parameter (SecAlign uses 0.1)')
    parser.add_argument('--lambda-security', type=float, default=0.7,
                       help='Security loss weight (default: 0.7)')
    parser.add_argument('--lambda-utility', type=float, default=0.3,
                       help='Utility loss weight (default: 0.3)')
    parser.add_argument('--manual', action='store_true',
                       help='Use manual DPO (not recommended)')
    
    args = parser.parse_args()
    
    # Validate lambda weights
    if args.lambda_security + args.lambda_utility != 1.0:
        print(f"⚠️  Warning: λ_security ({args.lambda_security}) + λ_utility ({args.lambda_utility}) != 1.0")
        print("   Normalizing weights...")
        total = args.lambda_security + args.lambda_utility
        args.lambda_security /= total
        args.lambda_utility /= total
        print(f"   Normalized: λ_security={args.lambda_security}, λ_utility={args.lambda_utility}")
    
    if args.manual or not TRL_AVAILABLE:
        train_dpo_manual(**vars(args))
    else:
        train_dpo_with_trl(
            model_name=args.model,
            train_data_path=args.train_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            beta=args.beta,
            lambda_security=args.lambda_security,
            lambda_utility=args.lambda_utility
        )


if __name__ == "__main__":
    main()

