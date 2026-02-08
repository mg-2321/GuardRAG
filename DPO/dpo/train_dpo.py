#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from transformers import TrainingArguments

from guardrag_training.dpo.trainer import GuardRAGDPOConfig, WeightedEntropyDPOTrainer
from guardrag_training.utils.modeling import PolicyLoadConfig, load_policy_with_lora, load_tokenizer


def main() -> None:
    """
    GuardRAG weighted DPO trainer.

    Implements Eq (2) from the paper:
      L = λ_s E_{P_s}[L_DPO] + λ_u E_{P_u}[L_DPO]

    plus an optional entropy regularizer hook (set --entropy-coef > 0).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ref-model", default=None)
    parser.add_argument("--dataset", required=True, help="JSONL with: prompt, chosen, rejected, pair_type")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambda-security", type=float, default=1.0)
    parser.add_argument("--lambda-utility", type=float, default=0.8)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.model)

    model = load_policy_with_lora(PolicyLoadConfig(model_name_or_path=args.model, load_in_4bit=args.load_in_4bit))
    ref_name = args.ref_model or args.model
    ref_model = load_policy_with_lora(PolicyLoadConfig(model_name_or_path=ref_name, load_in_4bit=args.load_in_4bit))

    train_ds = load_dataset("json", data_files={"train": args.dataset})["train"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.steps,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to=[],
    )

    guardrag_cfg = GuardRAGDPOConfig(
        lambda_security=args.lambda_security,
        lambda_utility=args.lambda_utility,
        entropy_coef=args.entropy_coef,
    )

    trainer = WeightedEntropyDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=min(1024, args.max_length - 256),
        guardrag_cfg=guardrag_cfg,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

