#!/usr/bin/env python3
from __future__ import annotations

"""
Constrained PPO (RLHF refinement) as described in the paper:

  maximize E[r_phi(c,y)]
  s.t.    E[c_psi(c,y)] <= tau

Lagrangian relaxation (Eq 7):
  J = E[r - lambda_c * c] - beta_KL * KL(pi || pi_DPO)

We implement this using TRL's PPOTrainer by folding cost into the reward:
  reward' = r - lambda_c * c

and adapt lambda_c every `--lagrange-update-every` steps:
  if mean_cost > tau: lambda_c *= 1.2
  else:               lambda_c *= 0.9
  clamp to [0.1, 10.0]
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


@torch.no_grad()
def score_scalar(model, tokenizer, texts, max_length: int) -> torch.Tensor:
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(model.device)
    logits = model(**enc).logits.squeeze(-1)
    return logits.detach().float().cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="DPO policy checkpoint (LoRA merged or adapters)")
    p.add_argument("--ref-policy", required=True, help="Reference policy for KL control (π_DPO)")
    p.add_argument("--reward-model", required=True)
    p.add_argument("--cost-model", required=True)
    p.add_argument("--dataset", required=True, help="JSONL with prompts (field: prompt)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--gen-max-new", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--beta-kl", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.0, help="Cost threshold τ")
    p.add_argument("--lambda-c", type=float, default=1.0, help="Initial Lagrange multiplier")
    p.add_argument("--lagrange-update-every", type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.ref_policy, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Policy + value head
    # NOTE: PPO requires a value head. We load from checkpoints directly.
    # If you are using LoRA adapters, pass the *merged* policy checkpoint path here
    # (or extend this script to apply PEFT adapters to the value-head model).
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy, device_map="auto")
    ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.ref_policy, device_map="auto")

    rm = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, device_map="auto")
    cm = AutoModelForSequenceClassification.from_pretrained(args.cost_model, num_labels=1, device_map="auto")

    ds = load_dataset("json", data_files={"train": args.dataset})["train"]

    ppo_cfg = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        learning_rate=1e-6,
        ppo_epochs=4,
        cliprange=0.2,
        vf_coef=1.0,
        kl_penalty="kl",
        kl_coef=args.beta_kl,
    )

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy,
        ref_model=ref_policy,
        tokenizer=tok,
        dataset=ds,
    )

    lambda_c = float(args.lambda_c)
    costs_window = []

    for step, batch in enumerate(trainer.dataloader):
        if step >= args.steps:
            break

        prompts = batch["prompt"]
        query_tensors = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).input_ids

        response_tensors = trainer.generate(
            query_tensors.to(trainer.accelerator.device),
            max_new_tokens=args.gen_max_new,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

        responses = tok.batch_decode(response_tensors, skip_special_tokens=True)

        # Score prompt+response
        texts = responses
        rewards = score_scalar(rm, tok, texts, max_length=args.max_length)
        costs = score_scalar(cm, tok, texts, max_length=args.max_length)
        costs_window.append(costs.mean().item())

        # Fold constrained objective into scalar reward
        shaped = rewards - (lambda_c * costs)
        shaped_list = [float(x) for x in shaped.tolist()]

        trainer.step(query_tensors, response_tensors, shaped_list)

        # Update lambda_c
        if (step + 1) % args.lagrange_update_every == 0:
            mean_cost = float(sum(costs_window) / len(costs_window))
            costs_window.clear()
            if mean_cost > args.tau:
                lambda_c *= 1.2
            else:
                lambda_c *= 0.9
            lambda_c = float(max(0.1, min(10.0, lambda_c)))
            if trainer.accelerator.is_main_process:
                print(f"step {step} | mean_cost={mean_cost:.4f} | tau={args.tau:.4f} | lambda_c={lambda_c:.3f}")

        if (step + 1) % 200 == 0 and trainer.accelerator.is_main_process:
            out = Path(args.out_dir) / f"checkpoint_{step+1}"
            out.mkdir(parents=True, exist_ok=True)
            trainer.save_pretrained(out)

    if trainer.accelerator.is_main_process:
        trainer.save_pretrained(Path(args.out_dir) / "final")


if __name__ == "__main__":
    main()

