#!/usr/bin/env python3
from __future__ import annotations

"""
Train reward and cost models (Eq 4, Eq 5 in the paper).

We train two scalar scoring models on preference pairs:
  - reward model r_phi prefers chosen over rejected
  - cost model  c_psi prefers rejected over chosen (violation severity)

This implementation uses a lightweight training loop with Accelerate.
"""

import argparse
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from DPO.reward_model.reward_cost_models import pairwise_cost_loss, pairwise_ranking_loss


def _collate(tokenizer, batch, max_length: int):
    prompts = [b["prompt"] for b in batch]
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]

    # Score full prompt+response, as in common RM training
    chosen_text = [p + c for p, c in zip(prompts, chosen)]
    rejected_text = [p + r for p, r in zip(prompts, rejected)]

    chosen_enc = tokenizer(chosen_text, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    rejected_enc = tokenizer(rejected_text, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    return chosen_enc, rejected_enc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="Base LM for initializing RM/CM (sequence classification head)")
    p.add_argument("--dataset", required=True, help="JSONL with prompt/chosen/rejected")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    accelerator = Accelerator()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Scalar regression head: num_labels=1
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    cost_model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)

    ds = load_dataset("json", data_files={"train": args.dataset})["train"]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: _collate(tok, b, args.max_length))

    opt_r = torch.optim.AdamW(reward_model.parameters(), lr=args.lr, weight_decay=0.01)
    opt_c = torch.optim.AdamW(cost_model.parameters(), lr=args.lr, weight_decay=0.01)

    sched_r = get_cosine_schedule_with_warmup(opt_r, num_warmup_steps=args.warmup, num_training_steps=args.steps)
    sched_c = get_cosine_schedule_with_warmup(opt_c, num_warmup_steps=args.warmup, num_training_steps=args.steps)

    reward_model, cost_model, opt_r, opt_c, dl, sched_r, sched_c = accelerator.prepare(
        reward_model, cost_model, opt_r, opt_c, dl, sched_r, sched_c
    )

    reward_model.train()
    cost_model.train()

    step = 0
    it = iter(dl)
    while step < args.steps:
        try:
            chosen_enc, rejected_enc = next(it)
        except StopIteration:
            it = iter(dl)
            chosen_enc, rejected_enc = next(it)

        # Reward model: chosen > rejected
        r_pos = reward_model(**chosen_enc).logits.squeeze(-1)
        r_neg = reward_model(**rejected_enc).logits.squeeze(-1)
        loss_r = pairwise_ranking_loss(r_pos, r_neg)

        # Cost model: rejected (bad) > chosen (good)
        c_bad = cost_model(**rejected_enc).logits.squeeze(-1)
        c_good = cost_model(**chosen_enc).logits.squeeze(-1)
        loss_c = pairwise_cost_loss(c_bad, c_good)

        accelerator.backward(loss_r)
        accelerator.backward(loss_c)

        opt_r.step()
        opt_c.step()
        sched_r.step()
        sched_c.step()
        opt_r.zero_grad(set_to_none=True)
        opt_c.zero_grad(set_to_none=True)

        if accelerator.is_main_process and step % 25 == 0:
            accelerator.print(f"step {step} | loss_r={loss_r.item():.4f} | loss_c={loss_c.item():.4f}")
        step += 1

    if accelerator.is_main_process:
        out = Path(args.out_dir)
        (out / "reward_model").mkdir(parents=True, exist_ok=True)
        (out / "cost_model").mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(reward_model).save_pretrained(out / "reward_model")
        accelerator.unwrap_model(cost_model).save_pretrained(out / "cost_model")
        tok.save_pretrained(out / "reward_model")
        tok.save_pretrained(out / "cost_model")


if __name__ == "__main__":
    main()
