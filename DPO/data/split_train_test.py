#!/usr/bin/env python3
"""
Split DPO preference pairs into train/test sets.

Splits by query_id to avoid data leakage (all pairs for a given query
go to either train or test, never both).

Usage:
    python split_train_test.py \
        --input data/preference/scifact_v4b_clean.jsonl \
        --train-out data/preference/scifact_train.jsonl \
        --test-out data/preference/scifact_test.jsonl \
        --train-frac 0.9 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Split DPO pairs into train/test by query_id")
    ap.add_argument("--input", required=True, help="Input JSONL file")
    ap.add_argument("--train-out", required=True, help="Output train JSONL file")
    ap.add_argument("--test-out", required=True, help="Output test JSONL file")
    ap.add_argument("--train-frac", type=float, default=0.9, help="Fraction for train (default: 0.9)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--stratify-by-type", action="store_true", 
                    help="Stratify split by pair_type (security/utility)")
    args = ap.parse_args()

    # Group examples by query_id
    groups = defaultdict(list)
    for ex in read_jsonl(args.input):
        qid = ex.get("query_id")
        if qid is None:
            # fallback: keep no-qid examples together
            qid = "__NO_QID__"
        groups[qid].append(ex)

    if args.stratify_by_type:
        # Stratified split: maintain pair_type ratio in train/test
        security_qids = []
        utility_qids = []
        
        for qid, rows in groups.items():
            # Determine dominant type for this query
            types = [r.get("pair_type", "utility") for r in rows]
            if types.count("security") > types.count("utility"):
                security_qids.append(qid)
            else:
                utility_qids.append(qid)
        
        rng = random.Random(args.seed)
        rng.shuffle(security_qids)
        rng.shuffle(utility_qids)
        
        n_train_sec = int(len(security_qids) * args.train_frac)
        n_train_util = int(len(utility_qids) * args.train_frac)
        
        train_ids = set(security_qids[:n_train_sec] + utility_qids[:n_train_util])
        
        print(f"Stratified split:")
        print(f"  Security: {n_train_sec}/{len(security_qids)} train, {len(security_qids)-n_train_sec} test")
        print(f"  Utility:  {n_train_util}/{len(utility_qids)} train, {len(utility_qids)-n_train_util} test")
    else:
        # Simple random split
        qids = list(groups.keys())
        random.Random(args.seed).shuffle(qids)
        n_train = int(len(qids) * args.train_frac)
        train_ids = set(qids[:n_train])

    train_rows, test_rows = [], []
    for qid, rows in groups.items():
        (train_rows if qid in train_ids else test_rows).extend(rows)

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.test_out, test_rows)

    print(f"\nSplit complete:")
    print(f"  Queries: {len(groups)} total | {len(train_ids)} train | {len(groups)-len(train_ids)} test")
    print(f"  Pairs:   {len(train_rows)} train | {len(test_rows)} test")
    print(f"  Train:   {args.train_out}")
    print(f"  Test:    {args.test_out}")


if __name__ == "__main__":
    main()
