#!/usr/bin/env python3
"""
Run Stages 6-7 evaluation using Llama 3.1 / 3.3 70B (4-bit quantized).

This is a thin wrapper around run_stages_6_7_with_generation.py that
defaults model to llama-3.1-70b-4bit and exposes the same CLI flags.

Usage:
  python evaluation/run_stages_6_7_llama70b.py --corpus nfcorpus
  python evaluation/run_stages_6_7_llama70b.py --corpus nfcorpus --model llama-3.3-70b-4bit
  python evaluation/run_stages_6_7_llama70b.py --corpus scifact --sample-size 200
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_stages_6_7_with_generation import evaluate_stages_6_7, CORPUS_CONFIGS

LARGE_MODEL_CHOICES = [
    'llama-3.1-70b',
    'llama-3.1-70b-4bit',
    'llama-3.3-70b',
    'llama-3.3-70b-4bit',
    'llama-2-70b',
    'mixtral-8x7b',
    'gpt-4o',
    'gpt-4o-mini',
    'claude-3-5-sonnet',
    'claude-3-5-haiku',
]


def main():
    parser = argparse.ArgumentParser(
        description='Run Stages 6-7 evaluation with large / API models'
    )
    parser.add_argument('--corpus', required=True, choices=list(CORPUS_CONFIGS.keys()),
                        help='Corpus name')
    parser.add_argument('--model', default='llama-3.1-70b-4bit',
                        help='Model key (default: llama-3.1-70b-4bit) or local path to model directory')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of queries to evaluate (default: all)')

    args = parser.parse_args()
    cfg  = CORPUS_CONFIGS[args.corpus]

    if not Path(cfg['corpus']).exists():
        print(f"Corpus file not found: {cfg['corpus']}")
        sys.exit(1)
    if not Path(cfg['queries']).exists():
        print(f"Queries file not found: {cfg['queries']}")
        sys.exit(1)

    # Accept either a known model key or a local path
    model_arg = args.model
    if Path(model_arg).exists():
        model_key = model_arg  # local path
    else:
        model_key = model_arg  # keep as string for known models

    evaluate_stages_6_7(
        corpus_name=args.corpus,
        corpus_path=cfg['corpus'],
        metadata_path=cfg['metadata'],
        queries_path=cfg['queries'],
        model_key=model_key,
        sample_size=args.sample_size,
    )


if __name__ == '__main__':
    main()
