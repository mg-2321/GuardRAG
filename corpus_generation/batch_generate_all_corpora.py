#!/usr/bin/env python3
"""
Batch Generate IPI Attacks for All Corpora
Generates realistic / hard / stress poisoned corpora for:
  - nfcorpus (medical)
  - scifact  (scientific)
  - fiqa     (financial)
  - hotpotqa (general multi-hop)
  - msmarco  (web passages)
"""

import subprocess
import sys
from pathlib import Path
import argparse


CORPUS_CONFIGS = {
    'nfcorpus': {
        'script':  'corpus_generation/ipi_generator_v4_semantic.py',
        'corpus':  'data/corpus/beir/nfcorpus/corpus.jsonl',
        'queries': 'data/corpus/beir/nfcorpus/queries.jsonl',
        'dataset': 'nfcorpus',
        'domain':  'biomedical',
    },
    'scifact': {
        'script':  'corpus_generation/ipi_generator_scifact.py',
        'corpus':  'data/corpus/beir/scifact/corpus.jsonl',
        'queries': 'data/corpus/beir/scifact/queries.jsonl',
        'dataset': 'scifact',
        'domain':  None,   
    },
    'fiqa': {
        'script':  'corpus_generation/ipi_generator_v4_semantic.py',
        'corpus':  'data/corpus/beir/fiqa/corpus.jsonl',
        'queries': 'data/corpus/beir/fiqa/queries.jsonl',
        'dataset': 'fiqa',
        'domain':  'financial',
    },
    'hotpotqa': {
        'script':  'corpus_generation/ipi_generator_v4_semantic.py',
        'corpus':  'data/corpus/beir/hotpotqa/corpus.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl',
        'dataset': 'hotpotqa',
        'domain':  'general',
    },
    'msmarco': {
        'script':  'corpus_generation/ipi_generator_v4_semantic.py',
        'corpus':  'data/corpus/beir/msmarco/corpus.jsonl',
        'queries': 'data/corpus/beir/msmarco/queries.jsonl',
        'dataset': 'msmarco',
        'domain':  'web',
    },
}


MODE_SUFFIX = {
    "realistic": "realistic_attack",
    "hard":      "hard_attacks",
    "stress":    "stress_test_attacks",
}


def check_file(filepath: str) -> bool:
    path = Path(filepath)
    if not path.exists():
        print(f" Not found: {filepath}")
        return False
    return True


def generate_one(corpus_name: str, config: dict, mode: str, skip_existing: bool) -> bool:
    """Generate IPI attacks for one corpus + mode combination."""
    print("\n" + "=" * 70)
    print(f"  {corpus_name.upper()}  |  mode={mode}")
    print("=" * 70)

    if not check_file(config['corpus']):
        print(f" Skipping — corpus file missing")
        return False

    # Output directory: IPI_generators/ipi_<dataset>_<mode>
    out_dir = Path(f"IPI_generators/ipi_{config['dataset']}_{mode}")

    # Skip check: look for the main poisoned corpus file
    suffix = MODE_SUFFIX[mode]
    expected_file = out_dir / f"{config['dataset']}_{suffix}.jsonl"
    if skip_existing and expected_file.exists():
        print(f"  ✓ Already exists — skipping ({expected_file})")
        return True

    # Build command
    cmd = [
        sys.executable, config['script'],
        '--corpus',  config['corpus'],
        '--out',     str(out_dir),
        '--dataset', config['dataset'],
        '--mode',    mode,
    ]

    if not config.get('queries') or not check_file(config['queries']):
        print(f"  ✗ Queries file missing — required for semantic query selection")
        return False
    cmd += ['--queries', config['queries']]

    if config.get('domain'):
        cmd += ['--domain', config['domain']]

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Done — output in {out_dir}")
        if result.stdout:
            # Print last 800 chars of stdout for progress visibility
            print(result.stdout[-800:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error (return code {e.returncode})")
        if e.stdout:
            print(e.stdout[-500:])
        if e.stderr:
            print(e.stderr[-500:])
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch generate IPI poisoned corpora (realistic / hard / stress tiers)'
    )
    parser.add_argument(
        '--corpus',
        choices=list(CORPUS_CONFIGS.keys()) + ['all'],
        default='all',
        help='Which corpus to generate (default: all)'
    )
    parser.add_argument(
        '--mode',
        choices=['realistic', 'hard', 'stress', 'all'],
        default='all',
        help='Which tier to generate (default: all three tiers)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip corpora/modes that already have generated files'
    )

    args = parser.parse_args()

    corpora_to_run = list(CORPUS_CONFIGS.keys()) if args.corpus == 'all' else [args.corpus]
    modes_to_run   = ['realistic', 'hard', 'stress'] if args.mode == 'all' else [args.mode]

    print("=" * 70)
    print("BATCH IPI CORPUS GENERATION")
    print("=" * 70)
    print(f"  Corpora : {corpora_to_run}")
    print(f"  Modes   : {modes_to_run}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Total runs: {len(corpora_to_run) * len(modes_to_run)}")

    results = {}
    for corpus_name in corpora_to_run:
        for mode in modes_to_run:
            key = f"{corpus_name}/{mode}"
            config = CORPUS_CONFIGS[corpus_name]
            results[key] = generate_one(corpus_name, config, mode, args.skip_existing)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for v in results.values() if v)
    failed     = len(results) - successful

    for key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {key}")

    print(f"\nSuccessful: {successful}/{len(results)}")
    if failed:
        print(f"Failed    : {failed}/{len(results)}")
        sys.exit(1)
    else:
        print("All done!")


if __name__ == "__main__":
    main()
