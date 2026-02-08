#!/usr/bin/env python3
"""
Batch Generate IPI Attacks for All Corpora
Generates poisoned and mixed corpora for:
- BEIR datasets (nfcorpus, fiqa, hotpotqa, scifact, nq)
- HotpotQA
- MS MARCO
- Natural Questions
- Synthetic corpus
"""

import subprocess
import sys
from pathlib import Path
import argparse

# Corpus configurations
CORPUS_CONFIGS = {
    'beir_nfcorpus': {
        'corpus': 'data/corpus/beir/nfcorpus/corpus.jsonl',
        'queries': 'data/corpus/beir/nfcorpus/queries.jsonl',
        'output_dir': 'IPI_generators/ipi_nfcorpus',
        'dataset': 'nfcorpus',
        'num_attacks': 3000  # Increased for better statistical significance
    },
    'beir_fiqa': {
        'corpus': 'data/corpus/beir/fiqa/corpus.jsonl',
        'queries': 'data/corpus/beir/fiqa/queries.jsonl',
        'output_dir': 'IPI_generators/ipi_fiqa',
        'dataset': 'fiqa',
        'num_attacks': 5000
    },
    'beir_hotpotqa': {
        'corpus': 'data/corpus/beir/hotpotqa/corpus.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl',
        'output_dir': 'IPI_generators/ipi_hotpotqa',
        'dataset': 'hotpotqa',
        'num_attacks': 3000
    },
    'beir_scifact': {
        'corpus': 'data/corpus/beir/scifact/corpus.jsonl',
        'queries': 'data/corpus/beir/scifact/queries.jsonl',
        'output_dir': 'IPI_generators/ipi_scifact',
        'dataset': 'scifact',
        'num_attacks': 3000
    },
    'beir_nq': {
        'corpus': 'data/corpus/beir/nq/corpus.jsonl',
        'queries': 'data/corpus/beir/nq/queries.jsonl',
        'output_dir': 'IPI_generators/ipi_nq',
        'dataset': 'nq',
        'num_attacks': 3000
    },
    'hotpotqa_standalone': {
        'corpus': 'data/corpus/hotpotqa/hotpotqa_validation.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl',  # Use BEIR queries if available
        'output_dir': 'IPI_generators/ipi_hotpotqa_standalone',
        'dataset': 'hotpotqa_standalone',
        'num_attacks': 3000
    },
    'msmarco': {
        'corpus': 'data/corpus/msmarco/msmarco_validation.jsonl',
        'queries': None,  # No queries file - generator will work without it
        'output_dir': 'IPI_generators/ipi_msmarco',
        'dataset': 'msmarco',
        'num_attacks': 3000
    },
    'natural_questions': {
        'corpus': 'data/corpus/beir/nq/corpus.jsonl',  # Use BEIR NQ corpus instead of empty validation file
        'queries': 'data/corpus/beir/nq/queries.jsonl',
        'output_dir': 'IPI_generators/ipi_natural_questions',
        'dataset': 'natural_questions',
        'num_attacks': 3000
    },
    'synthetic': {
        'corpus': 'data/corpus/synthetic/synthetic_corpus.jsonl',
        'queries': None,  # No queries file - generator will work without it
        'output_dir': 'IPI_generators/ipi_synthetic',
        'dataset': 'synthetic',
        'num_attacks': 3000
    }
}


def check_file_exists(filepath):
    """Check if file exists"""
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️  Warning: {filepath} does not exist")
        return False
    return True


def generate_for_corpus(corpus_name, config, skip_existing=False):
    """Generate IPI attacks for a single corpus"""
    print("\n" + "="*80)
    print(f"GENERATING IPI ATTACKS FOR: {corpus_name.upper()}")
    print("="*80)
    
    # Check if corpus file exists
    if not check_file_exists(config['corpus']):
        print(f"✗ Skipping {corpus_name} - corpus file not found")
        return False
    
    # Check if output already exists
    output_dir = Path(config['output_dir'])
    poisoned_file = output_dir / f"{config['dataset']}_ipi_poisoned_v2.jsonl"
    if skip_existing and poisoned_file.exists():
        print(f"✓ {corpus_name} already exists, skipping...")
        return True
    
    # Build command
    cmd = [
        sys.executable,
        'IPI_generators/fixed_advanced_ipi_nfcorpus.py',
        '--corpus', config['corpus'],
        '--output-dir', config['output_dir'],
        '--dataset', config['dataset'],
        '--num-attacks', str(config['num_attacks'])
    ]
    
    # Add queries if available
    if config['queries'] and check_file_exists(config['queries']):
        cmd.extend(['--queries', config['queries']])
    else:
        print(f"⚠️  No queries file for {corpus_name}, generating without query alignment")
        # Generator can work without queries - it will use document titles as fallback
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully generated IPI attacks for {corpus_name}")
        if result.stdout:
            print(result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating {corpus_name}:")
        print(f"  Return code: {e.returncode}")
        if e.stdout:
            print(f"  stdout: {e.stdout[-500:]}")
        if e.stderr:
            print(f"  stderr: {e.stderr[-500:]}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch generate IPI attacks for all corpora')
    parser.add_argument('--corpus', choices=list(CORPUS_CONFIGS.keys()) + ['all'],
                       default='all', help='Which corpus to generate (default: all)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip corpora that already have generated files')
    parser.add_argument('--num-attacks', type=int, default=3000,
                       help='Number of attacks per corpus (default: 3000)')
    
    args = parser.parse_args()
    
    # Update num_attacks if specified
    if args.num_attacks != 1000:
        for config in CORPUS_CONFIGS.values():
            config['num_attacks'] = args.num_attacks
    
    print("="*80)
    print("BATCH IPI ATTACK GENERATION FOR ALL CORPORA")
    print("="*80)
    print(f"Total corpora: {len(CORPUS_CONFIGS)}")
    print(f"Attacks per corpus: {args.num_attacks}")
    print(f"Skip existing: {args.skip_existing}")
    
    if args.corpus == 'all':
        corpora_to_process = list(CORPUS_CONFIGS.keys())
    else:
        corpora_to_process = [args.corpus]
    
    results = {}
    for corpus_name in corpora_to_process:
        config = CORPUS_CONFIGS[corpus_name]
        success = generate_for_corpus(corpus_name, config, args.skip_existing)
        results[corpus_name] = success
    
    # Summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    for corpus_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {corpus_name}")
    
    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\n⚠️  Some corpora failed to generate. Check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All corpora generated successfully!")


if __name__ == "__main__":
    main()

