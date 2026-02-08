#!/usr/bin/env python3
"""
Unified Evaluation Runner for GuardRAG
Handles all evaluation modes: comparative, stage-by-stage, and generation evaluation
Supports all corpora and models (8B and 70B)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Corpus configurations
CORPUS_CONFIGS = {
    'nfcorpus': {
        'corpus': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/nfcorpus/queries.jsonl',
        'query_count': 3237
    },
    'fiqa': {
        'corpus': 'IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/fiqa/queries.jsonl',
        'query_count': 6648
    },
    'scifact': {
        'corpus': 'IPI_generators/ipi_scifact/scifact_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/scifact/queries.jsonl',
        'query_count': 1109
    },
    'hotpotqa': {
        'corpus': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl',
        'query_count': 97852
    },
    'natural_questions': {
        'corpus': 'IPI_generators/ipi_natural_questions/natural_questions_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_natural_questions/natural_questions_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/nq/queries.jsonl',
        'query_count': 3237
    },
}

CORPORA = list(CORPUS_CONFIGS.keys())


def run_comparative_evaluation(corpus: str, sample_size: Optional[int] = None):
    """Run comparative retriever evaluation"""
    config = CORPUS_CONFIGS[corpus]
    
    print(f"\n{'='*80}")
    print(f"COMPARATIVE RETRIEVER EVALUATION: {corpus.upper()}")
    print(f"{'='*80}")
    print("Comparing: BM25, Dense (E5), SPLADE, Hybrid (α=0.2, 0.5, 0.8)")
    
    cmd = [
        sys.executable,
        'evaluation/run_comparative_retriever_evaluation.py',
        '--corpus', config['corpus'],
        '--metadata', config['metadata'],
        '--queries', config['queries'],
        '--corpus-name', corpus,
        '--output-dir', 'evaluation/comparative_results'
    ]
    
    if sample_size:
        cmd.extend(['--sample-size', str(sample_size)])
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_stage_evaluation(corpus: str, retriever: str = 'bm25', sample_size: Optional[int] = None):
    """Run stage-by-stage evaluation"""
    config = CORPUS_CONFIGS[corpus]
    
    print(f"\n{'='*80}")
    print(f"STAGE-BY-STAGE EVALUATION: {corpus.upper()} ({retriever})")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        'evaluation/stage_by_stage_evaluation.py',
        '--corpus', config['corpus'],
        '--metadata', config['metadata'],
        '--queries', config['queries'],
        '--corpus-name', corpus,
        '--retriever', retriever,
        '--output-dir', 'evaluation/stage_by_stage_results'
    ]
    
    if sample_size:
        cmd.extend(['--sample-size', str(sample_size)])
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_generation_evaluation(corpus: str, model: str = 'llama-3.1-8b', 
                             sample_size: Optional[int] = None):
    """Run generation evaluation (Stages 6-7)"""
    config = CORPUS_CONFIGS[corpus]
    
    print(f"\n{'='*80}")
    print(f"GENERATION EVALUATION: {corpus.upper()} ({model})")
    print(f"{'='*80}")
    
    if '70b' in model.lower():
        script = 'evaluation/run_stages_6_7_llama70b.py'
    else:
        script = 'evaluation/run_stages_6_7_with_generation.py'
    
    cmd = [sys.executable, script, '--corpus', corpus]
    
    if sample_size:
        cmd.extend(['--sample-size', str(sample_size)])
    
    if '70b' in model.lower():
        cmd.extend(['--model', model])
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_all_comparative(sample_size: Optional[int] = None):
    """Run comparative evaluation on all corpora"""
    print("\n" + "="*80)
    print("BATCH COMPARATIVE EVALUATION - ALL CORPORA")
    print("="*80)
    
    results = {}
    start_time = time.time()
    
    for i, corpus in enumerate(CORPORA, 1):
        print(f"\n[{i}/{len(CORPORA)}] Processing {corpus}...")
        try:
            success = run_comparative_evaluation(corpus, sample_size)
            results[corpus] = 'success' if success else 'failed'
        except Exception as e:
            print(f"Error: {e}")
            results[corpus] = 'error'
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE - {elapsed/60:.1f} minutes")
    print("="*80)
    for corpus, status in results.items():
        symbol = '✅' if status == 'success' else '❌'
        print(f"{symbol} {corpus}: {status}")


def run_all_stages(retriever: str = 'bm25', sample_size: Optional[int] = None):
    """Run stage evaluation on all corpora"""
    print("\n" + "="*80)
    print(f"BATCH STAGE EVALUATION - ALL CORPORA ({retriever})")
    print("="*80)
    
    results = {}
    start_time = time.time()
    
    for i, corpus in enumerate(CORPORA, 1):
        print(f"\n[{i}/{len(CORPORA)}] Processing {corpus}...")
        try:
            success = run_stage_evaluation(corpus, retriever, sample_size)
            results[corpus] = 'success' if success else 'failed'
        except Exception as e:
            print(f"Error: {e}")
            results[corpus] = 'error'
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE - {elapsed/60:.1f} minutes")
    print("="*80)
    for corpus, status in results.items():
        symbol = '✅' if status == 'success' else '❌'
        print(f"{symbol} {corpus}: {status}")


def run_all_generation(model: str = 'llama-3.1-8b', sample_size: Optional[int] = None):
    """Run generation evaluation on all corpora"""
    print("\n" + "="*80)
    print(f"BATCH GENERATION EVALUATION - ALL CORPORA ({model})")
    print("="*80)
    
    if not sample_size and model == 'llama-3.1-8b':
        # Warn about large scale evaluation
        print("\n⚠️  WARNING: Running FULL evaluation (all queries)")
        print("This will take a very long time!")
        print("Estimated time per corpus: 3-72 hours depending on query count")
        confirm = input("\nContinue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
    
    results = {}
    start_time = time.time()
    
    for i, corpus in enumerate(CORPORA, 1):
        print(f"\n[{i}/{len(CORPORA)}] Processing {corpus}...")
        try:
            success = run_generation_evaluation(corpus, model, sample_size)
            results[corpus] = 'success' if success else 'failed'
        except Exception as e:
            print(f"Error: {e}")
            results[corpus] = 'error'
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE - {elapsed/60:.1f} minutes")
    print("="*80)
    for corpus, status in results.items():
        symbol = '✅' if status == 'success' else '❌'
        print(f"{symbol} {corpus}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified evaluation runner for GuardRAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Comparative evaluation on single corpus
  python evaluation/run_evaluation.py --mode comparative --corpus hotpotqa --sample 100

  # All corpora stage-by-stage evaluation
  python evaluation/run_evaluation.py --mode stages --all

  # Generation evaluation with sample
  python evaluation/run_evaluation.py --mode generation --corpus fiqa --sample 50

  # Full generation evaluation with 70B model
  python evaluation/run_evaluation.py --mode generation --all --model llama-3.1-70b
        """
    )
    
    parser.add_argument('--mode', required=True,
                       choices=['comparative', 'stages', 'generation'],
                       help='Evaluation mode')
    
    parser.add_argument('--corpus', choices=CORPORA,
                       help='Single corpus to evaluate')
    
    parser.add_argument('--all', action='store_true',
                       help='Run on all corpora (batch mode)')
    
    parser.add_argument('--sample', type=int,
                       help='Sample size per corpus (for testing)')
    
    parser.add_argument('--retriever', choices=['bm25', 'dense', 'hybrid', 'splade'],
                       default='bm25',
                       help='Retriever type for stage evaluation')
    
    parser.add_argument('--model', choices=['llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-70b-4bit'],
                       default='llama-3.1-8b',
                       help='Model for generation evaluation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.all and args.corpus:
        print("Error: Cannot use --all and --corpus together")
        sys.exit(1)
    
    if not args.all and not args.corpus:
        print("Error: Must specify either --corpus or --all")
        sys.exit(1)
    
    # Run evaluation
    try:
        if args.mode == 'comparative':
            if args.all:
                run_all_comparative(args.sample)
            else:
                run_comparative_evaluation(args.corpus, args.sample)
        
        elif args.mode == 'stages':
            if args.all:
                run_all_stages(args.retriever, args.sample)
            else:
                run_stage_evaluation(args.corpus, args.retriever, args.sample)
        
        elif args.mode == 'generation':
            if args.all:
                run_all_generation(args.model, args.sample)
            else:
                run_generation_evaluation(args.corpus, args.model, args.sample)
        
        print("\n✅ Evaluation complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
