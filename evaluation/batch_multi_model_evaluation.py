#!/usr/bin/env python3
"""
Batch Multi-Model Evaluation for All Corpora
Runs comprehensive multi-model retriever evaluation on all mixed corpora.
Tests BM25, 5 Dense models, and 5 Hybrid models for each corpus.
"""

import subprocess
import sys
from pathlib import Path
import argparse
import json

# Corpus configurations
CORPUS_CONFIGS = {
    'nfcorpus': {
        'corpus': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/nfcorpus/queries.jsonl',
        'output_dir': 'evaluation/nfcorpus_multi_model',
        'query_format': 'beir'
    },
    'fiqa': {
        'corpus': 'IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/fiqa/queries.jsonl',
        'output_dir': 'evaluation/fiqa_multi_model',
        'query_format': 'beir'
    },
    'hotpotqa': {
        'corpus': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl',
        'output_dir': 'evaluation/hotpotqa_multi_model',
        'query_format': 'beir'
    },
    'scifact': {
        'corpus': 'IPI_generators/ipi_scifact/scifact_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/scifact/queries.jsonl',
        'output_dir': 'evaluation/scifact_multi_model',
        'query_format': 'beir'
    },
    'nq': {
        'corpus': 'IPI_generators/ipi_nq/nq_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/nq/queries.jsonl',
        'output_dir': 'evaluation/nq_multi_model',
        'query_format': 'beir'
    },
    'hotpotqa_standalone': {
        'corpus': 'IPI_generators/ipi_hotpotqa_standalone/hotpotqa_standalone_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/hotpotqa/hotpotqa_validation.jsonl',
        'output_dir': 'evaluation/hotpotqa_standalone_multi_model',
        'query_format': 'hotpotqa'
    },
    'msmarco': {
        'corpus': 'IPI_generators/ipi_msmarco/msmarco_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/msmarco/msmarco_validation.jsonl',
        'output_dir': 'evaluation/msmarco_multi_model',
        'query_format': 'msmarco'
    },
    'natural_questions': {
        'corpus': 'IPI_generators/ipi_natural_questions/natural_questions_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/natural_questions/nq_validation.jsonl',
        'output_dir': 'evaluation/natural_questions_multi_model',
        'query_format': 'beir'
    },
    'synthetic': {
        'corpus': 'IPI_generators/ipi_synthetic/synthetic_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/synthetic/synthetic_corpus.jsonl',
        'output_dir': 'evaluation/synthetic_multi_model',
        'query_format': 'synthetic'
    }
}


def convert_queries_to_beir_format(input_file: str, output_file: str, query_format: str):
    """Convert queries from various formats to BEIR format."""
    queries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            
            if query_format == 'beir':
                queries.append(data)
            elif query_format == 'hotpotqa':
                queries.append({
                    "_id": data.get("_id", f"q_{len(queries)}"),
                    "text": data.get("question", ""),
                    "metadata": {}
                })
            elif query_format == 'msmarco':
                queries.append({
                    "_id": data.get("_id", f"q_{len(queries)}"),
                    "text": data.get("query", ""),
                    "metadata": {}
                })
            elif query_format == 'synthetic':
                queries.append({
                    "_id": data.get("id", f"q_{len(queries)}"),
                    "text": data.get("query", ""),
                    "metadata": {}
                })
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in queries:
            f.write(json.dumps(q) + '\n')
    
    return len(queries)


def evaluate_corpus(corpus_name: str, config: dict, skip_query_conversion: bool = False):
    """Run multi-model evaluation for a single corpus."""
    print(f"\n{'='*80}")
    print(f"MULTI-MODEL EVALUATION: {corpus_name.upper()}")
    print(f"{'='*80}")
    
    corpus_path = Path(config['corpus'])
    queries_path = Path(config['queries'])
    output_dir = config['output_dir']
    
    # Check if files exist
    if not corpus_path.exists():
        print(f"⚠️  Skipping {corpus_name}: Corpus file not found: {corpus_path}")
        return False
    
    if not queries_path.exists():
        print(f"⚠️  Skipping {corpus_name}: Queries file not found: {queries_path}")
        return False
    
    # Convert queries if needed
    converted_queries = None
    if not skip_query_conversion and config['query_format'] != 'beir':
        converted_queries = Path(output_dir) / 'converted_queries.jsonl'
        print(f"Converting queries from {config['query_format']} format...")
        num_queries = convert_queries_to_beir_format(
            str(queries_path),
            str(converted_queries),
            config['query_format']
        )
        print(f"✓ Converted {num_queries} queries")
        queries_to_use = str(converted_queries)
    else:
        queries_to_use = str(queries_path)
    
    # Build command (no --sample means all queries)
    cmd = [
        sys.executable,
        'evaluation/comprehensive_multi_model_evaluation.py',
        '--corpus', str(corpus_path),
        '--queries', queries_to_use,
        '--output', output_dir
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    print("This will evaluate:")
    print("  - BM25")
    print("  - 5 Dense models (all-MiniLM-L6-v2, msmarco-MiniLM-L6-v3, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1, all-MiniLM-L12-v2)")
    print("  - 5 Hybrid models (BM25 + each Dense model)")
    print("  - On ALL queries (complete corpus)")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {corpus_name} multi-model evaluation complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {corpus_name} evaluation failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted for {corpus_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch multi-model evaluation for all corpora')
    parser.add_argument('--corpora', nargs='+', default=None,
                       help='Specific corpora to evaluate (default: all)')
    parser.add_argument('--skip-conversion', action='store_true',
                       help='Skip query format conversion')
    parser.add_argument('--list', action='store_true',
                       help='List available corpora and exit')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available corpora:")
        for name in CORPUS_CONFIGS.keys():
            print(f"  - {name}")
        return
    
    # Determine which corpora to evaluate
    if args.corpora:
        corpora_to_eval = [c for c in args.corpora if c in CORPUS_CONFIGS]
        if not corpora_to_eval:
            print(f"⚠️  No valid corpora found. Available: {list(CORPUS_CONFIGS.keys())}")
            return
    else:
        corpora_to_eval = list(CORPUS_CONFIGS.keys())
    
    print(f"\n{'='*80}")
    print(f"BATCH MULTI-MODEL EVALUATION: {len(corpora_to_eval)} CORPORA")
    print(f"{'='*80}")
    print(f"Corpora: {', '.join(corpora_to_eval)}")
    print(f"Models: BM25 + 5 Dense + 5 Hybrid = 11 retrievers per corpus")
    print(f"Queries: ALL (complete corpus)")
    print()
    print("⚠️  This will take significant time (encoding embeddings for each model)")
    print()
    
    # Evaluate each corpus
    results = {}
    for corpus_name in corpora_to_eval:
        config = CORPUS_CONFIGS[corpus_name]
        success = evaluate_corpus(
            corpus_name,
            config,
            skip_query_conversion=args.skip_conversion
        )
        results[corpus_name] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"  - {name}")
    
    print(f"\nResults saved to: evaluation/*_multi_model/")


if __name__ == "__main__":
    main()

