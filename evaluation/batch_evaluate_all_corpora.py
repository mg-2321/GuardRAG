#!/usr/bin/env python3
"""
Batch Evaluation Script for All Corpora
Runs comprehensive retriever evaluation on all poisoned corpora.

Supports:
- BEIR datasets (nfcorpus, fiqa, hotpotqa, scifact, nq)
- HotpotQA standalone
- MS MARCO
- Natural Questions
- Synthetic corpus
"""

import subprocess
import sys
from pathlib import Path
import argparse
import json

# Corpus configurations with query file paths
CORPUS_CONFIGS = {
    'nfcorpus': {
        'corpus': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/nfcorpus/queries.jsonl',
        'metadata': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/nfcorpus_comprehensive_v2',
        'query_format': 'beir'  # {"_id": "...", "text": "..."}
    },
    'fiqa': {
        'corpus': 'IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/fiqa/queries.jsonl',
        'metadata': 'IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/fiqa_comprehensive_v2',
        'query_format': 'beir'
    },
    'hotpotqa': {
        'corpus': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl',
        'metadata': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/hotpotqa_comprehensive_v2',
        'query_format': 'beir'
    },
    'scifact': {
        'corpus': 'IPI_generators/ipi_scifact/scifact_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/scifact/queries.jsonl',
        'metadata': 'IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/scifact_comprehensive_v2',
        'query_format': 'beir'
    },
    'nq': {
        'corpus': 'IPI_generators/ipi_nq/nq_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/beir/nq/queries.jsonl',
        'metadata': 'IPI_generators/ipi_nq/nq_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/nq_comprehensive_v2',
        'query_format': 'beir'
    },
    'hotpotqa_standalone': {
        'corpus': 'IPI_generators/ipi_hotpotqa_standalone/hotpotqa_standalone_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/hotpotqa/hotpotqa_validation.jsonl',
        'metadata': 'IPI_generators/ipi_hotpotqa_standalone/hotpotqa_standalone_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/hotpotqa_standalone_comprehensive_v2',
        'query_format': 'hotpotqa'  # {"question": "...", ...}
    },
    'msmarco': {
        'corpus': 'IPI_generators/ipi_msmarco/msmarco_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/msmarco/msmarco_validation.jsonl',
        'metadata': 'IPI_generators/ipi_msmarco/msmarco_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/msmarco_comprehensive_v2',
        'query_format': 'msmarco'  # {"query": "...", ...}
    },
    'natural_questions': {
        'corpus': 'IPI_generators/ipi_natural_questions/natural_questions_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/natural_questions/nq_validation.jsonl',
        'metadata': 'IPI_generators/ipi_natural_questions/natural_questions_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/natural_questions_comprehensive_v2',
        'query_format': 'beir'  # Assuming similar format
    },
    'synthetic': {
        'corpus': 'IPI_generators/ipi_synthetic/synthetic_ipi_mixed_v2.jsonl',
        'queries': 'data/corpus/synthetic/synthetic_corpus.jsonl',
        'metadata': 'IPI_generators/ipi_synthetic/synthetic_ipi_metadata_v2.jsonl',
        'output_dir': 'evaluation/synthetic_comprehensive_v2',
        'query_format': 'synthetic'  # {"id": "...", "query": "...", ...}
    }
}


def convert_queries_to_beir_format(input_file: str, output_file: str, query_format: str):
    """Convert queries from various formats to BEIR format for evaluation scripts."""
    queries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            
            if query_format == 'beir':
                # Already in BEIR format
                queries.append(data)
            elif query_format == 'hotpotqa':
                # Convert {"question": "...", ...} to {"_id": "...", "text": "..."}
                queries.append({
                    "_id": data.get("_id", f"q_{len(queries)}"),
                    "text": data.get("question", ""),
                    "metadata": {}
                })
            elif query_format == 'msmarco':
                # Convert {"query": "...", ...} to {"_id": "...", "text": "..."}
                queries.append({
                    "_id": data.get("_id", f"q_{len(queries)}"),
                    "text": data.get("query", ""),
                    "metadata": {}
                })
            elif query_format == 'synthetic':
                # Convert {"id": "...", "query": "...", ...} to {"_id": "...", "text": "..."}
                queries.append({
                    "_id": data.get("id", f"q_{len(queries)}"),
                    "text": data.get("query", ""),
                    "metadata": {}
                })
    
    # Write converted queries
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in queries:
            f.write(json.dumps(q) + '\n')
    
    return len(queries)


def evaluate_corpus(corpus_name: str, config: dict, sample_size: int = None, dense_model: str = 'all-MiniLM-L6-v2', skip_query_conversion: bool = False):
    """Run comprehensive evaluation for a single corpus."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {corpus_name.upper()}")
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
    
    # Build command
    cmd = [
        sys.executable,
        'evaluation/comprehensive_retriever_evaluation.py',
        '--corpus', str(corpus_path),
        '--queries', queries_to_use,
        '--output', output_dir,
        '--dense-model', dense_model
    ]
    
    if sample_size:
        cmd.extend(['--sample', str(sample_size)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {corpus_name} evaluation complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {corpus_name} evaluation failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted for {corpus_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate all corpora')
    parser.add_argument('--corpora', nargs='+', default=None,
                       help='Specific corpora to evaluate (default: all)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size per corpus (default: all queries)')
    parser.add_argument('--dense-model', type=str, default='all-MiniLM-L6-v2',
                       help='SentenceTransformer model for dense retrieval')
    parser.add_argument('--skip-conversion', action='store_true',
                       help='Skip query format conversion (assumes BEIR format)')
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
    print(f"BATCH EVALUATION: {len(corpora_to_eval)} CORPORA")
    print(f"{'='*80}")
    print(f"Corpora: {', '.join(corpora_to_eval)}")
    if args.sample:
        print(f"Sample size: {args.sample} queries per corpus")
    print(f"Dense model: {args.dense_model}")
    print()
    
    # Evaluate each corpus
    results = {}
    for corpus_name in corpora_to_eval:
        config = CORPUS_CONFIGS[corpus_name]
        success = evaluate_corpus(
            corpus_name,
            config,
            sample_size=args.sample,
            dense_model=args.dense_model,
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
    
    print(f"\nResults saved to: evaluation/*_comprehensive_v2/")


if __name__ == "__main__":
    main()

