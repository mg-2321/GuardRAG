#!/usr/bin/env python3
"""
Batch run stage-by-stage evaluation on all corpora
"""

import subprocess
import sys
from pathlib import Path

CORPUS_CONFIGS = {
    'nfcorpus': {
        'corpus': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/nfcorpus/queries.jsonl'
    },
    'fiqa': {
        'corpus': 'IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/fiqa/queries.jsonl'
    },
    'scifact': {
        'corpus': 'IPI_generators/ipi_scifact/scifact_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/scifact/queries.jsonl'
    },
    'hotpotqa': {
        'corpus': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl'
    },
    'msmarco': {
        'corpus': 'IPI_generators/ipi_msmarco/msmarco_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_msmarco/msmarco_ipi_metadata_v2.jsonl',
        'queries': None  # No queries file
    }
}

RETRIEVERS = ['bm25']  # Add more: 'dense', 'hybrid' as needed

def main():
    print("="*80)
    print("BATCH STAGE-BY-STAGE EVALUATION")
    print("="*80)
    
    for corpus_name, config in CORPUS_CONFIGS.items():
        if not config['queries']:
            print(f"\nSkipping {corpus_name} - no queries file")
            continue
        
        for retriever in RETRIEVERS:
            print(f"\n{'='*80}")
            print(f"Evaluating: {corpus_name} with {retriever}")
            print(f"{'='*80}")
            
            cmd = [
                sys.executable,
                'evaluation/stage_by_stage_evaluation.py',
                '--corpus', config['corpus'],
                '--metadata', config['metadata'],
                '--queries', config['queries'],
                '--corpus-name', corpus_name,
                '--retriever', retriever,
                '--output-dir', 'evaluation/stage_by_stage_results',
                # No --sample-size means use ALL queries
                '--skip-generation',  # Skip slow generation stages
                '--no-generation'  # Don't load model
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error evaluating {corpus_name} with {retriever}: {e}")
    
    print("\n" + "="*80)
    print("BATCH EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

