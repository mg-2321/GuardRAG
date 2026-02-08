#!/usr/bin/env python3
"""
Batch run comparative retriever evaluation on all corpora
Study A: Retriever-Level Comparison
"""

import subprocess
import sys
from pathlib import Path
import time

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
        'corpus': 'IPI_generators/ipi_scifact_aligned/scifact_ipi_query_aligned.jsonl',
        'metadata': 'IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/scifact/queries.jsonl'
    },
    'hotpotqa': {
        'corpus': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl',
        'metadata': 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl',
        'queries': 'data/corpus/beir/hotpotqa/queries.jsonl'
    },
}

def run_comparative_evaluation():
    """Run comparative retriever evaluation on all corpora"""
    print("="*80)
    print("BATCH COMPARATIVE RETRIEVER EVALUATION (STUDY A)")
    print("="*80)
    print("Comparing: BM25, Dense (E5), SPLADE, Hybrid (α=0.2, 0.5, 0.8)")
    print()
    
    results = {}
    start_time = time.time()
    
    for corpus_name, config in CORPUS_CONFIGS.items():
        print("\n" + "="*80)
        print(f"Evaluating: {corpus_name.upper()}")
        print("="*80)
        
        corpus_start = time.time()
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    'evaluation/run_comparative_retriever_evaluation.py',
                    '--corpus', config['corpus'],
                    '--metadata', config['metadata'],
                    '--queries', config['queries'],
                    '--corpus-name', corpus_name,
                    '--output-dir', 'evaluation/comparative_results',
                    # Use sample size for initial test, remove for full evaluation
                    '--sample-size', '500'  # Start with 500 queries per corpus
                ],
                cwd=Path(__file__).parent.parent,
                capture_output=False,
                text=True
            )
            
            corpus_time = time.time() - corpus_start
            corpus_minutes = corpus_time / 60
            
            if result.returncode == 0:
                print(f"\n✅ {corpus_name.upper()} completed in {corpus_minutes:.1f} minutes")
                results[corpus_name] = {'status': 'success', 'time': corpus_time}
            else:
                print(f"\n❌ {corpus_name.upper()} failed with return code {result.returncode}")
                results[corpus_name] = {'status': 'failed', 'time': corpus_time}
                
        except Exception as e:
            corpus_time = time.time() - corpus_start
            print(f"\n❌ {corpus_name.upper()} error: {e}")
            results[corpus_name] = {'status': 'error', 'error': str(e), 'time': corpus_time}
    
    total_time = time.time() - start_time
    total_minutes = total_time / 60
    
    print("\n" + "="*80)
    print("BATCH COMPARATIVE EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {total_minutes:.1f} minutes ({total_time:.0f} seconds)")
    print()
    print("Results Summary:")
    for corpus, result in results.items():
        status = result['status']
        time_taken = result.get('time', 0) / 60
        if status == 'success':
            print(f"  ✅ {corpus.upper()}: {time_taken:.1f} minutes")
        else:
            print(f"  ❌ {corpus.upper()}: {status}")
    
    print("\n" + "="*80)
    print("All results saved to: evaluation/comparative_results/")
    print("="*80)

if __name__ == "__main__":
    run_comparative_evaluation()

