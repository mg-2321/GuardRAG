#!/usr/bin/env python3
"""
Batch run Stages 6-7 for all corpora with Llama 3.1 70B
"""

import subprocess
import sys
from pathlib import Path
import time
import argparse

CORPORA = ['scifact', 'nfcorpus', 'fiqa', 'hotpotqa']

# Estimated times with Llama 70B (3x slower than 8B)
ESTIMATED_TIMES = {
    'scifact': 1109,    # queries
    'nfcorpus': 3237,
    'fiqa': 6648,
    'hotpotqa': 97852
}

def run_all_corpora(model_key='llama-3.1-70b', retriever='bm25', sample_size=None):
    """Run Stages 6-7 for all corpora with Llama 70B"""
    print("="*80)
    print(f"BATCH RUN: STAGES 6-7 WITH {model_key.upper()}")
    print("="*80)
    print(f"Retriever: {retriever}")
    print()
    
    if sample_size:
        print(f"⚠️  Sample mode: {sample_size} queries per corpus")
        total_queries = sample_size * len(CORPORA)
    else:
        print("⚠️  WARNING: Processing ALL queries for each corpus!")
        total_queries = sum(ESTIMATED_TIMES.values())
    
    print()
    print("Query counts:")
    for corpus in CORPORA:
        count = ESTIMATED_TIMES[corpus] if not sample_size else min(sample_size, ESTIMATED_TIMES[corpus])
        time_est = count * 30 / 3600  # 30s per query with 70B
        print(f"  - {corpus}: {count:,} queries (~{time_est:.1f} hours)")
    
    total_time_est = total_queries * 30 / 3600
    print()
    print(f"Total estimated time: ~{total_time_est:.1f} hours ({total_time_est/24:.1f} days)")
    print()
    
    if not sample_size:
        confirm = input("This will take a VERY long time. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
    
    print(f"Corpora order: {', '.join(CORPORA)}")
    print()
    
    results = {}
    start_time = time.time()
    
    for i, corpus in enumerate(CORPORA, 1):
        print("\n" + "="*80)
        print(f"[{i}/{len(CORPORA)}] Running Stages 6-7 for {corpus.upper()}")
        print("="*80)
        
        corpus_start = time.time()
        
        try:
            cmd = [
                sys.executable,
                'evaluation/run_stages_6_7_llama70b.py',
                '--corpus', corpus,
                '--model', model_key,
                '--retriever', retriever
            ]
            
            if sample_size:
                cmd.extend(['--sample-size', str(sample_size)])
            
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            corpus_time = time.time() - corpus_start
            corpus_minutes = corpus_time / 60
            
            if result.returncode == 0:
                print(f"\n✅ {corpus.upper()} completed in {corpus_minutes:.1f} minutes")
                results[corpus] = {'status': 'success', 'time': corpus_time}
            else:
                print(f"\n❌ {corpus.upper()} failed with return code {result.returncode}")
                results[corpus] = {'status': 'failed', 'time': corpus_time}
                
        except Exception as e:
            corpus_time = time.time() - corpus_start
            print(f"\n❌ {corpus.upper()} error: {e}")
            results[corpus] = {'status': 'error', 'error': str(e), 'time': corpus_time}
    
    total_time = time.time() - start_time
    total_hours = total_time / 3600
    
    print("\n" + "="*80)
    print("BATCH RUN COMPLETE")
    print("="*80)
    print(f"Total time: {total_hours:.1f} hours ({total_time/60:.1f} minutes)")
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
    print("All results saved to: evaluation/stage_by_stage_results/")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch run Stages 6-7 with Llama 70B')
    parser.add_argument('--model', type=str, default='llama-3.1-70b',
                      choices=['llama-3.1-70b', 'llama-3.1-70b-4bit'],
                      help='Model configuration')
    parser.add_argument('--retriever', type=str, default='bm25',
                      choices=['bm25', 'dense', 'hybrid', 'splade'],
                      help='Retriever type')
    parser.add_argument('--sample-size', type=int, default=None,
                      help='Sample size per corpus for testing')
    
    args = parser.parse_args()
    
    run_all_corpora(
        model_key=args.model,
        retriever=args.retriever,
        sample_size=args.sample_size
    )
