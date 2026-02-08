#!/usr/bin/env python3
"""
Batch run Stages 6-7 for all corpora
"""

import subprocess
import sys
from pathlib import Path
import time

CORPORA = ['nfcorpus', 'fiqa', 'scifact', 'hotpotqa']
# Query counts: nfcorpus=3237, fiqa=6648, scifact=1109, hotpotqa=97852
# Total: ~108,846 queries × 10s = ~1,088,460 seconds = ~302 hours = ~12.6 days
# WARNING: This will take a VERY long time!

def run_all_corpora():
    """Run Stages 6-7 for all corpora with ALL queries"""
    print("="*80)
    print("BATCH RUN: STAGES 6-7 FOR ALL CORPORA - ALL QUERIES")
    print("="*80)
    print("⚠️  WARNING: This will process ALL queries for each corpus!")
    print()
    print("Query counts:")
    print("  - nfcorpus: ~3,237 queries")
    print("  - fiqa: ~6,648 queries")
    print("  - scifact: ~1,109 queries")
    print("  - hotpotqa: ~97,852 queries (VERY LARGE!)")
    print()
    print("Estimated time per corpus (~10s per query):")
    print("  - nfcorpus: ~9 hours")
    print("  - fiqa: ~18 hours")
    print("  - scifact: ~3 hours")
    print("  - hotpotqa: ~272 hours (~11 days!)")
    print()
    print("TOTAL ESTIMATED TIME: ~302 hours (~12.6 days)")
    print()
    print("This will run in the background. Progress will be logged.")
    print(f"Corpora: {', '.join(CORPORA)}")
    print()
    
    results = {}
    start_time = time.time()
    
    for i, corpus in enumerate(CORPORA, 1):
        print("\n" + "="*80)
        print(f"[{i}/{len(CORPORA)}] Running Stages 6-7 for {corpus.upper()}")
        print("="*80)
        
        corpus_start = time.time()
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    'evaluation/run_stages_6_7_with_generation.py',
                    '--corpus', corpus
                    # No --sample-size means use all queries
                ],
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
    total_minutes = total_time / 60
    
    print("\n" + "="*80)
    print("BATCH RUN COMPLETE")
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
    print("All results saved to: evaluation/stage_by_stage_results/")
    print("="*80)

if __name__ == "__main__":
    run_all_corpora()

