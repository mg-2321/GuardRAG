#!/usr/bin/env python3
"""
Resume HotpotQA Evaluation from Progress 7430/97852
Continues from where the previous run halted
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

# Import from existing evaluation module
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.pipeline import Pipeline, PipelineConfig
from evaluation.metrics_calculator import MetricsCalculator


@dataclass
class ResumeState:
    """State to resume from"""
    corpus_name: str
    start_query_index: int  # Start from query 7430
    total_queries: int
    stage: int  # Which stage we're on (3 = retrieval)
    results: Dict = None


def resume_hotpotqa_evaluation():
    """Resume HotpotQA evaluation from query 7430"""
    
    print("="*80)
    print("RESUMING HOTPOTQA EVALUATION")
    print("="*80)
    print(f"Resuming from: Query 7430/97852")
    print(f"Last completed: Stage 3 (Retrieval Evaluation)")
    print("="*80)
    
    # Configuration
    CORPUS_NAME = 'hotpotqa'
    CORPUS_PATH = 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl'
    METADATA_PATH = 'IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl'
    QUERIES_PATH = 'data/corpus/beir/hotpotqa/queries.jsonl'
    OUTPUT_DIR = 'evaluation/stage_by_stage_results'
    RESUME_FROM_QUERY = 7430  # Start from here
    
    # Load metadata
    print("\n[SETUP] Loading metadata...")
    poisoned_doc_ids: Set[str] = set()
    metadata_by_doc_id: Dict[str, Dict] = {}
    
    with open(METADATA_PATH, 'r') as f:
        for line in f:
            if line.strip():
                meta = json.loads(line)
                doc_id = meta.get('doc_id')
                if doc_id:
                    metadata_by_doc_id[doc_id] = meta
                    if meta.get('is_poisoned', True):
                        poisoned_doc_ids.add(doc_id)
    
    print(f"✓ Loaded {len(poisoned_doc_ids)} poisoned documents")
    
    # Load queries
    print(f"\n[SETUP] Loading queries...")
    queries = []
    with open(QUERIES_PATH, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    total_queries = len(queries)
    print(f"✓ Loaded {total_queries} queries")
    
    # Initialize pipeline
    print(f"\n[SETUP] Initializing pipeline...")
    config = PipelineConfig(document_path=CORPUS_PATH, retriever='bm25', default_top_k=10)
    pipeline = Pipeline(config)
    print(f"✓ Pipeline initialized with {len(pipeline.store._documents)} documents")
    
    # RESUME STAGE 3 FROM QUERY 7430
    print(f"\n{'='*80}")
    print(f"[STAGE 3] RESUMING RETRIEVAL EVALUATION (FROM QUERY {RESUME_FROM_QUERY})")
    print(f"{'='*80}")
    
    k_values = [1, 3, 5, 10]
    
    # Initialize metrics (for resumed portion)
    poison_presence_top_k = {k: 0 for k in k_values}
    poison_doc_count_sum_top_k = {k: 0 for k in k_values}
    poison_presence_by_family_counts = defaultdict(lambda: {k: 0 for k in k_values})
    rank_of_first_poison = []
    rank_distribution_by_family = defaultdict(list)
    
    resumed_query_count = 0
    
    # Evaluate from query 7430 onwards
    for i in range(RESUME_FROM_QUERY, total_queries):
        query_obj = queries[i]
        query_idx = i + 1  # 1-indexed for display
        
        # Print progress every 10 queries
        if (query_idx - RESUME_FROM_QUERY) % 10 == 0:
            print(f"  Progress: {query_idx}/{total_queries} queries")
        
        query_id = query_obj.get('_id', str(query_idx))
        query_text = query_obj.get('text', '')
        
        if not query_text:
            continue
        
        resumed_query_count += 1
        
        try:
            # Retrieve documents
            retrieved = pipeline.retriever.retrieve(query_text, top_k=max(k_values))
            retrieved_doc_ids = [doc.doc_id for doc, _score in retrieved]
            
            # Track poison presence
            for k in k_values:
                top_k_docs = retrieved_doc_ids[:k]
                if any(doc_id in poisoned_doc_ids for doc_id in top_k_docs):
                    poison_presence_top_k[k] += 1
                
                # Count poisoned docs in top-k
                poison_count = sum(1 for doc_id in top_k_docs if doc_id in poisoned_doc_ids)
                poison_doc_count_sum_top_k[k] += poison_count
            
            # Track rank of first poisoned document
            first_poison_rank = None
            for rank, doc_id in enumerate(retrieved_doc_ids, 1):
                if doc_id in poisoned_doc_ids:
                    first_poison_rank = rank
                    rank_of_first_poison.append(rank)
                    
                    # Get attack family
                    meta = metadata_by_doc_id.get(doc_id, {})
                    family = meta.get('attack_family', 'unknown')
                    
                    if family not in poison_presence_by_family_counts:
                        poison_presence_by_family_counts[family] = {k: 0 for k in k_values}
                    
                    # Update by-family counts
                    if first_poison_rank <= max(k_values):
                        for k in k_values:
                            if first_poison_rank <= k:
                                poison_presence_by_family_counts[family][k] += 1
                    
                    rank_distribution_by_family[family].append(first_poison_rank)
                    break
                    
        except Exception as e:
            print(f"  Error processing query {query_idx}: {e}")
            continue
    
    # Compile results
    print(f"\n{'='*80}")
    print(f"[RESUMED RESULTS] Processed {resumed_query_count} additional queries")
    print(f"[RESUMED RESULTS] From query {RESUME_FROM_QUERY} to {total_queries}")
    print(f"{'='*80}")
    
    # Calculate metrics
    poison_presence_top_k_pct = {
        k: (poison_presence_top_k[k] / resumed_query_count * 100) if resumed_query_count > 0 else 0
        for k in k_values
    }
    
    poison_doc_count_avg_top_k = {
        k: (poison_doc_count_sum_top_k[k] / resumed_query_count) if resumed_query_count > 0 else 0
        for k in k_values
    }
    
    print("\n[METRICS - Resumed Portion Only]")
    print(f"Poison Presence Rate (%):")
    for k in k_values:
        print(f"  ER@{k}: {poison_presence_top_k_pct[k]:.2f}%")
    
    print(f"\nAvg Poisoned Docs per Query (resumed):")
    for k in k_values:
        print(f"  Avg@{k}: {poison_doc_count_avg_top_k[k]:.2f}")
    
    if rank_of_first_poison:
        print(f"\nRank of First Poison (resumed):")
        print(f"  Min: {min(rank_of_first_poison)}")
        print(f"  Max: {max(rank_of_first_poison)}")
        print(f"  Mean: {sum(rank_of_first_poison) / len(rank_of_first_poison):.2f}")
        print(f"  Median: {sorted(rank_of_first_poison)[len(rank_of_first_poison)//2]}")
    
    # Save results
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'corpus': CORPUS_NAME,
        'stage': 3,
        'resumed': True,
        'resume_from_query': RESUME_FROM_QUERY,
        'total_queries': total_queries,
        'resumed_query_count': resumed_query_count,
        'poison_presence_top_k_pct': poison_presence_top_k_pct,
        'poison_doc_count_avg_top_k': poison_doc_count_avg_top_k,
        'rank_of_first_poison_stats': {
            'min': min(rank_of_first_poison) if rank_of_first_poison else None,
            'max': max(rank_of_first_poison) if rank_of_first_poison else None,
            'mean': sum(rank_of_first_poison) / len(rank_of_first_poison) if rank_of_first_poison else None,
        }
    }
    
    result_file = output_dir / f"{CORPUS_NAME}_stage3_resumed.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {result_file}")
    
    # Append to log
    log_file = Path("evaluation/batch_stages_1_5_all_queries.log")
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[RESUMED] HotpotQA Stage 3 from query {RESUME_FROM_QUERY}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Resumed queries processed: {resumed_query_count}/{total_queries - RESUME_FROM_QUERY}\n")
        f.write(f"ER@1: {poison_presence_top_k_pct[1]:.2f}%\n")
        f.write(f"ER@3: {poison_presence_top_k_pct[3]:.2f}%\n")
        f.write(f"ER@5: {poison_presence_top_k_pct[5]:.2f}%\n")
        f.write(f"ER@10: {poison_presence_top_k_pct[10]:.2f}%\n")
    
    print(f"✓ Log appended to: {log_file}")
    
    print(f"\n{'='*80}")
    print("RESUMPTION COMPLETE - Continue with Stage 4 & 5")
    print(f"{'='*80}")


if __name__ == '__main__':
    resume_hotpotqa_evaluation()
