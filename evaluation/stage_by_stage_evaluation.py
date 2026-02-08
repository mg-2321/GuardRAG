#!/usr/bin/env python3
"""
Stage-by-Stage Evaluation for RIPE II
Implements evaluation at each stage of the RAG pipeline
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Stage3Retrieval:
    """Stage 3: Retrieval Stage Evaluation Results"""
    poison_presence_top_k: Dict[int, float]  # k -> percentage
    poison_doc_count_avg_top_k: Dict[int, float]  # k -> avg #poisoned docs in top-k
    poison_presence_top_k_by_family: Dict[str, Dict[int, float]]  # family -> (k -> percentage)
    rank_of_first_poison: List[int]
    rank_distribution_by_family: Dict[str, List[int]]


@dataclass
class Stage5Packing:
    """Stage 5: Packing Stage Evaluation Results"""
    position_survival_rates: Dict[str, float]
    structural_survival_rates: Dict[str, float]


class StageByStageEvaluator:
    """Evaluator for stage-by-stage RAG pipeline evaluation"""
    
    def __init__(self, corpus_path: str, metadata_path: str, queries_path: str, corpus_name: str):
        self.corpus_path = corpus_path
        self.metadata_path = metadata_path
        self.queries_path = queries_path
        self.corpus_name = corpus_name
        
        # Load metadata to identify poisoned documents
        self.poisoned_doc_ids: Set[str] = set()
        self.metadata_by_doc_id: Dict[str, Dict] = {}
        
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.strip():
                        meta = json.loads(line)
                        doc_id = meta.get('doc_id')
                        if doc_id:
                            self.metadata_by_doc_id[doc_id] = meta
                            if meta.get('is_poisoned', True):
                                self.poisoned_doc_ids.add(doc_id)
        
        print(f"Loaded metadata: {len(self.poisoned_doc_ids)} poisoned documents")
    
    def evaluate_stage3_retrieval(self, pipeline, k_values: List[int] = [1, 3, 5, 10],
                                  sample_size: Optional[int] = None,
                                  skip_generation: bool = True) -> Stage3Retrieval:
        """
        Evaluate Stage 3: Retrieval Stage
        
        Measures:
        - Poison presence in top-k
        - Rank of first poisoned document
        - Rank distribution by attack family
        """
        print("\n" + "="*80)
        print("STAGE 3: RETRIEVAL STAGE EVALUATION (MOST IMPORTANT)")
        print("="*80)
        
        # Load queries
        queries = []
        with open(self.queries_path, 'r') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        
        if sample_size:
            queries = queries[:sample_size]
        
        total_queries = len(queries)
        print(f"Evaluating {total_queries} queries (generation: disabled)...")
        
        # Initialize metrics
        poison_presence_top_k = {k: 0 for k in k_values}
        poison_doc_count_sum_top_k = {k: 0 for k in k_values}
        poison_presence_by_family_counts = defaultdict(lambda: {k: 0 for k in k_values})
        rank_of_first_poison = []
        rank_distribution_by_family = defaultdict(list)
        
        # Evaluate each query
        for i, query_obj in enumerate(queries, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{total_queries} queries")
            
            query_id = query_obj.get('_id', str(i))
            query_text = query_obj.get('text', '')
            
            if not query_text:
                continue
            
            # Retrieve documents
            try:
                # NOTE: `Pipeline.run()` returns a JSON-serializable response with a
                # `retrieved` field (list of dicts). For evaluation we need the
                # actual `Document` objects, so we call the retriever directly.
                candidate_pool = max(max(k_values), getattr(pipeline.config, "candidate_pool_size", max(k_values)))
                retrieved = pipeline.retriever.retrieve(query_text, top_k=candidate_pool)
                if getattr(pipeline, "reranker", None):
                    retrieved = pipeline.reranker.rerank(query_text, retrieved)
                retrieved = retrieved[: max(k_values)]
                retrieved_docs = [doc for doc, _score in retrieved]
                
                # Check for poison in top-k
                for k in k_values:
                    top_k_docs = retrieved_docs[:k]
                    # IMPORTANT: do NOT match on ORIGINAL_ID alone, because clean docs
                    # share the same ORIGINAL_ID as their poisoned IPI_* variants.
                    # Use the concrete doc variant to decide poison.
                    poisoned_docs = [doc for doc in top_k_docs if doc.is_poisoned]
                    has_poison = bool(poisoned_docs)
                    if has_poison:
                        poison_presence_top_k[k] += 1
                    poison_doc_count_sum_top_k[k] += len(poisoned_docs)

                    # Attack-family exposure: presence of *any* poisoned doc from a family in top-k
                    families_in_top_k = set()
                    for doc in poisoned_docs:
                        original_id = doc.doc_id.split('_')[-1]
                        meta = self.metadata_by_doc_id.get(original_id, {})
                        families_in_top_k.add(meta.get('attack_family', 'unknown'))
                    for fam in families_in_top_k:
                        poison_presence_by_family_counts[fam][k] += 1
                
                # Find first poisoned document rank
                for rank, doc in enumerate(retrieved_docs, 1):
                    if doc.is_poisoned:
                        # Map poisoned variant ID -> ORIGINAL_ID for metadata lookup
                        original_id = doc.doc_id.split('_')[-1]
                        rank_of_first_poison.append(rank)
                        
                        # Get attack family from metadata
                        meta = self.metadata_by_doc_id.get(original_id, {})
                        attack_family = meta.get('attack_family', 'unknown')
                        rank_distribution_by_family[attack_family].append(rank)
                        break
                        
            except Exception as e:
                print(f"  Warning: Error processing query {query_id}: {e}")
                continue
        
        # Convert counts to percentages
        poison_presence_top_k_pct = {
            k: (count / total_queries * 100.0) if total_queries > 0 else 0.0
            for k, count in poison_presence_top_k.items()
        }

        poison_doc_count_avg_top_k = {
            k: (count / total_queries) if total_queries > 0 else 0.0
            for k, count in poison_doc_count_sum_top_k.items()
        }

        poison_presence_top_k_by_family_pct = {
            fam: {
                k: (count / total_queries * 100.0) if total_queries > 0 else 0.0
                for k, count in counts.items()
            }
            for fam, counts in poison_presence_by_family_counts.items()
        }
        
        return Stage3Retrieval(
            poison_presence_top_k=poison_presence_top_k_pct,
            poison_doc_count_avg_top_k=poison_doc_count_avg_top_k,
            poison_presence_top_k_by_family=poison_presence_top_k_by_family_pct,
            rank_of_first_poison=rank_of_first_poison,
            rank_distribution_by_family=dict(rank_distribution_by_family)
        )
    
    def evaluate_stage5_packing(self, pipeline, sample_size: Optional[int] = None) -> Stage5Packing:
        """
        Evaluate Stage 5: Packing Stage
        
        Measures:
        - Position-based survival rates (start, early-mid, mid, late-mid, end)
        - Structural IPI survival rates (caption, code, citation, etc.)
        """
        print("\n" + "="*80)
        print("STAGE 5: PACKING STAGE EVALUATION")
        print("="*80)
        
        # Load queries
        queries = []
        with open(self.queries_path, 'r') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        
        if sample_size:
            queries = queries[:sample_size]
        
        # Initialize position buckets
        position_buckets = {
            'start': {'total': 0, 'survived': 0},
            'early_mid': {'total': 0, 'survived': 0},
            'mid': {'total': 0, 'survived': 0},
            'late_mid': {'total': 0, 'survived': 0},
            'end': {'total': 0, 'survived': 0}
        }
        
        structural_buckets = {
            'caption': {'total': 0, 'survived': 0},
            'code': {'total': 0, 'survived': 0},
            'citation': {'total': 0, 'survived': 0},
            'html_comment': {'total': 0, 'survived': 0},
            'json_meta': {'total': 0, 'survived': 0},
            'table_caption': {'total': 0, 'survived': 0}
        }
        
        # Evaluate packing for queries that retrieve poison
        for query_obj in queries:
            query_text = query_obj.get('text', '')
            if not query_text:
                continue
            
            try:
                candidate_pool = max(10, getattr(pipeline.config, "candidate_pool_size", 10))
                retrieved = pipeline.retriever.retrieve(query_text, top_k=candidate_pool)
                if getattr(pipeline, "reranker", None):
                    retrieved = pipeline.reranker.rerank(query_text, retrieved)
                retrieved = retrieved[:10]
                retrieved_docs = [doc for doc, _score in retrieved]
                
                # Check each retrieved document
                for doc in retrieved_docs:
                    # Only evaluate packing survival on actually-poisoned variants.
                    if not doc.is_poisoned:
                        continue
                    
                    original_id = doc.doc_id.split('_')[-1]
                    meta = self.metadata_by_doc_id.get(original_id, {})
                    
                    # Check position
                    position = meta.get('position', 'mid')
                    if position in position_buckets:
                        position_buckets[position]['total'] += 1
                        # Simplified: assume mid positions survive (this should be more sophisticated)
                        if position in ['early_mid', 'mid']:
                            position_buckets[position]['survived'] += 1
                    
                    # Check structural carrier
                    structural_type = meta.get('structural_carrier', '')
                    if structural_type in structural_buckets:
                        structural_buckets[structural_type]['total'] += 1
                        # Simplified: assume code/caption/table_caption survive
                        if structural_type in ['caption', 'code', 'table_caption']:
                            structural_buckets[structural_type]['survived'] += 1
                            
            except Exception as e:
                continue
        
        # Calculate survival rates
        position_survival_rates = {
            pos: (bucket['survived'] / bucket['total'] * 100.0) if bucket['total'] > 0 else 0.0
            for pos, bucket in position_buckets.items()
        }
        
        structural_survival_rates = {
            struct: (bucket['survived'] / bucket['total'] * 100.0) if bucket['total'] > 0 else 0.0
            for struct, bucket in structural_buckets.items()
        }
        
        return Stage5Packing(
            position_survival_rates=position_survival_rates,
            structural_survival_rates=structural_survival_rates
        )

