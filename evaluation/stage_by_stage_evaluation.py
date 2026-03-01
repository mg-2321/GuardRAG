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
    mrr: float  # Mean Reciprocal Rank of first poisoned doc (0 if not retrieved)
    total_queries: int


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

        for query_obj in queries:
            
            query_id = query_obj.get('_id', '')
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
                        meta = self.metadata_by_doc_id.get(doc.doc_id, {})
                        families_in_top_k.add(meta.get('attack_family', 'unknown'))
                    for fam in families_in_top_k:
                        poison_presence_by_family_counts[fam][k] += 1
                
                # Find first poisoned document rank
                for rank, doc in enumerate(retrieved_docs, 1):
                    if doc.is_poisoned:
                        rank_of_first_poison.append(rank)

                        # Get attack family from metadata
                        meta = self.metadata_by_doc_id.get(doc.doc_id, {})
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

        # MRR: mean of 1/rank over ALL queries (0 contribution if no poison retrieved)
        mrr = (
            sum(1.0 / r for r in rank_of_first_poison) / total_queries
            if total_queries > 0 else 0.0
        )

        return Stage3Retrieval(
            poison_presence_top_k=poison_presence_top_k_pct,
            poison_doc_count_avg_top_k=poison_doc_count_avg_top_k,
            poison_presence_top_k_by_family=poison_presence_top_k_by_family_pct,
            rank_of_first_poison=rank_of_first_poison,
            rank_distribution_by_family=dict(rank_distribution_by_family),
            mrr=mrr,
            total_queries=total_queries,
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
        
        # Initialize position buckets — keys match actual metadata values from IPI generator
        # ('early', 'late', 'random' appear in metadata; 'early_mid'/'late_mid' do not)
        position_buckets = {
            'start':  {'total': 0, 'survived': 0},
            'early':  {'total': 0, 'survived': 0},
            'mid':    {'total': 0, 'survived': 0},
            'late':   {'total': 0, 'survived': 0},
            'end':    {'total': 0, 'survived': 0},
            'random': {'total': 0, 'survived': 0},
        }
        
        structural_buckets = {
            'caption': {'total': 0, 'survived': 0},
            'code': {'total': 0, 'survived': 0},
            'citation': {'total': 0, 'survived': 0},
            'html_comment': {'total': 0, 'survived': 0},
            'json_meta': {'total': 0, 'survived': 0},
            'table_caption': {'total': 0, 'survived': 0}
        }
        
        # Character budget for the context window simulation.
        # Represents a realistic LLM context limit (≈2 000 tokens worth of context text,
        # leaving headroom for the system prompt, question, and response).
        MAX_CONTEXT_CHARS = 8000

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

                # Simulate the packing stage: concatenate docs in rank order and
                # truncate at MAX_CONTEXT_CHARS (mirrors _format_context in pipeline.py).
                context_parts = []
                char_count = 0
                for d in retrieved_docs:
                    part = f"{d.title}\n{d.text}" if getattr(d, 'title', '') else d.text
                    context_parts.append(part)
                    char_count += len(part) + 2  # +2 for the "\n\n" separator
                full_context = "\n\n".join(context_parts)
                packed_context = full_context[:MAX_CONTEXT_CHARS]

                # Check each retrieved poisoned document for span survival.
                for doc in retrieved_docs:
                    if not doc.is_poisoned:
                        continue

                    meta = self.metadata_by_doc_id.get(doc.doc_id, {})

                    # Use directive_preview (first ~50 chars of the injected span) as the
                    # survival probe — if it appears in the truncated context, the span survived.
                    directive_preview = meta.get('directive_preview', '')
                    if not directive_preview:
                        # Fall back to a substring of the doc text at the span offsets.
                        s, e = meta.get('span_start', 0), meta.get('span_end', 0)
                        if e > s:
                            directive_preview = doc.text[s:min(e, s + 50)]
                    probe = directive_preview[:50].strip()
                    survived = bool(probe) and (probe in packed_context)

                    # Position survival
                    position = meta.get('position', 'mid')
                    if position in position_buckets:
                        position_buckets[position]['total'] += 1
                        if survived:
                            position_buckets[position]['survived'] += 1

                    # Structural carrier survival
                    structural_type = meta.get('structural_carrier', '')
                    if structural_type in structural_buckets:
                        structural_buckets[structural_type]['total'] += 1
                        if survived:
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

