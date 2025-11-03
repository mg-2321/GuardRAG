#!/usr/bin/env python3
"""
Mid-Position Attack Evaluation with Dense/Hybrid Retrievers
Compares ASR for mid-position attacks across BM25, Dense, and Hybrid
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import torch
    from sentence_transformers import SentenceTransformer
    DENSE_AVAILABLE = True
except (ImportError, TypeError, AttributeError) as e:
    DENSE_AVAILABLE = False
    print(f"⚠ Dense retriever not available: {e}")

class MidPositionRetrieverEvaluator:
    """Evaluate mid-position attacks across different retrievers"""
    
    def __init__(self, poisoned_corpus_path: str, queries_path: str, output_dir: str, dense_model_name: str = 'all-MiniLM-L6-v2'):
        self.poisoned_corpus_path = poisoned_corpus_path
        self.queries_path = queries_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.dense_model_name = dense_model_name
        
        self.poisoned_corpus = self._load_corpus(poisoned_corpus_path)
        self.queries = self._load_queries(queries_path)
        
        # Separate by position
        self.mid_poisoned = [d for d in self.poisoned_corpus if d.get('_poisoned') and 'mid' in d.get('_placement', '').lower()]
        self.start_poisoned = [d for d in self.poisoned_corpus if d.get('_poisoned') and ('start' in d.get('_placement', '').lower() or 'title' in d.get('_placement', '').lower())]
        self.all_poisoned = [d for d in self.poisoned_corpus if d.get('_poisoned')]
        
        print(f"✓ Loaded {len(self.poisoned_corpus)} docs")
        print(f"  - Mid-position poisoned: {len(self.mid_poisoned)}")
        print(f"  - Start-position poisoned: {len(self.start_poisoned)}")
        print(f"  - All poisoned: {len(self.all_poisoned)}")
    
    def _load_corpus(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    
    def _load_queries(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    
    def _tokenize(self, text: str):
        return text.lower().split()
    
    def setup_retrievers(self):
        """Setup BM25 and Dense retrievers"""
        corpus_texts = []
        for doc in self.poisoned_corpus:
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(self._tokenize(text))
        
        # BM25
        self.bm25_index = BM25Okapi(corpus_texts) if BM25_AVAILABLE else None
        
        # Dense
        if DENSE_AVAILABLE:
            print("Setting up Dense retriever...")
            self.dense_model = SentenceTransformer(self.dense_model_name)
            corpus_texts_full = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in self.poisoned_corpus]
            print("Encoding corpus...")
            self.corpus_embeddings = self.dense_model.encode(corpus_texts_full, show_progress_bar=False)
        else:
            self.dense_model = None
    
    def retrieve_bm25(self, query: str, k: int = 50):
        """BM25 retrieval"""
        if not self.bm25_index:
            return []
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [(self.poisoned_corpus[idx]['_id'], float(scores[idx])) for idx in top_k_indices]
    
    def retrieve_dense(self, query: str, k: int = 50):
        """Dense retrieval"""
        if not self.dense_model:
            return []
        query_embedding = self.dense_model.encode([query])[0]
        scores = np.dot(self.corpus_embeddings, query_embedding) / (
            np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [(self.poisoned_corpus[idx]['_id'], float(scores[idx])) for idx in top_k_indices]
    
    def retrieve_hybrid(self, query: str, k: int = 50, alpha: float = 0.5):
        """Hybrid retrieval"""
        bm25_results = self.retrieve_bm25(query, k*2)
        dense_results = self.retrieve_dense(query, k*2)
        
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        dense_scores = {doc_id: score for doc_id, score in dense_results}
        
        # Normalize
        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) if max(bm25_scores.values()) > 0 else 1
            bm25_scores = {k: v/max_bm25 for k, v in bm25_scores.items()}
        if dense_scores:
            max_dense = max(dense_scores.values()) if max(dense_scores.values()) > 0 else 1
            dense_scores = {k: v/max_dense for k, v in dense_scores.items()}
        
        # Combine
        combined = {}
        all_docs = set(bm25_scores.keys()) | set(dense_scores.keys())
        for doc_id in all_docs:
            combined[doc_id] = alpha * bm25_scores.get(doc_id, 0) + (1-alpha) * dense_scores.get(doc_id, 0)
        
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    
    def calculate_asr(self, results, poisoned_ids, k: int):
        """Calculate ASR"""
        if not results:
            return 0.0
        top_k_ids = [doc_id for doc_id, _ in results[:k]]
        poisoned_in_topk = sum(1 for doc_id in top_k_ids if doc_id in poisoned_ids)
        return poisoned_in_topk / min(k, len(results))
    
    def evaluate_by_position(self, sample_size: int = 50, k_values: list = [10, 50]):
        """Evaluate ASR by attack position"""
        print(f"\n{'='*80}")
        print("MID-POSITION vs START-POSITION COMPARISON")
        print(f"{'='*80}")
        
        self.setup_retrievers()
        
        sample_queries = self.queries[:sample_size]
        mid_poisoned_ids = {doc['_id'] for doc in self.mid_poisoned}
        start_poisoned_ids = {doc['_id'] for doc in self.start_poisoned}
        all_poisoned_ids = {doc['_id'] for doc in self.all_poisoned}
        
        results = {}
        
        for retriever_name, retrieve_func in [
            ('BM25', self.retrieve_bm25),
            ('Dense', self.retrieve_dense),
            ('Hybrid', lambda q, k: self.retrieve_hybrid(q, k))
        ]:
            if retriever_name == 'Dense' and not DENSE_AVAILABLE:
                continue
            if retriever_name == 'BM25' and not BM25_AVAILABLE:
                continue
            
            print(f"\nEvaluating {retriever_name}...")
            
            retriever_results = {}
            for position_type, poisoned_ids in [
                ('Mid-Position', mid_poisoned_ids),
                ('Start-Position', start_poisoned_ids),
                ('All-Position', all_poisoned_ids)
            ]:
                asr_scores = defaultdict(list)
                
                for query in sample_queries:
                    results_list = retrieve_func(query['text'], k=max(k_values))
                    
                    for k in k_values:
                        asr = self.calculate_asr(results_list, poisoned_ids, k)
                        asr_scores[k].append(asr)
                
                # Average
                retriever_results[position_type] = {
                    f'ASR@{k}': np.mean(asr_scores[k]) for k in k_values
                }
            
            results[retriever_name] = retriever_results
        
        # Report
        print(f"\n{'='*80}")
        print("RESULTS: ASR BY POSITION AND RETRIEVER")
        print(f"{'='*80}")
        
        for k in k_values:
            print(f"\nASR@{k}:")
            print(f"{'Retriever':<15} {'Mid-Pos':<12} {'Start-Pos':<12} {'All-Pos':<12}")
            print("-" * 60)
            
            for retriever in ['BM25', 'Dense', 'Hybrid']:
                if retriever not in results:
                    continue
                mid_asr = results[retriever]['Mid-Position'].get(f'ASR@{k}', 0)
                start_asr = results[retriever]['Start-Position'].get(f'ASR@{k}', 0)
                all_asr = results[retriever]['All-Position'].get(f'ASR@{k}', 0)
                
                print(f"{retriever:<15} {mid_asr:>10.3f}    {start_asr:>10.3f}    {all_asr:>10.3f}")
        
        # Key findings
        print(f"\n{'='*80}")
        print("KEY FINDINGS:")
        print(f"{'='*80}")
        
        if 'BM25' in results and 'Dense' in results:
            bm25_mid = results['BM25']['Mid-Position'].get('ASR@10', 0)
            dense_mid = results['Dense']['Mid-Position'].get('ASR@10', 0)
            
            print(f"Mid-Position ASR@10:")
            print(f"  BM25:  {bm25_mid:.3f}")
            print(f"  Dense: {dense_mid:.3f}")
            
            if dense_mid > bm25_mid:
                improvement = (dense_mid - bm25_mid) / bm25_mid * 100 if bm25_mid > 0 else 0
                print(f"  ✅ Dense is {improvement:.1f}% better for mid-position attacks!")
            
        # Save
        output_file = self.output_dir / 'mid_position_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved results to {output_file}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='IPI_generators/ipi_scifact/scifact_ipi_poisoned.jsonl')
    parser.add_argument('--queries', default='data/corpus/beir/scifact/queries.jsonl')
    parser.add_argument('--output', default='evaluation/mid_position_results')
    parser.add_argument('--sample', type=int, default=50)
    parser.add_argument('--dense-model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name for dense retrieval')
    
    args = parser.parse_args()
    
    evaluator = MidPositionRetrieverEvaluator(
        poisoned_corpus_path=args.corpus,
        queries_path=args.queries,
        output_dir=args.output,
        dense_model_name=args.dense_model
    )
    
    evaluator.evaluate_by_position(sample_size=args.sample)


if __name__ == "__main__":
    main()

