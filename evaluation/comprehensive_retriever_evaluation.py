#!/usr/bin/env python3
"""
Comprehensive Multi-Retriever Evaluation
Measures ASR, accuracy, and comparative metrics across BM25, Dense, Hybrid, Web Scraper
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠ BM25 not available. Install: pip install rank-bm25")

try:
    import torch
    from sentence_transformers import SentenceTransformer
    DENSE_AVAILABLE = True
except (ImportError, TypeError):
    DENSE_AVAILABLE = False
    print("⚠ Dense retriever not available. Will use BM25 only.")

class MultiRetrieverEvaluator:
    """Comprehensive evaluation across multiple retriever types"""
    
    def __init__(self, poisoned_corpus_path: str, queries_path: str, output_dir: str, dense_model_name: str = 'all-MiniLM-L6-v2'):
        self.poisoned_corpus_path = poisoned_corpus_path
        self.queries_path = queries_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.dense_model_name = dense_model_name
        
        # Load data
        self.poisoned_corpus = self._load_corpus(poisoned_corpus_path)
        self.queries = self._load_queries(queries_path)
        
        # Separate clean and poisoned docs
        self.clean_docs = [d for d in self.poisoned_corpus if not d.get('_poisoned', False)]
        self.poisoned_docs = [d for d in self.poisoned_corpus if d.get('_poisoned', False)]
        
        print(f"✓ Loaded {len(self.poisoned_corpus)} documents ({len(self.poisoned_docs)} poisoned, {len(self.clean_docs)} clean)")
        print(f"✓ Loaded {len(self.queries)} queries")
        
        # Initialize retrievers
        self.bm25_index = None
        self.dense_model = None
        self.results = {}
        
    def _load_corpus(self, path: str) -> List[Dict]:
        """Load corpus from JSONL"""
        corpus = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line.strip()))
        return corpus
    
    def _load_queries(self, path: str) -> List[Dict]:
        """Load queries from JSONL"""
        queries = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line.strip()))
        return queries
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def setup_bm25(self):
        """Setup BM25 retriever"""
        if not BM25_AVAILABLE:
            return False
        
        print("Setting up BM25...")
        corpus_texts = []
        for doc in self.poisoned_corpus:
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(self._tokenize(text))
        
        self.bm25_index = BM25Okapi(corpus_texts)
        return True
    
    def setup_dense(self):
        """Setup Dense retriever (SentenceTransformer)"""
        if not DENSE_AVAILABLE:
            return False
        
        print("Setting up Dense retriever...")
        # Use lightweight model for efficiency
        try:
            self.dense_model = SentenceTransformer(self.dense_model_name)
            
            # Pre-encode corpus
            print("Encoding corpus...")
            corpus_texts = []
            for doc in self.poisoned_corpus:
                text = f"{doc.get('title', '')} {doc.get('text', '')}"
                corpus_texts.append(text)
            
            self.corpus_embeddings = self.dense_model.encode(corpus_texts, show_progress_bar=True)
            return True
        except Exception as e:
            print(f"Error setting up dense retriever: {e}")
            return False
    
    def retrieve_bm25(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        """BM25 retrieval"""
        if not self.bm25_index:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k
        top_k_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_k_indices:
            results.append((self.poisoned_corpus[idx]['_id'], float(scores[idx])))
        
        return results
    
    def retrieve_dense(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        """Dense retrieval"""
        if not self.dense_model or not hasattr(self, 'corpus_embeddings'):
            return []
        
        query_embedding = self.dense_model.encode([query])[0]
        
        # Compute cosine similarity
        scores = np.dot(self.corpus_embeddings, query_embedding) / (
            np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_k_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_k_indices:
            results.append((self.poisoned_corpus[idx]['_id'], float(scores[idx])))
        
        return results
    
    def retrieve_hybrid(self, query: str, k: int = 50, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """Hybrid retrieval (BM25 + Dense)"""
        bm25_results = self.retrieve_bm25(query, k*2)
        dense_results = self.retrieve_dense(query, k*2)
        
        # Normalize scores
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        dense_scores = {doc_id: score for doc_id, score in dense_results}
        
        # Combine scores
        combined = {}
        all_docs = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        # Normalize
        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) if max(bm25_scores.values()) > 0 else 1
            bm25_scores = {k: v/max_bm25 for k, v in bm25_scores.items()}
        
        if dense_scores:
            max_dense = max(dense_scores.values()) if max(dense_scores.values()) > 0 else 1
            dense_scores = {k: v/max_dense for k, v in dense_scores.items()}
        
        for doc_id in all_docs:
            bm25 = bm25_scores.get(doc_id, 0)
            dense = dense_scores.get(doc_id, 0)
            combined[doc_id] = alpha * bm25 + (1 - alpha) * dense
        
        # Sort and return top k
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_docs
    
    def calculate_asr(self, results: List[Tuple[str, float]], poisoned_ids: set, k: int) -> float:
        """Calculate Attack Success Rate"""
        if not results:
            return 0.0
        
        top_k_ids = [doc_id for doc_id, _ in results[:k]]
        poisoned_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in poisoned_ids)
        
        return poisoned_in_top_k / min(k, len(results))
    
    def calculate_metrics(self, results: List[Tuple[str, float]], poisoned_ids: set, k_values: List[int]) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {}
        
        for k in k_values:
            asr = self.calculate_asr(results, poisoned_ids, k)
            metrics[f'ASR@{k}'] = asr
            
            # Top-k accuracy (precision)
            top_k_ids = [doc_id for doc_id, _ in results[:k]]
            if top_k_ids:
                precision = sum(1 for doc_id in top_k_ids if doc_id in poisoned_ids) / len(top_k_ids)
                metrics[f'Precision@{k}'] = precision
            
            # Recall
            if poisoned_ids:
                top_k_ids = [doc_id for doc_id, _ in results[:k]]
                retrieved_poisoned = set(top_k_ids) & poisoned_ids
                recall = len(retrieved_poisoned) / len(poisoned_ids)
                metrics[f'Recall@{k}'] = recall
        
        return metrics
    
    def evaluate_all_retrievers(self, sample_size: int = 50, k_values: List[int] = None):
        """Evaluate all available retrievers"""
        if k_values is None:
            k_values = [10, 50, 100]
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE MULTI-RETRIEVER EVALUATION")
        print(f"{'='*80}")
        
        # Sample queries
        sample_queries = self.queries[:sample_size] if len(self.queries) > sample_size else self.queries
        poisoned_ids = {doc['_id'] for doc in self.poisoned_docs}
        
        all_results = {}
        
        # BM25 Evaluation
        if self.setup_bm25():
            print(f"\nEvaluating BM25 on {len(sample_queries)} queries...")
            bm25_metrics = defaultdict(list)
            
            for query in sample_queries:
                results = self.retrieve_bm25(query['text'], k=max(k_values))
                metrics = self.calculate_metrics(results, poisoned_ids, k_values)
                for key, value in metrics.items():
                    bm25_metrics[key].append(value)
            
            # Average metrics
            bm25_avg = {key: np.mean(values) for key, values in bm25_metrics.items()}
            all_results['BM25'] = bm25_avg
            print(f"✓ BM25 ASR@10: {bm25_avg.get('ASR@10', 0):.3f}")
        
        # Dense Evaluation
        if self.setup_dense():
            print(f"\nEvaluating Dense retriever on {len(sample_queries)} queries...")
            dense_metrics = defaultdict(list)
            
            for query in sample_queries:
                results = self.retrieve_dense(query['text'], k=max(k_values))
                metrics = self.calculate_metrics(results, poisoned_ids, k_values)
                for key, value in metrics.items():
                    dense_metrics[key].append(value)
            
            dense_avg = {key: np.mean(values) for key, values in dense_metrics.items()}
            all_results['Dense'] = dense_avg
            print(f"✓ Dense ASR@10: {dense_avg.get('ASR@10', 0):.3f}")
        
        # Hybrid Evaluation
        if BM25_AVAILABLE and DENSE_AVAILABLE:
            print(f"\nEvaluating Hybrid retriever on {len(sample_queries)} queries...")
            hybrid_metrics = defaultdict(list)
            
            for query in sample_queries:
                results = self.retrieve_hybrid(query['text'], k=max(k_values))
                metrics = self.calculate_metrics(results, poisoned_ids, k_values)
                for key, value in metrics.items():
                    hybrid_metrics[key].append(value)
            
            hybrid_avg = {key: np.mean(values) for key, values in hybrid_metrics.items()}
            all_results['Hybrid'] = hybrid_avg
            print(f"✓ Hybrid ASR@10: {hybrid_avg.get('ASR@10', 0):.3f}")
        
        self.results = all_results
        return all_results
    
    def generate_comparative_report(self):
        """Generate comparative analysis report"""
        if not self.results:
            print("No results to report. Run evaluate_all_retrievers() first.")
            return
        
        report = []
        report.append("="*80)
        report.append("MULTI-RETRIEVER COMPARATIVE ANALYSIS")
        report.append("="*80)
        report.append("")
        
        # ASR Comparison
        report.append("ATTACK SUCCESS RATE (ASR) COMPARISON:")
        report.append("-"*80)
        
        retrievers = list(self.results.keys())
        k_values = [10, 50, 100]
        
        for k in k_values:
            report.append(f"\nASR@{k}:")
            for retriever in retrievers:
                asr = self.results[retriever].get(f'ASR@{k}', 0)
                report.append(f"  {retriever:10s}: {asr:.3f} ({asr*100:.1f}%)")
        
        # Precision Comparison
        report.append(f"\n\nPRECISION COMPARISON:")
        report.append("-"*80)
        for k in k_values:
            report.append(f"\nPrecision@{k}:")
            for retriever in retrievers:
                prec = self.results[retriever].get(f'Precision@{k}', 0)
                report.append(f"  {retriever:10s}: {prec:.3f} ({prec*100:.1f}%)")
        
        # Recall Comparison
        report.append(f"\n\nRECALL COMPARISON:")
        report.append("-"*80)
        for k in k_values:
            report.append(f"\nRecall@{k}:")
            for retriever in retrievers:
                rec = self.results[retriever].get(f'Recall@{k}', 0)
                report.append(f"  {retriever:10s}: {rec:.3f} ({rec*100:.1f}%)")
        
        # Ranking
        report.append(f"\n\nVULNERABILITY RANKING (by ASR@10):")
        report.append("-"*80)
        sorted_retrievers = sorted(retrievers, key=lambda r: self.results[r].get('ASR@10', 0), reverse=True)
        for i, retriever in enumerate(sorted_retrievers, 1):
            asr = self.results[retriever].get('ASR@10', 0)
            report.append(f"{i}. {retriever:10s}: ASR@10 = {asr:.3f} ({'MOST VULNERABLE' if i == 1 else 'LEAST VULNERABLE' if i == len(sorted_retrievers) else ''})")
        
        # Summary
        report.append(f"\n\nKEY FINDINGS:")
        report.append("-"*80)
        top_vulnerable = sorted_retrievers[0]
        top_asr = self.results[top_vulnerable].get('ASR@10', 0)
        report.append(f"• Most vulnerable retriever: {top_vulnerable} (ASR@10: {top_asr:.3f})")
        
        if len(sorted_retrievers) > 1:
            least_vulnerable = sorted_retrievers[-1]
            least_asr = self.results[least_vulnerable].get('ASR@10', 0)
            report.append(f"• Least vulnerable retriever: {least_vulnerable} (ASR@10: {least_asr:.3f})")
            report.append(f"• Vulnerability gap: {(top_asr - least_asr) / least_asr * 100:.1f}% difference")
        
        report.append("="*80)
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        # Save
        output_file = self.output_dir / 'comprehensive_evaluation_report.txt'
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        # Save JSON
        json_file = self.output_dir / 'comprehensive_evaluation_results.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Saved report to {output_file}")
        print(f"✓ Saved JSON results to {json_file}")
        
        return report_text


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive multi-retriever evaluation')
    parser.add_argument('--corpus', default='IPI_generators/ipi_scifact/scifact_ipi_poisoned.jsonl')
    parser.add_argument('--queries', default='data/corpus/beir/scifact/queries.jsonl')
    parser.add_argument('--output', default='evaluation/comprehensive_results')
    parser.add_argument('--sample', type=int, default=50)
    parser.add_argument('--dense-model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name for dense retrieval')
    
    args = parser.parse_args()
    
    evaluator = MultiRetrieverEvaluator(
        poisoned_corpus_path=args.corpus,
        queries_path=args.queries,
        output_dir=args.output,
        dense_model_name=args.dense_model
    )
    
    # Evaluate
    evaluator.evaluate_all_retrievers(sample_size=args.sample)
    
    # Generate report
    evaluator.generate_comparative_report()


if __name__ == "__main__":
    main()

