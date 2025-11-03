#!/usr/bin/env python3
"""
ASR Diagnostic Script - Investigates why ASR might be lower than expected
Validates query-document alignment, checks retrieval logic, and provides insights
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠ BM25 not available")

class ASRDiagnostic:
    """Diagnose ASR calculation issues"""
    
    def __init__(self, poisoned_corpus_path: str, queries_path: str, metadata_path: str = None):
        self.poisoned_corpus_path = poisoned_corpus_path
        self.queries_path = queries_path
        self.metadata_path = metadata_path
        
        self.poisoned_corpus = self._load_corpus(poisoned_corpus_path)
        self.queries = self._load_queries(queries_path)
        
        # Separate docs
        self.clean_docs = [d for d in self.poisoned_corpus if not d.get('_poisoned', False)]
        self.poisoned_docs = [d for d in self.poisoned_corpus if d.get('_poisoned', False)]
        
        # Load metadata if available
        if metadata_path and Path(metadata_path).exists():
            self.metadata = self._load_metadata(metadata_path)
        else:
            self.metadata = []
        
        print(f"✓ Loaded {len(self.poisoned_corpus)} docs ({len(self.poisoned_docs)} poisoned)")
        print(f"✓ Loaded {len(self.queries)} queries")
    
    def _load_corpus(self, path: str) -> List[Dict]:
        corpus = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line.strip()))
        return corpus
    
    def _load_queries(self, path: str) -> List[Dict]:
        queries = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line.strip()))
        return queries
    
    def _load_metadata(self, path: str) -> List[Dict]:
        metadata = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line.strip()))
        return metadata
    
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def diagnose_query_alignment(self, sample_size: int = 10):
        """Check if queries align with poisoned documents"""
        print(f"\n{'='*80}")
        print("DIAGNOSTIC 1: QUERY-DOCUMENT ALIGNMENT")
        print(f"{'='*80}")
        
        if not BM25_AVAILABLE:
            print("⚠ BM25 not available for alignment check")
            return
        
        # Build BM25 index
        corpus_texts = []
        for doc in self.poisoned_corpus:
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(self._tokenize(text))
        
        bm25_index = BM25Okapi(corpus_texts)
        
        # Sample queries
        sample_queries = random.sample(self.queries, min(sample_size, len(self.queries)))
        poisoned_ids = {doc['_id'] for doc in self.poisoned_docs}
        
        alignment_stats = {
            'queries_tested': 0,
            'poisoned_retrieved': 0,
            'clean_retrieved': 0,
            'retrieval_details': []
        }
        
        for query in sample_queries:
            query_text = query.get('text', '')
            query_tokens = self._tokenize(query_text)
            scores = bm25_index.get_scores(query_tokens)
            
            # Top 10 results
            top_10_indices = np.argsort(scores)[::-1][:10]
            top_10_ids = [self.poisoned_corpus[idx]['_id'] for idx in top_10_indices]
            
            poisoned_in_top10 = sum(1 for doc_id in top_10_ids if doc_id in poisoned_ids)
            clean_in_top10 = 10 - poisoned_in_top10
            
            alignment_stats['queries_tested'] += 1
            alignment_stats['poisoned_retrieved'] += poisoned_in_top10
            alignment_stats['clean_retrieved'] += clean_in_top10
            
            alignment_stats['retrieval_details'].append({
                'query': query_text[:60],
                'poisoned_in_top10': poisoned_in_top10,
                'clean_in_top10': clean_in_top10,
                'top_score': float(scores[top_10_indices[0]]) if len(top_10_indices) > 0 else 0
            })
        
        # Report
        avg_poisoned = alignment_stats['poisoned_retrieved'] / alignment_stats['queries_tested']
        avg_clean = alignment_stats['clean_retrieved'] / alignment_stats['queries_tested']
        
        print(f"Average poisoned docs in top-10: {avg_poisoned:.2f}")
        print(f"Average clean docs in top-10: {avg_clean:.2f}")
        
        if avg_poisoned < 1.0:
            print("\n⚠️  ISSUE: Very few poisoned docs being retrieved")
            print("   Possible causes:")
            print("   - Queries don't match poisoned document keywords")
            print("   - Poisoned docs lack query-relevant content")
            print("   - Injection doesn't boost retrieval enough")
        else:
            print("✓ Poisoned documents are being retrieved")
        
        return alignment_stats
    
    def diagnose_injection_effectiveness(self, sample_size: int = 20):
        """Check if injections actually boost retrieval"""
        print(f"\n{'='*80}")
        print("DIAGNOSTIC 2: INJECTION EFFECTIVENESS")
        print(f"{'='*80}")
        
        if not BM25_AVAILABLE:
            print("⚠ BM25 not available")
            return
        
        # Sample poisoned docs
        sample_poisoned = random.sample(self.poisoned_docs, min(sample_size, len(self.poisoned_docs)))
        
        effectiveness_stats = {
            'analyzed': 0,
            'with_query_keywords': 0,
            'with_directive_in_start': 0,
            'avg_injection_tokens': 0
        }
        
        for doc in sample_poisoned:
            effectiveness_stats['analyzed'] += 1
            
            text = doc.get('text', '')
            title = doc.get('title', '')
            
            # Check if directive is at start
            directive = doc.get('_directive', '')
            if directive and text.startswith(directive[:20]):
                effectiveness_stats['with_directive_in_start'] += 1
            
            # Check injection length
            if doc.get('_span_tokens'):
                effectiveness_stats['avg_injection_tokens'] += doc['_span_tokens']
            
            # Check for query-like keywords in title/text
            combined = f"{title} {text}"
            query_markers = ['research', 'study', 'findings', 'analysis', 'method', 'result']
            if any(marker in combined.lower()[:200] for marker in query_markers):
                effectiveness_stats['with_query_keywords'] += 1
        
        if effectiveness_stats['analyzed'] > 0:
            effectiveness_stats['avg_injection_tokens'] /= effectiveness_stats['analyzed']
        
        print(f"Docs with directive at start: {effectiveness_stats['with_directive_in_start']}/{effectiveness_stats['analyzed']}")
        print(f"Docs with query keywords: {effectiveness_stats['with_query_keywords']}/{effectiveness_stats['analyzed']}")
        print(f"Average injection tokens: {effectiveness_stats['avg_injection_tokens']:.1f}")
        
        if effectiveness_stats['with_directive_in_start'] < effectiveness_stats['analyzed'] * 0.5:
            print("\n⚠️  ISSUE: Many injections not at document start")
            print("   - BM25 attacks benefit from start-position placement")
        
        return effectiveness_stats
    
    def diagnose_asr_calculation(self, k: int = 10):
        """Verify ASR calculation logic"""
        print(f"\n{'='*80}")
        print("DIAGNOSTIC 3: ASR CALCULATION VERIFICATION")
        print(f"{'='*80}")
        
        if not BM25_AVAILABLE:
            print("⚠ BM25 not available")
            return
        
        # Build index
        corpus_texts = []
        for doc in self.poisoned_corpus:
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(self._tokenize(text))
        
        bm25_index = BM25Okapi(corpus_texts)
        
        poisoned_ids = {doc['_id'] for doc in self.poisoned_docs}
        total_poisoned = len(poisoned_ids)
        
        print(f"Total poisoned documents: {total_poisoned}")
        print(f"Testing top-{k} retrieval...")
        
        # Test ASR calculation
        asr_scores = []
        
        for query in self.queries[:50]:  # Test 50 queries
            query_text = query.get('text', '')
            query_tokens = self._tokenize(query_text)
            scores = bm25_index.get_scores(query_tokens)
            
            # Get top k
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_ids = [self.poisoned_corpus[idx]['_id'] for idx in top_k_indices]
            
            # Calculate ASR
            poisoned_in_topk = sum(1 for doc_id in top_k_ids if doc_id in poisoned_ids)
            asr = poisoned_in_topk / k  # ASR as fraction of top-k that are poisoned
            asr_scores.append(asr)
        
        avg_asr = np.mean(asr_scores)
        
        print(f"\nCalculated ASR@{k}: {avg_asr:.3f} ({avg_asr*100:.1f}%)")
        print(f"Expected ASR (from papers): ~0.72 (72%)")
        
        if avg_asr < 0.20:
            print("\n⚠️  LOW ASR DETECTED")
            print("   Potential issues:")
            print("   1. Poisoned docs not optimized for query keywords")
            print("   2. Queries don't match poisoned document domains")
            print("   3. Need query-specific poisoning (not general)")
        
        return {'asr': avg_asr, 'scores': asr_scores}
    
    def compare_with_expected(self):
        """Compare results with paper expectations"""
        print(f"\n{'='*80}")
        print("DIAGNOSTIC 4: COMPARISON WITH PAPERS")
        print(f"{'='*80}")
        
        print("Expected from Zou et al. (2023):")
        print("  - BM25 optimized attacks: ~72% ASR")
        print("  - Keyword packing technique: High success")
        print("  - Injection at start/title: Optimal")
        
        print("\nYour corpus characteristics:")
        print(f"  - Total poisoned: {len(self.poisoned_docs)}")
        print(f"  - Poisoning rate: {len(self.poisoned_docs)/len(self.poisoned_corpus)*100:.1f}%")
        
        # Analyze techniques
        techniques = defaultdict(int)
        placements = defaultdict(int)
        
        for doc in self.poisoned_docs:
            tech = doc.get('_technique', 'unknown')
            placement = doc.get('_placement', 'unknown')
            techniques[tech] += 1
            placements[placement] += 1
        
        print(f"\nAttack techniques used:")
        for tech, count in sorted(techniques.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {tech}: {count}")
        
        print(f"\nPlacement distribution:")
        for placement, count in sorted(placements.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {placement}: {count}")
        
        # Check if using optimal techniques
        optimal_techs = ['keyword_packing', 'bm25_keyword_packing', 'near_query_placement']
        has_optimal = any(tech in techniques for tech in optimal_techs)
        
        if has_optimal:
            print("\n✓ Using BM25-optimized techniques")
        else:
            print("\n⚠️  Not using known BM25-optimized techniques")
            print("   Consider adding keyword_packing attacks")
    
    def run_full_diagnostic(self):
        """Run all diagnostics"""
        print("="*80)
        print("ASR DIAGNOSTIC SUITE")
        print("="*80)
        
        results = {}
        
        # Diagnostic 1: Query alignment
        results['alignment'] = self.diagnose_query_alignment(sample_size=20)
        
        # Diagnostic 2: Injection effectiveness
        results['effectiveness'] = self.diagnose_injection_effectiveness(sample_size=30)
        
        # Diagnostic 3: ASR calculation
        results['asr_verification'] = self.diagnose_asr_calculation(k=10)
        
        # Diagnostic 4: Paper comparison
        self.compare_with_expected()
        
        # Summary
        print(f"\n{'='*80}")
        print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
        print(f"{'='*80}")
        
        avg_poisoned = results['alignment']['poisoned_retrieved'] / results['alignment']['queries_tested']
        asr = results['asr_verification']['asr']
        
        print(f"\nKey Metrics:")
        print(f"  - Average poisoned docs in top-10: {avg_poisoned:.2f}")
        print(f"  - Calculated ASR@10: {asr:.3f}")
        
        print(f"\nRecommendations:")
        if avg_poisoned < 1.0 or asr < 0.20:
            print("  1. ✅ Use query-specific keyword injection")
            print("  2. ✅ Ensure injections in title/start position for BM25")
            print("  3. ✅ Test with nfcorpus (showed 19-54% ASR)")
            print("  4. ✅ Consider using queries that match poisoned doc topics")
        else:
            print("  ✓ ASR calculation appears correct")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='IPI_generators/ipi_scifact/scifact_ipi_poisoned.jsonl')
    parser.add_argument('--queries', default='data/corpus/beir/scifact/queries.jsonl')
    parser.add_argument('--metadata', default=None)
    
    args = parser.parse_args()
    
    diagnostic = ASRDiagnostic(
        poisoned_corpus_path=args.corpus,
        queries_path=args.queries,
        metadata_path=args.metadata
    )
    
    results = diagnostic.run_full_diagnostic()
    
    # Save results
    output_path = Path('evaluation/diagnostic_results.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Saved diagnostic results to {output_path}")


if __name__ == "__main__":
    main()




