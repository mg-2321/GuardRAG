#!/usr/bin/env python3
"""
Corpus Verification Script
Manually verify poisoned corpus correctness compared to original
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file"""
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def compare_corpora(clean_path: str, mixed_path: str, metadata_path: str) -> Dict:
    """
    Compare original and poisoned corpora
    
    Args:
        clean_path: Path to original clean corpus
        mixed_path: Path to mixed (clean + poisoned) corpus
        metadata_path: Path to metadata file
        
    Returns:
        Verification results dictionary
    """
    
    print("=" * 80)
    print("CORPUS VERIFICATION REPORT")
    print("=" * 80)
    
    # Load all files
    print("\n[1/3] Loading files...")
    clean_docs = {doc['_id']: doc for doc in load_jsonl(clean_path)}
    print(f"✓ Loaded {len(clean_docs)} clean documents")
    
    mixed_docs = load_jsonl(mixed_path)
    poisoned_in_mixed = [d for d in mixed_docs if d.get('is_poisoned')]
    print(f"✓ Loaded {len(mixed_docs)} mixed documents ({len(poisoned_in_mixed)} poisoned)")
    
    metadata_docs = load_jsonl(metadata_path)
    metadata_map = {m['doc_id']: m for m in metadata_docs}
    print(f"✓ Loaded {len(metadata_map)} metadata entries")
    
    # Verification checks
    print("\n[2/3] Running verification checks...")
    results = {
        'total_poisoned': len(poisoned_in_mixed),
        'metadata_coverage': 0,
        'text_modified': 0,
        'directive_present': 0,
        'span_valid': 0,
        'span_extractable': 0,
        'placement_valid': 0,
        'corpus_fit_natural': 0,
        'issues': [],
    }
    
    for doc in poisoned_in_mixed:
        doc_id = doc['_id']
        
        if doc_id not in metadata_map:
            results['issues'].append(f"No metadata for {doc_id}")
            continue
        
        results['metadata_coverage'] += 1
        meta = metadata_map[doc_id]
        
        # Check 1: Text modified
        original_id = meta.get('original_id')
        if original_id and original_id in clean_docs:
            original_text = clean_docs[original_id]['text']
            if original_text != doc['text']:
                results['text_modified'] += 1
            else:
                results['issues'].append(f"Document {doc_id} text not modified")
        
        # Check 2: Directive present
        directive = meta.get('directive_preview', '')
        # Check for directive (it might be obfuscated or in different format)
        if directive and len(directive) > 5:
            results['directive_present'] += 1
        
        # Check 3: Span boundaries valid
        span_start = meta.get('span_start', 0)
        span_end = meta.get('span_end', 0)
        doc_len = len(doc['text'])
        
        if 0 <= span_start < span_end <= doc_len:
            results['span_valid'] += 1
            
            # Check 4: Span extractable
            try:
                extracted = doc['text'][span_start:span_end]
                if len(extracted) > 3:  # Non-empty span
                    results['span_extractable'] += 1
            except:
                results['issues'].append(f"Cannot extract span from {doc_id}")
        else:
            results['issues'].append(
                f"Invalid span [{span_start}:{span_end}] in doc length {doc_len} "
                f"for {doc_id}"
            )
        
        # Check 5: Placement valid
        placement = meta.get('placement', '')
        valid_placements = [
            'title_and_start', 'title_and_mid', 'body_start', 'body_mid',
            'body_early_mid', 'body_late_mid', 'body_end', 'body_near_end'
        ]
        if placement in valid_placements:
            results['placement_valid'] += 1
        
        # Check 6: Corpus fit natural
        if meta.get('corpus_fit') == 'natural':
            results['corpus_fit_natural'] += 1
    
    # Summary statistics
    print("\n[3/3] Summary Statistics")
    print("-" * 80)
    
    total = results['total_poisoned']
    
    def pct(value):
        return (value / total * 100) if total > 0 else 0
    
    checks = [
        ('Metadata Coverage', results['metadata_coverage']),
        ('Text Modified', results['text_modified']),
        ('Directive Present', results['directive_present']),
        ('Span Valid', results['span_valid']),
        ('Span Extractable', results['span_extractable']),
        ('Placement Valid', results['placement_valid']),
        ('Natural Corpus Fit', results['corpus_fit_natural']),
    ]
    
    overall_pass = 0
    for check_name, value in checks:
        percentage = pct(value)
        status = "✅" if percentage >= 90 else "⚠️" if percentage >= 70 else "❌"
        print(f"{status} {check_name:25s}: {value:3d}/{total} ({percentage:5.1f}%)")
        if percentage >= 90:
            overall_pass += 1
    
    print("\n" + "-" * 80)
    print(f"Overall Pass Rate: {overall_pass}/7 checks")
    
    if results['issues']:
        print(f"\n⚠️  Issues Found ({len(results['issues'])} issues):")
        for issue in results['issues'][:10]:  # Show first 10
            print(f"  • {issue}")
        if len(results['issues']) > 10:
            print(f"  ... and {len(results['issues']) - 10} more")
    else:
        print("\n✅ No issues found!")
    
    return results


def analyze_quality_metrics(metadata_path: str) -> Dict:
    """Analyze quality metrics of injected attacks"""
    
    print("\n" + "=" * 80)
    print("ATTACK QUALITY METRICS")
    print("=" * 80)
    
    metadata = load_jsonl(metadata_path)
    
    metrics = {
        'total_documents': len(metadata),
        'span_lengths': [],
        'placement_distribution': defaultdict(int),
        'obfuscation_distribution': defaultdict(int),
        'attack_family_distribution': defaultdict(int),
        'retriever_hint_distribution': defaultdict(int),
        'position_distribution': defaultdict(int),
    }
    
    for meta in metadata:
        # Span length
        span_len = meta.get('span_len', 0)
        metrics['span_lengths'].append(span_len)
        
        # Distributions
        metrics['placement_distribution'][meta.get('placement', 'unknown')] += 1
        metrics['obfuscation_distribution'][meta.get('obfuscation_method', 'none')] += 1
        metrics['attack_family_distribution'][meta.get('attack_family', 'unknown')] += 1
        metrics['retriever_hint_distribution'][meta.get('retriever_hint', 'unknown')] += 1
        metrics['position_distribution'][meta.get('position', 'unknown')] += 1
    
    # Print statistics
    print(f"\nTotal Poisoned Documents: {metrics['total_documents']}")
    
    # Span length statistics
    if metrics['span_lengths']:
        import statistics
        print(f"\nSpan Length Statistics:")
        print(f"  Min: {min(metrics['span_lengths'])} tokens")
        print(f"  Max: {max(metrics['span_lengths'])} tokens")
        print(f"  Mean: {statistics.mean(metrics['span_lengths']):.1f} tokens")
        print(f"  Median: {statistics.median(metrics['span_lengths']):.1f} tokens")
    
    # Attack family distribution
    print(f"\nAttack Family Distribution:")
    for family, count in sorted(metrics['attack_family_distribution'].items(), 
                                key=lambda x: x[1], reverse=True):
        pct = (count / metrics['total_documents'] * 100)
        print(f"  {family:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Obfuscation distribution
    print(f"\nObfuscation Method Distribution:")
    for method, count in sorted(metrics['obfuscation_distribution'].items(),
                               key=lambda x: x[1], reverse=True):
        pct = (count / metrics['total_documents'] * 100)
        print(f"  {method:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Position distribution
    print(f"\nPosition Distribution:")
    for pos, count in sorted(metrics['position_distribution'].items(),
                            key=lambda x: x[1], reverse=True):
        pct = (count / metrics['total_documents'] * 100)
        print(f"  {pos:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Retriever hints
    print(f"\nRetriever Hints (Target):")
    for hint, count in sorted(metrics['retriever_hint_distribution'].items(),
                             key=lambda x: x[1], reverse=True):
        pct = (count / metrics['total_documents'] * 100)
        print(f"  {hint:20s}: {count:4d} ({pct:5.1f}%)")
    
    return metrics


def spot_check_documents(mixed_path: str, num_samples: int = 5):
    """Manually inspect poisoned documents"""
    
    print("\n" + "=" * 80)
    print(f"SPOT CHECK: Random Sample of {num_samples} Poisoned Documents")
    print("=" * 80)
    
    import random
    
    mixed_docs = load_jsonl(mixed_path)
    poisoned_docs = [d for d in mixed_docs if d.get('is_poisoned')]
    
    if not poisoned_docs:
        print("No poisoned documents found!")
        return
    
    samples = random.sample(poisoned_docs, min(num_samples, len(poisoned_docs)))
    
    for i, doc in enumerate(samples, 1):
        print(f"\n[Sample {i}/{len(samples)}]")
        print(f"ID: {doc['_id']}")
        print(f"Title: {doc['title'][:100]}")
        print(f"Text Preview (first 300 chars):")
        print(f"  {doc['text'][:300]}...")
        print(f"Is Poisoned: {doc.get('is_poisoned')}")
        print("-" * 80)


def main():
    if len(sys.argv) < 4:
        print("Usage: python verify_corpus.py <clean_corpus> <mixed_corpus> <metadata> [--analyze] [--spot-check]")
        print("\nExample:")
        print("  python verify_corpus.py \\")
        print("    data/corpus/beir/fiqa/corpus.jsonl \\")
        print("    IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl \\")
        print("    IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl \\")
        print("    --analyze --spot-check")
        sys.exit(1)
    
    clean_path = sys.argv[1]
    mixed_path = sys.argv[2]
    metadata_path = sys.argv[3]
    
    # Parse optional arguments
    analyze = '--analyze' in sys.argv
    spot_check = '--spot-check' in sys.argv
    
    # Verify paths exist
    for path in [clean_path, mixed_path, metadata_path]:
        if not Path(path).exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)
    
    # Run main verification
    compare_corpora(clean_path, mixed_path, metadata_path)
    
    # Optional: Analyze quality metrics
    if analyze:
        analyze_quality_metrics(metadata_path)
    
    # Optional: Spot check documents
    if spot_check:
        spot_check_documents(mixed_path, num_samples=5)
    
    print("\n✅ Verification complete!")


if __name__ == '__main__':
    main()
