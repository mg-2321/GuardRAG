#!/usr/bin/env python3
"""
Comparative Retriever Evaluation (Study A)
Compare BM25, Dense (E5), SPLADE, and Hybrid retrievers
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline_components.pipeline import Pipeline, PipelineConfig
from evaluation.stage_by_stage_evaluation import StageByStageEvaluator

RETRIEVERS = {
    'bm25': {'retriever': 'bm25'},
    'dense': {'retriever': 'dense', 'retriever_kwargs': {'model_name': 'intfloat/e5-large-v2'}},
    'splade': {'retriever': 'splade'},
    'hybrid_0.2': {'retriever': 'hybrid', 'retriever_kwargs': {'alpha': 0.2, 'dense_model': 'intfloat/e5-large-v2'}},
    'hybrid_0.5': {'retriever': 'hybrid', 'retriever_kwargs': {'alpha': 0.5, 'dense_model': 'intfloat/e5-large-v2'}},
    'hybrid_0.8': {'retriever': 'hybrid', 'retriever_kwargs': {'alpha': 0.8, 'dense_model': 'intfloat/e5-large-v2'}},
}

def evaluate_retriever(corpus_path: str, metadata_path: str, queries_path: str,
                       corpus_name: str, retriever_name: str, retriever_config: Dict,
                       output_dir: Path, sample_size: int = None):
    """Evaluate a single retriever"""
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {retriever_name.upper()} on {corpus_name.upper()}")
    print(f"{'='*80}")
    
    if retriever_name.startswith('dense') or retriever_name == 'splade':
        print("⚠️  This will take 3-5 minutes (model loading + corpus encoding)")
        print("   Subsequent runs will be faster (cached embeddings)")
    
    try:
        # Add cache directory for dense/hybrid retrievers
        retriever_kwargs = retriever_config.get('retriever_kwargs', {}).copy()
        if retriever_config['retriever'] in ['dense', 'splade', 'hybrid']:
            cache_dir = str(output_dir / 'embeddings_cache')
            retriever_kwargs['cache_dir'] = cache_dir
        
        config = PipelineConfig(
            document_path=corpus_path,
            retriever=retriever_config['retriever'],
            retriever_kwargs=retriever_kwargs,
            default_top_k=10,
            generation=None  # Skip generation for speed
        )
        
        pipeline = Pipeline(config)
        
        evaluator = StageByStageEvaluator(
            corpus_path=corpus_path,
            metadata_path=metadata_path,
            queries_path=queries_path,
            corpus_name=corpus_name
        )
        
        # Run Stage 3 (retrieval) - most important
        stage3_result = evaluator.evaluate_stage3_retrieval(
            pipeline,
            k_values=[1, 3, 5, 10],
            sample_size=sample_size,
            skip_generation=True
        )
        
        # Run Stage 5 (packing)
        stage5_result = evaluator.evaluate_stage5_packing(
            pipeline,
            sample_size=sample_size
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {retriever_name.upper()} completed in {elapsed/60:.1f} minutes")
        
        return {
            'retriever': retriever_name,
            'stage3_retrieval': {
                'total_queries': stage3_result.total_queries,
                'poison_presence_top_k': stage3_result.poison_presence_top_k,
                'poison_doc_count_avg_top_k': stage3_result.poison_doc_count_avg_top_k,
                'poison_presence_top_k_by_family': stage3_result.poison_presence_top_k_by_family,
                # MRR = mean(1/rank) over ALL queries (0 contribution if no poison retrieved)
                'mrr': stage3_result.mrr,
                # avg_rank_first_poison: mean rank over queries WHERE poison WAS retrieved (different from MRR)
                'avg_rank_first_poison': (
                    sum(stage3_result.rank_of_first_poison) / len(stage3_result.rank_of_first_poison)
                    if stage3_result.rank_of_first_poison else 0.0
                ),
                'rank_distribution_by_family': stage3_result.rank_distribution_by_family
            },
            'stage5_packing': {
                'position_survival_rates': stage5_result.position_survival_rates,
                'structural_survival_rates': stage5_result.structural_survival_rates
            }
        }
        
    except Exception as e:
        print(f"⚠️  Error with {retriever_name}: {e}")
        import traceback
        traceback.print_exc()
        return {'retriever': retriever_name, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Comparative retriever evaluation')
    parser.add_argument('--corpus', required=True, help='Path to mixed corpus JSONL')
    parser.add_argument('--metadata', required=True, help='Path to metadata JSONL')
    parser.add_argument('--queries', required=True, help='Path to queries JSONL')
    parser.add_argument('--corpus-name', required=True, help='Corpus name')
    parser.add_argument('--output-dir', default='evaluation/comparative_results', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size (default: all queries)')
    parser.add_argument('--retrievers', nargs='+', default=None,
                        help='Retrievers to run (default: all). Choices: bm25 dense splade hybrid_0.2 hybrid_0.5 hybrid_0.8')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_retrievers = {
        k: v for k, v in RETRIEVERS.items()
        if args.retrievers is None or k in args.retrievers
    }

    results = {}

    for retriever_name, retriever_config in active_retrievers.items():
        result = evaluate_retriever(
            args.corpus, args.metadata, args.queries,
            args.corpus_name, retriever_name, retriever_config,
            output_dir, args.sample_size
        )
        results[retriever_name] = result
    
    # Save results
    output_file = output_dir / f"{args.corpus_name}_retriever_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RETRIEVER COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Retriever':<20} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Top-10':<10} {'MRR':<8} {'AvgRank':<8}")
    print("-"*80)

    for retriever_name, result in results.items():
        if 'error' in result:
            print(f"{retriever_name:<20} {'ERROR':<8}")
            continue

        s3 = result.get('stage3_retrieval', {})
        poison_presence = s3.get('poison_presence_top_k', {})
        top1 = poison_presence.get(1, poison_presence.get('1', 0))
        top3 = poison_presence.get(3, poison_presence.get('3', 0))
        top5 = poison_presence.get(5, poison_presence.get('5', 0))
        top10 = poison_presence.get(10, poison_presence.get('10', 0))
        mrr = s3.get('mrr', 0)
        avg_rank = s3.get('avg_rank_first_poison', 0)

        print(f"{retriever_name:<20} {top1:>6.1f}% {top3:>6.1f}% {top5:>6.1f}% {top10:>8.1f}% {mrr:>6.4f} {avg_rank:>7.2f}")

if __name__ == "__main__":
    main()

