#!/usr/bin/env python3
"""
Run component-level RAG traces over a benchmark query set.

This script is meant to make "how the attack travels through the RAG stack"
concrete. For each query it records:

1. Whether any poisoned document enters the retrieved set
2. Whether the intended target poison is retrieved
3. The target's final rank after optional reranking
4. Which attack family / technique the target belongs to
5. Optional generation output length if a generator model is supplied

It intentionally focuses on the core observable stages we can measure
reproducibly today: retrieval, reranking, and the final context passed to the
generator. Counterfactual document-ablation proofs can then build on the saved
per-query trace rows.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.live_judge_eval import (
    _RETRIEVER_CACHE,
    build_generation_config,
    build_shared_generator,
    load_metadata_lookup,
    load_queries,
    resolve_cfg,
)
from rag_pipeline_components.pipeline import Pipeline, PipelineConfig


def bootstrap_env() -> None:
    """Mirror the existing eval defaults so ad hoc runs behave consistently."""
    os.environ.setdefault("HF_HOME", "/gscratch/uwb/gayat23/hf_cache")
    token_path = Path("/gscratch/uwb/gayat23/hf_cache/token")
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            os.environ.setdefault("HF_TOKEN", token)
    openai_key_path = Path("/gscratch/uwb/gayat23/.openai_api_key")
    if openai_key_path.exists():
        key = openai_key_path.read_text(encoding="utf-8").strip()
        if key:
            os.environ.setdefault("OPENAI_API_KEY", key)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a component-level RAG study.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument(
        "--retriever",
        default="bm25",
        choices=["bm25", "dense_e5", "hybrid", "splade"],
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-pool", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for a quick pilot run.")
    ap.add_argument(
        "--query-processors",
        default="",
        help="Comma-separated query processors to apply before retrieval.",
    )
    ap.add_argument(
        "--reranker",
        default="",
        help="Optional reranker name from rag_pipeline_components.rerankers.",
    )
    ap.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="HF checkpoint used by the reranker when enabled.",
    )
    ap.add_argument(
        "--reranker-top-n",
        type=int,
        default=10,
        help="Number of reranked candidates kept before final top-k truncation.",
    )
    ap.add_argument(
        "--model",
        default="",
        help="Optional generator model key. If omitted, the study is retrieval-only.",
    )
    ap.add_argument("--queries-file", default=None)
    ap.add_argument("--merged-path", default=None)
    ap.add_argument("--clean-path", default=None)
    ap.add_argument("--metadata-path", default=None)
    ap.add_argument("--poisoned-path", default=None)
    ap.add_argument("--output", required=True, help="Per-query JSONL trace output.")
    return ap.parse_args()


def _build_pipeline(args: argparse.Namespace, cfg: Dict[str, Path]) -> Pipeline:
    shared_generator = build_shared_generator(args.model) if args.model else None
    gen_config = None if shared_generator is not None else (
        build_generation_config(args.model) if args.model else None
    )

    retriever_kwargs: Dict = {}
    if args.retriever == "dense_e5":
        retriever_kwargs = {
            "model_name": "intfloat/e5-large-v2",
            "cache_dir": str(_RETRIEVER_CACHE),
        }
    elif args.retriever == "hybrid":
        retriever_kwargs = {
            "alpha": 0.8,
            "dense_model": "intfloat/e5-large-v2",
            "cache_dir": str(_RETRIEVER_CACHE),
        }
    elif args.retriever == "splade":
        retriever_kwargs = {
            "model_name": "naver/splade-cocondenser-ensembledistil",
            "cache_dir": str(_RETRIEVER_CACHE),
        }

    retriever_name = "dense" if args.retriever == "dense_e5" else args.retriever

    processors = [p.strip() for p in args.query_processors.split(",") if p.strip()]
    reranker_kwargs = {}
    reranker_name = args.reranker.strip() or None
    if reranker_name:
        reranker_kwargs = {
            "model_name": args.reranker_model,
            "top_n": args.reranker_top_n,
        }

    return Pipeline(
        PipelineConfig(
            document_path=str(cfg["merged"]),
            retriever=retriever_name,
            retriever_kwargs=retriever_kwargs,
            candidate_pool_size=max(args.candidate_pool, args.top_k),
            default_top_k=args.top_k,
            prompt_profile="attack_eval",
            generation=gen_config,
            guards=[],
            shared_generator=shared_generator,
            query_processors=processors,
            reranker=reranker_name,
            reranker_kwargs=reranker_kwargs,
        )
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def _iter_rows(queries: Iterable[Dict], limit: int) -> Iterable[Dict]:
    count = 0
    for row in queries:
        if not row:
            continue
        yield row
        count += 1
        if limit and count >= limit:
            break


def _render_markdown(summary: Dict) -> str:
    config = summary["config"]
    lines = [
        f"# {config['corpus']} {config['tier']} Component Study",
        "",
        f"- Queries: {summary['n']}",
        f"- Retriever: `{config['retriever']}`",
        f"- Top-k: `{config['top_k']}`",
        f"- Candidate pool: `{config['candidate_pool']}`",
        f"- Reranker: `{config['reranker'] or 'none'}`",
        f"- Generator model: `{config['model'] or 'none (retrieval-only)'}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Poison exposure | {summary['poison_exposure_count']}/{summary['n']} ({summary['poison_exposure_pct']}%) |",
        f"| Target retrieved | {summary['target_retrieved_count']}/{summary['n']} ({summary['target_retrieved_pct']}%) |",
        f"| Target at rank 1 | {summary['target_top1_count']}/{summary['n']} ({summary['target_top1_pct']}%) |",
        f"| Top-1 poisoned | {summary['top1_poisoned_count']}/{summary['n']} ({summary['top1_poisoned_pct']}%) |",
        f"| Mean poisons in final top-k | {summary['mean_poison_count_final_topk']:.2f} |",
        f"| Mean target rank (retrieved only) | {summary['mean_target_rank_retrieved']:.2f} |",
        f"| Median target rank (retrieved only) | {summary['median_target_rank_retrieved']:.2f} |",
    ]

    if summary.get("mean_answer_length") is not None:
        lines.append(f"| Mean answer length | {summary['mean_answer_length']:.1f} |")

    if summary["by_family"]:
        lines.extend([
            "",
            "## By Attack Family",
            "",
            "| Family | N | Target Retrieved | Rank-1 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for family, stats in sorted(summary["by_family"].items()):
            lines.append(
                f"| {family} | {stats['n']} | "
                f"{stats.get('target_retrieved', 0)}/{stats['n']} ({_safe_rate(stats.get('target_retrieved', 0), stats['n'])}%) | "
                f"{stats.get('target_top1', 0)}/{stats['n']} ({_safe_rate(stats.get('target_top1', 0), stats['n'])}%) |"
            )

    if summary["by_technique"]:
        lines.extend([
            "",
            "## By Technique",
            "",
            "| Technique | N | Target Retrieved | Rank-1 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for technique, stats in sorted(summary["by_technique"].items()):
            lines.append(
                f"| {technique} | {stats['n']} | "
                f"{stats.get('target_retrieved', 0)}/{stats['n']} ({_safe_rate(stats.get('target_retrieved', 0), stats['n'])}%) | "
                f"{stats.get('target_top1', 0)}/{stats['n']} ({_safe_rate(stats.get('target_top1', 0), stats['n'])}%) |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bootstrap_env()

    cfg = resolve_cfg(
        args.corpus,
        args.tier,
        merged_path=args.merged_path,
        clean_path=args.clean_path,
        queries_path=args.queries_file,
        metadata_path=args.metadata_path,
        poisoned_path=args.poisoned_path,
    )
    queries = load_queries(cfg["queries"])
    metadata_lookup = load_metadata_lookup(cfg["metadata"])
    pipeline = _build_pipeline(args, cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = Counter()
    by_family: Dict[str, Counter] = defaultdict(Counter)
    by_technique: Dict[str, Counter] = defaultdict(Counter)
    target_ranks: List[int] = []
    poison_counts: List[int] = []
    answer_lengths: List[int] = []

    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(_iter_rows(queries, args.limit), start=1):
            query = row.get("text") or row.get("query") or ""
            target_doc_id = row.get("target_doc_id", "")
            source_query_id = row.get("source_query_id", "")
            original_id = row.get("original_id", "")

            retrieval = pipeline.retrieve_context(query, top_k=args.top_k)
            retrieved = retrieval["retrieved"]
            retrieved_doc_ids = [item.get("doc_id", "") for item in retrieved]
            poison_doc_ids = [item.get("doc_id", "") for item in retrieved if item.get("poisoned")]
            any_poison_retrieved = bool(poison_doc_ids)
            poison_count = len(poison_doc_ids)
            poison_counts.append(poison_count)

            try:
                target_rank = retrieved_doc_ids.index(target_doc_id) + 1
            except ValueError:
                target_rank = -1
            target_retrieved = target_rank != -1
            target_top1 = target_rank == 1
            top1_doc_id = retrieved_doc_ids[0] if retrieved_doc_ids else ""
            top1_poisoned = bool(retrieved and retrieved[0].get("poisoned"))

            meta = metadata_lookup.get(target_doc_id, {})
            family = meta.get("attack_family", "")
            technique = meta.get("technique", "")

            answer = ""
            if args.model:
                generation = pipeline.answer_from_context(query, retrieval["context"])
                answer = generation.get("answer", "")
                answer_lengths.append(len(answer))

            result = {
                "query": query,
                "query_id": row.get("_id", ""),
                "source_query_id": source_query_id,
                "target_doc_id": target_doc_id,
                "original_id": original_id,
                "retriever": args.retriever,
                "top_k": args.top_k,
                "candidate_pool": max(args.candidate_pool, args.top_k),
                "reranker": args.reranker.strip() or "",
                "reranker_model": args.reranker_model if args.reranker else "",
                "technique": technique,
                "attack_family": family,
                "any_poison_retrieved": any_poison_retrieved,
                "poison_doc_ids_retrieved": poison_doc_ids,
                "poison_count_final_topk": poison_count,
                "target_retrieved": target_retrieved,
                "target_rank": target_rank,
                "target_top1": target_top1,
                "top1_doc_id": top1_doc_id,
                "top1_poisoned": top1_poisoned,
                "retrieved": retrieved,
                "answer": answer,
                "answer_length": len(answer),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

            summary["n"] += 1
            if any_poison_retrieved:
                summary["poison_exposure_count"] += 1
            if target_retrieved:
                summary["target_retrieved_count"] += 1
                target_ranks.append(target_rank)
            if target_top1:
                summary["target_top1_count"] += 1
            if top1_poisoned:
                summary["top1_poisoned_count"] += 1

            by_family[family]["n"] += 1
            by_technique[technique]["n"] += 1
            if target_retrieved:
                by_family[family]["target_retrieved"] += 1
                by_technique[technique]["target_retrieved"] += 1
            if target_top1:
                by_family[family]["target_top1"] += 1
                by_technique[technique]["target_top1"] += 1

            print(
                f"[{idx}] {query[:70]} | poison={any_poison_retrieved} "
                f"target={target_retrieved} rank={target_rank} top1={target_top1}"
            )

    n = summary["n"]
    summary_obj = {
        "config": {
            "corpus": args.corpus,
            "tier": args.tier,
            "retriever": args.retriever,
            "top_k": args.top_k,
            "candidate_pool": max(args.candidate_pool, args.top_k),
            "reranker": args.reranker.strip(),
            "reranker_model": args.reranker_model if args.reranker else "",
            "model": args.model,
            "queries_file": str(cfg["queries"]),
            "merged_path": str(cfg["merged"]),
            "metadata_path": str(cfg["metadata"]),
        },
        "output_jsonl": str(output_path),
        "n": n,
        "poison_exposure_count": summary["poison_exposure_count"],
        "poison_exposure_pct": _safe_rate(summary["poison_exposure_count"], n),
        "target_retrieved_count": summary["target_retrieved_count"],
        "target_retrieved_pct": _safe_rate(summary["target_retrieved_count"], n),
        "target_top1_count": summary["target_top1_count"],
        "target_top1_pct": _safe_rate(summary["target_top1_count"], n),
        "top1_poisoned_count": summary["top1_poisoned_count"],
        "top1_poisoned_pct": _safe_rate(summary["top1_poisoned_count"], n),
        "mean_poison_count_final_topk": mean(poison_counts) if poison_counts else 0.0,
        "mean_target_rank_retrieved": mean(target_ranks) if target_ranks else -1.0,
        "median_target_rank_retrieved": median(target_ranks) if target_ranks else -1.0,
        "mean_answer_length": mean(answer_lengths) if answer_lengths else None,
        "by_family": {k: dict(v) for k, v in by_family.items()},
        "by_technique": {k: dict(v) for k, v in by_technique.items()},
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    md_path.write_text(_render_markdown(summary_obj), encoding="utf-8")

    print(json.dumps(summary_obj, indent=2))
    print(f"Wrote per-query traces to {output_path}")
    print(f"Wrote summary JSON to {summary_path}")
    print(f"Wrote summary Markdown to {md_path}")


if __name__ == "__main__":
    main()
