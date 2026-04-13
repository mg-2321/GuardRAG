#!/usr/bin/env python3
"""
Run component-level RAG traces over a benchmark query set.

Author: Gayatri Malladi

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
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:  # pragma: no cover
    from rag_pipeline_components.pipeline import Pipeline

_RETRIEVER_CACHE = Path("/gscratch/uwb/gayat23/GuardRAG/cache/retriever_embeddings")


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


def _retriever_prefers_gpu() -> bool:
    """Use GPU for dense retrieval whenever this process has a GPU allocation."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return bool(cuda_visible and cuda_visible != "NoDevFiles")


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
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Token chunk size for document splitting (0 = no chunking, use full docs).",
    )
    ap.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Token overlap between consecutive chunks (default: 200).",
    )
    ap.add_argument("--output", required=True, help="Per-query JSONL trace output.")
    return ap.parse_args()


def _build_pipeline(args: argparse.Namespace, cfg: Dict[str, Path]) -> "Pipeline":
    from rag_pipeline_components.pipeline import Pipeline, PipelineConfig

    if args.model:
        from evaluation.live_judge_eval import build_generation_config, build_shared_generator

        shared_generator = build_shared_generator(args.model)
        gen_config = None if shared_generator is not None else build_generation_config(args.model)
    else:
        shared_generator = None
        gen_config = None

    retriever_kwargs: Dict = {}
    if args.retriever == "dense_e5":
        retriever_kwargs = {
            "model_name": "intfloat/e5-large-v2",
            "cache_dir": str(_RETRIEVER_CACHE),
            "device": "cuda" if _retriever_prefers_gpu() else None,
            "require_gpu": _retriever_prefers_gpu(),
            "index_backend": os.environ.get("GUARDRAG_VECTOR_BACKEND", "auto"),
        }
    elif args.retriever == "hybrid":
        retriever_kwargs = {
            "alpha": 0.8,
            "dense_model": "intfloat/e5-large-v2",
            "cache_dir": str(_RETRIEVER_CACHE),
            "device": "cuda" if _retriever_prefers_gpu() else None,
            "require_gpu": _retriever_prefers_gpu(),
            "index_backend": os.environ.get("GUARDRAG_VECTOR_BACKEND", "auto"),
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

    chunker_cfg = None
    if args.chunk_size > 0:
        from rag_pipeline_components.chunking import ChunkerConfig
        chunker_cfg = ChunkerConfig(
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )

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
            document_store_mode="auto",
            chunker=chunker_cfg,
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
    print(
        json.dumps(
            {
                "event": "component_study_start",
                "corpus": args.corpus,
                "tier": args.tier,
                "retriever": args.retriever,
                "reranker": args.reranker.strip() or "",
                "model": args.model,
                "vector_backend": os.environ.get("GUARDRAG_VECTOR_BACKEND", "auto"),
                "limit": args.limit,
                "top_k": args.top_k,
                "candidate_pool": max(args.candidate_pool, args.top_k),
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap if args.chunk_size > 0 else 0,
            }
        ),
        flush=True,
    )
    from evaluation.live_judge_eval import load_metadata_lookup, load_queries, resolve_cfg

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
    print(
        json.dumps(
            {
                "event": "component_study_inputs_ready",
                "queries_path": str(cfg["queries"]),
                "merged_path": str(cfg["merged"]),
                "metadata_path": str(cfg["metadata"]),
                "query_count": len(queries),
                "metadata_count": len(metadata_lookup),
            }
        ),
        flush=True,
    )
    pipeline = _build_pipeline(args, cfg)
    print(
        json.dumps(
            {
                "event": "component_study_pipeline_ready",
                "retriever": args.retriever,
                "reranker": args.reranker.strip() or "",
                "model": args.model,
            }
        ),
        flush=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = Counter()
    by_family: Dict[str, Counter] = defaultdict(Counter)
    by_technique: Dict[str, Counter] = defaultdict(Counter)
    target_ranks: List[int] = []
    poison_counts: List[int] = []
    answer_lengths: List[int] = []

    # ── Resume support: reload already-completed rows ──────────────────────
    n_done = 0
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", encoding="utf-8") as _f:
            for line in _f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_done += 1
                summary["n"] += 1
                any_p = prev.get("any_poison_retrieved", False)
                if any_p:
                    summary["poison_exposure_count"] += 1
                poison_counts.append(prev.get("poison_count_final_topk", 0))
                t_ret = prev.get("target_retrieved", False)
                t_top1 = prev.get("target_top1", False)
                top1_p = prev.get("top1_poisoned", False)
                if t_ret:
                    summary["target_retrieved_count"] += 1
                    r = prev.get("target_rank", -1)
                    if r > 0:
                        target_ranks.append(r)
                if t_top1:
                    summary["target_top1_count"] += 1
                if top1_p:
                    summary["top1_poisoned_count"] += 1
                fam = prev.get("attack_family", "")
                tec = prev.get("technique", "")
                by_family[fam]["n"] += 1
                by_technique[tec]["n"] += 1
                if t_ret:
                    by_family[fam]["target_retrieved"] += 1
                    by_technique[tec]["target_retrieved"] += 1
                if t_top1:
                    by_family[fam]["target_top1"] += 1
                    by_technique[tec]["target_top1"] += 1
                al = prev.get("answer_length", 0)
                if al:
                    answer_lengths.append(al)
        if n_done:
            print(json.dumps({"event": "resume", "rows_already_done": n_done}), flush=True)

    with output_path.open("a", encoding="utf-8") as out:
        for idx, row in enumerate(_iter_rows(queries, args.limit), start=1):
            if idx <= n_done:
                continue
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
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap if args.chunk_size > 0 else 0,
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
