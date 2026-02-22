#!/usr/bin/env python3
"""
CLI to run the GuardRAG pipeline on a batch of queries.
"""

from __future__ import annotations


import argparse
import json
from pathlib import Path
from tqdm import tqdm

from .chunking import ChunkerConfig
from .evaluation import PipelineRun, compute_asr
from .generator import GenerationConfig
from .guards import GUARD_REGISTRY
from .pipeline import Pipeline, PipelineConfig
from .query_processing import PROCESSOR_REGISTRY
from .rerankers import RERANKER_REGISTRY


def load_queries(path: str | Path, limit: int | None = None):
    path = Path(path)
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(data.get("text") or data.get("query") or data)
            if limit and len(queries) >= limit:
                break
    return queries


def main():
    parser = argparse.ArgumentParser(description="Run GuardRAG pipeline over queries.")
    parser.add_argument("--documents", required=True, help="Path to JSONL document store.")
    parser.add_argument("--queries", required=True, help="Path to JSONL queries file.")
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid", "splade"])
    parser.add_argument("--dense-model", default="intfloat/e5-large-v2", help="SentenceTransformer name.")
    parser.add_argument("--model", help="HF model path for generation (optional).")
    parser.add_argument("--top-k", type=int, default=6, help="Final number of documents passed to generation.")
    parser.add_argument("--candidate-pool", type=int, default=12, help="Initial retrieval pool size prior to re-ranking.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--separator", default="\n\n", help="Separator used when reconstructing chunk text.")
    parser.add_argument("--disable-chunking", action="store_true")
    parser.add_argument(
        "--query-processors",
        default="",
        help=f"Comma-separated processors ({', '.join(PROCESSOR_REGISTRY.keys())}).",
    )
    parser.add_argument(
        "--reranker",
        choices=list(RERANKER_REGISTRY.keys()),
        help="Optional reranker to apply after initial retrieval.",
    )
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--reranker-top-n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--guards",
        default="keyword_blocklist,secalign",
        help=f"Comma-separated guards ({', '.join(GUARD_REGISTRY.keys())}).",
    )
    parser.add_argument("--output", default="pipeline_runs.jsonl")
    args = parser.parse_args()

    generation = None
    if args.model:
        generation = GenerationConfig(
            model_name_or_path=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    retriever_kwargs = {}
    if args.retriever in {"dense", "hybrid"}:
        retriever_kwargs["model_name" if args.retriever == "dense" else "dense_model"] = args.dense_model

    chunker = None
    if not args.disable_chunking:
        chunker = ChunkerConfig(
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            separator=args.separator,
        )

    processors = [name.strip() for name in args.query_processors.split(",") if name.strip()]

    reranker_kwargs = {}
    reranker_name = args.reranker
    if reranker_name:
        reranker_kwargs = {
            "model_name": args.reranker_model,
            "top_n": args.reranker_top_n,
        }

    guard_names = [name.strip() for name in args.guards.split(",") if name.strip()]

    pipeline = Pipeline(
        PipelineConfig(
            document_path=args.documents,
            retriever=args.retriever,
            retriever_kwargs=retriever_kwargs,
            candidate_pool_size=args.candidate_pool,
            default_top_k=args.top_k,
            generation=generation,
            guard_names=guard_names or list(GUARD_REGISTRY.keys()),
            chunker=chunker,
            query_processors=processors,
            reranker=reranker_name,
            reranker_kwargs=reranker_kwargs,
        )
    )

    queries = load_queries(args.queries, limit=args.limit)
    runs = []

    with Path(args.output).open("w", encoding="utf-8") as f:
        for query in tqdm(queries, desc="Processing queries"):
            result = pipeline.run(query, top_k=args.top_k)
            f.write(json.dumps(result) + "\n")
            runs.append(
                PipelineRun(
                    original_query=result["original_query"],
                    processed_query=result["processed_query"],
                    answer=result["answer"],
                    retrieved=result["retrieved"],
                    guard_allow=result["guard_decision"]["allow"],
                )
            )

    asr = compute_asr(runs)
    print(f"Pipeline attack success rate over {len(runs)} queries: {asr:.2%}")
    print(f"Saved detailed runs to {args.output}")


if __name__ == "__main__":
    main()

