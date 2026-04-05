#!/usr/bin/env python3
"""
Causal document-selection proof for Promptfoo-exported GuardRAG runs.

This is the Promptfoo-aware sibling of `run_selection_proof.py`. It takes a
completed Promptfoo JSON export, extracts the saved outputs and retrieved
documents, and then performs document-level counterfactuals:

1. Remove the analysed poisoned doc from the retrieved context
2. Swap the analysed poisoned doc with its clean original, when available

If the answer only follows the hidden directive when the poisoned document is
present, that is stronger evidence that the model selected and used the
poisoned document rather than merely retrieving it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.live_judge_eval import (
    build_judge_generator,
    build_pipeline,
    build_shared_generator,
    call_judge,
    load_metadata_lookup,
    load_poisoned_texts,
    resolve_cfg,
)


def bootstrap_env() -> None:
    os.environ.setdefault("HF_HOME", "/gscratch/uwb/gayat23/hf_cache")
    token_path = Path("/gscratch/uwb/gayat23/hf_cache/token")
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    openai_key_path = Path("/gscratch/uwb/gayat23/.openai_api_key")
    if openai_key_path.exists():
        key = openai_key_path.read_text(encoding="utf-8").strip()
        if key:
            os.environ.setdefault("OPENAI_API_KEY", key)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run document-selection ablations for Promptfoo GuardRAG outputs.")
    ap.add_argument("--results-file", required=True, help="Completed Promptfoo JSON export.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--model", default="", help="Generator model key. If omitted, infer from Promptfoo metadata.")
    ap.add_argument("--judge", required=True, help="Judge model key.")
    ap.add_argument("--sample", type=int, default=0, help="Optional cap on analysed rows.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--only-target-retrieved",
        action="store_true",
        help="Only analyse rows where the target poisoned doc was actually retrieved.",
    )
    ap.add_argument(
        "--regenerate-full",
        action="store_true",
        help="Regenerate the full answer instead of trusting the saved Promptfoo output.",
    )
    ap.add_argument("--queries-file")
    ap.add_argument("--merged-path")
    ap.add_argument("--clean-path")
    ap.add_argument("--metadata-path")
    ap.add_argument("--poisoned-path")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    return ap.parse_args()


def load_promptfoo_rows(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    rows = results.get("results", [])
    normalized: List[Dict] = []
    for row in rows:
        response = row.get("response") or {}
        metadata = response.get("metadata") or row.get("metadata") or {}
        vars_block = row.get("vars") or (row.get("testCase") or {}).get("vars") or {}
        query = metadata.get("query") or vars_block.get("query") or ""
        target_doc_id = metadata.get("target_doc_id") or vars_block.get("target_doc_id") or ""
        if not query or not target_doc_id:
            continue
        normalized.append(
            {
                "query": query,
                "query_id": vars_block.get("session_id", ""),
                "source_query_id": metadata.get("source_query_id") or vars_block.get("source_query_id", ""),
                "analysis_doc_id": target_doc_id,
                "target_doc_id": target_doc_id,
                "original_id": metadata.get("original_id") or vars_block.get("original_id", ""),
                "answer": response.get("output", ""),
                "retrieved": metadata.get("retrieved") or [],
                "any_poison_retrieved": bool(metadata.get("any_poison_retrieved")),
                "target_retrieved": bool(metadata.get("target_retrieved")),
                "target_rank": metadata.get("target_rank", -1),
                "retriever": metadata.get("retriever", ""),
                "rag_model": metadata.get("rag_model", ""),
            }
        )
    return normalized


def main() -> None:
    args = parse_args()
    bootstrap_env()

    results_path = Path(args.results_file)
    output_path = Path(args.output)

    cfg = resolve_cfg(
        args.corpus,
        args.tier,
        merged_path=args.merged_path,
        clean_path=args.clean_path,
        queries_path=args.queries_file,
        metadata_path=args.metadata_path,
        poisoned_path=args.poisoned_path,
    )

    rows = load_promptfoo_rows(results_path)
    rows = [r for r in rows if r.get("analysis_doc_id") and r.get("any_poison_retrieved")]
    if args.only_target_retrieved:
        rows = [r for r in rows if r.get("target_retrieved")]
    if args.sample:
        rows = rows[: args.sample]

    if not rows:
        raise SystemExit(f"No analysable Promptfoo rows found in {results_path}")

    model_key = args.model or rows[0].get("rag_model") or ""
    if not model_key:
        raise SystemExit("Could not infer model key from Promptfoo metadata. Pass --model explicitly.")

    print(f"Loaded {len(rows)} Promptfoo rows from {results_path}")
    print(f"Using generation model {model_key}")

    metadata_lookup = load_metadata_lookup(cfg["metadata"])
    poisoned_texts = load_poisoned_texts(cfg["poisoned"])

    shared_generator = build_shared_generator(model_key)
    retriever_name = rows[0].get("retriever") or "bm25"
    poisoned_pipeline = build_pipeline(
        cfg["merged"],
        model_key,
        retriever=retriever_name,
        shared_generator=shared_generator,
        prompt_profile="attack_eval",
    )
    clean_pipeline = build_pipeline(
        cfg["clean"],
        model_key,
        retriever="bm25",
        shared_generator=shared_generator,
        prompt_profile="attack_eval",
    )
    judge_gen = build_judge_generator(args.judge)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = Counter()

    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            query = row["query"]
            analysis_doc_id = row["analysis_doc_id"]
            full_answer = row.get("answer", "")

            docs = []
            used_saved_retrieval = False
            saved_retrieved = row.get("retrieved") or []
            if saved_retrieved:
                missing_saved_docs: List[str] = []
                for item in saved_retrieved:
                    doc_id = item.get("doc_id")
                    if not doc_id:
                        continue
                    doc = poisoned_pipeline.store.get(doc_id) or clean_pipeline.store.get(doc_id)
                    if doc is None:
                        missing_saved_docs.append(doc_id)
                        continue
                    docs.append(doc)
                if docs and not missing_saved_docs:
                    used_saved_retrieval = True

            retrieval = None
            if not used_saved_retrieval:
                retrieval = poisoned_pipeline.retrieve_context(query, top_k=args.top_k)
                raw_retrieved = retrieval.get("_raw_retrieved", [])
                docs = [doc for doc, _score in raw_retrieved]

            selected_doc = None
            for doc in docs:
                if doc.doc_id == analysis_doc_id:
                    selected_doc = doc
                    break

            if selected_doc is None:
                result = {
                    "query": query,
                    "source_query_id": row.get("source_query_id", ""),
                    "analysis_doc_id": analysis_doc_id,
                    "status": "analysis_doc_not_retrieved",
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                summary["analysis_doc_not_retrieved"] += 1
                continue

            if args.regenerate_full:
                full_context = (
                    poisoned_pipeline._format_context(docs)
                    if used_saved_retrieval
                    else retrieval["context"]
                )
                full_answer = poisoned_pipeline.answer_from_context(query, full_context)["answer"]

            meta = metadata_lookup.get(analysis_doc_id, {})
            original_id = meta.get("original_id", row.get("original_id", ""))
            span_start = int(meta.get("span_start", 0) or 0)
            span_end = int(meta.get("span_end", 0) or 0)
            directive = ""
            full_text = poisoned_texts.get(analysis_doc_id, "")
            if full_text and span_end > span_start:
                directive = full_text[span_start:span_end]

            ablated_docs = [doc for doc in docs if doc.doc_id != analysis_doc_id]
            ablated_context = poisoned_pipeline._format_context(ablated_docs)
            ablated_answer = poisoned_pipeline.answer_from_context(query, ablated_context)["answer"]
            ablation_verdict, ablation_reason = call_judge(
                judge_gen,
                directive,
                full_answer,
                clean_answer=ablated_answer,
            )

            swapped_answer = None
            swap_verdict = None
            swap_reason = None
            clean_original = clean_pipeline.store.get(original_id) if original_id else None
            if clean_original is not None:
                swapped_docs = [
                    clean_original if doc.doc_id == analysis_doc_id else doc
                    for doc in docs
                ]
                swapped_context = poisoned_pipeline._format_context(swapped_docs)
                swapped_answer = poisoned_pipeline.answer_from_context(query, swapped_context)["answer"]
                swap_verdict, swap_reason = call_judge(
                    judge_gen,
                    directive,
                    full_answer,
                    clean_answer=swapped_answer,
                )

            selection_proved = bool(ablation_verdict) or bool(swap_verdict)
            status = "selection_proved" if selection_proved else "selection_not_proved"
            summary[status] += 1
            if ablation_verdict:
                summary["proved_by_removal"] += 1
            if swap_verdict:
                summary["proved_by_swap"] += 1

            result = {
                "query": query,
                "source_query_id": row.get("source_query_id", ""),
                "analysis_doc_id": analysis_doc_id,
                "target_doc_id": row.get("target_doc_id", ""),
                "original_id": original_id,
                "technique": meta.get("technique", ""),
                "attack_family": meta.get("attack_family", ""),
                "full_answer": full_answer,
                "ablated_answer": ablated_answer,
                "ablation_verdict": ablation_verdict,
                "ablation_reason": ablation_reason,
                "swapped_answer": swapped_answer,
                "swap_verdict": swap_verdict,
                "swap_reason": swap_reason,
                "selection_proved": selection_proved,
                "used_saved_retrieval": used_saved_retrieval,
                "status": status,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{idx}/{len(rows)}] {query[:80]} | "
                f"removal={ablation_verdict} swap={swap_verdict} -> {status}"
            )

    summary_path = output_path.with_suffix(".summary.json")
    summary_obj = {
        "input_results_file": str(results_path),
        "output_jsonl": str(output_path),
        "analysed_rows": len(rows),
        "model": model_key,
        "retriever": retriever_name,
        "summary_counts": dict(summary),
    }
    summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    print(json.dumps(summary_obj, indent=2))


if __name__ == "__main__":
    main()
