#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable


@dataclass(frozen=True)
class DomainConfig:
    name: str
    queries: str
    clean_corpus: str
    ipi_metadata: str
    ipi_poisoned: str
    ipi_mixed: str


DOMAINS: dict[str, DomainConfig] = {
    "nfcorpus": DomainConfig(
        name="nfcorpus",
        queries="data/corpus/beir/nfcorpus/queries.jsonl",
        clean_corpus="data/corpus/beir/nfcorpus/corpus.jsonl",
        ipi_metadata="IPI_generators/ipi_nfcorpus/nfcorpus_ipi_metadata_v2.jsonl",
        ipi_poisoned="IPI_generators/ipi_nfcorpus/nfcorpus_ipi_poisoned_v2.jsonl",
        ipi_mixed="IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl",
    ),
    "fiqa": DomainConfig(
        name="fiqa",
        queries="data/corpus/beir/fiqa/queries.jsonl",
        clean_corpus="data/corpus/beir/fiqa/corpus.jsonl",
        ipi_metadata="IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl",
        ipi_poisoned="IPI_generators/ipi_fiqa/fiqa_ipi_poisoned_v2.jsonl",
        ipi_mixed="IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl",
    ),
    "scifact": DomainConfig(
        name="scifact",
        queries="data/corpus/beir/scifact/queries.jsonl",
        clean_corpus="data/corpus/beir/scifact/corpus.jsonl",
        ipi_metadata="IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl",
        ipi_poisoned="IPI_generators/ipi_scifact/scifact_ipi_poisoned_v2.jsonl",
        ipi_mixed="IPI_generators/ipi_scifact/scifact_ipi_mixed_v2.jsonl",
    ),
    "hotpotqa": DomainConfig(
        name="hotpotqa",
        queries="data/corpus/beir/hotpotqa/queries.jsonl",
        clean_corpus="data/corpus/beir/hotpotqa/corpus.jsonl",
        ipi_metadata="IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl",
        ipi_poisoned="IPI_generators/ipi_hotpotqa/hotpotqa_ipi_poisoned_v2.jsonl",
        ipi_mixed="IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl",
    ),
    "nq": DomainConfig(
        name="nq",
        queries="data/corpus/beir/nq/queries.jsonl",
        clean_corpus="data/corpus/beir/nq/corpus.jsonl",
        ipi_metadata="IPI_generators/ipi_nq/nq_ipi_metadata_v2.jsonl",
        ipi_poisoned="IPI_generators/ipi_nq/nq_ipi_poisoned_v2.jsonl",
        ipi_mixed="IPI_generators/ipi_nq/nq_ipi_mixed_v2.jsonl",
    ),
}


def _ensure_paths(paths: Iterable[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--domains",
        default="all",
        help="Comma-separated list from: " + ",".join(DOMAINS.keys()) + " (default: all)",
    )
    p.add_argument("--out-dir", default="data/preference", help="Output root dir")
    p.add_argument("--retriever", default="dense", choices=["bm25", "dense", "hybrid", "splade"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dense-model", default="intfloat/e5-large-v2")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--use-existing-responses", action="store_true", default=False)
    p.add_argument("--generator-model", default=None, help="Required unless --use-existing-responses")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--pairing-strategy", default="score", choices=["score", "prefix"])
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--semantic-sim-model", default="intfloat/e5-large-v2")
    p.add_argument("--novelty-boost", type=float, default=0.2)
    p.add_argument("--disable-risk-weighting", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    if not args.use_existing_responses and not args.generator_model:
        raise SystemExit("--generator-model is required unless --use-existing-responses is set")

    if args.domains == "all":
        domain_names = list(DOMAINS.keys())
    else:
        domain_names = [d.strip() for d in args.domains.split(",") if d.strip()]

    for name in domain_names:
        if name not in DOMAINS:
            raise SystemExit(f"Unknown domain: {name}")

    out_root = Path(args.out_dir)
    for name in domain_names:
        cfg = DOMAINS[name]
        _ensure_paths([cfg.queries, cfg.clean_corpus, cfg.ipi_metadata, cfg.ipi_poisoned, cfg.ipi_mixed])
        domain_out = out_root / cfg.name
        domain_out.mkdir(parents=True, exist_ok=True)

        common = [
            sys.executable,
            "DPO/data/build_preference_pairs.py",
            "--queries",
            cfg.queries,
            "--retriever",
            args.retriever,
            "--top-k",
            str(args.top_k),
            "--dense-model",
            args.dense_model,
            "--alpha",
            str(args.alpha),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--pairing-strategy",
            args.pairing_strategy,
            "--candidates",
            str(args.candidates),
            "--semantic-sim-model",
            args.semantic_sim_model,
            "--novelty-boost",
            str(args.novelty_boost),
        ]
        if args.limit is not None:
            common += ["--limit", str(args.limit)]
        if args.use_existing_responses:
            common += ["--use-existing-responses"]
        else:
            common += ["--generator-model", args.generator_model]
        if args.disable_risk_weighting:
            common += ["--disable-risk-weighting"]

        _run(
            common
            + [
                "--corpus",
                cfg.clean_corpus,
                "--out",
                str(domain_out / "clean.jsonl"),
                "--pair-type",
                "utility",
            ],
            dry_run=args.dry_run,
        )

        _run(
            common
            + [
                "--corpus",
                cfg.ipi_mixed,
                "--metadata",
                cfg.ipi_metadata,
                "--out",
                str(domain_out / "mixed.jsonl"),
                "--pair-type",
                "auto",
            ],
            dry_run=args.dry_run,
        )

        _run(
            common
            + [
                "--corpus",
                cfg.ipi_poisoned,
                "--metadata",
                cfg.ipi_metadata,
                "--out",
                str(domain_out / "poisoned.jsonl"),
                "--pair-type",
                "security",
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
