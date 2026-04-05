#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
LIVE = Path("/gscratch/uwb/gayat23/GuardRAG/results/live_judge")
META = (
    ROOT
    / "IPI_generators"
    / "ipi_nfcorpus_realistic_main_candidate_v4"
    / "nfcorpus_realistic_main_candidate_v4_attack_metadata_v2.jsonl"
)
RESULTS = ROOT / "results"

RUNS: List[Tuple[str, Path]] = [
    ("8B + BM25", LIVE / "nfcorpus_realistic_llama-3.1-8b_20260402_165257.jsonl"),
    ("8B + hybrid", LIVE / "nfcorpus_realistic_llama-3.1-8b_20260402_172744.jsonl"),
    ("70B + BM25", LIVE / "nfcorpus_realistic_llama-3.3-70b-4bit_20260403_140150.jsonl"),
    ("70B + hybrid", LIVE / "nfcorpus_realistic_llama-3.3-70b-4bit_20260403_145525.jsonl"),
    ("GPT-5.2 + BM25", LIVE / "nfcorpus_realistic_gpt-5.2_20260403_235209.jsonl"),
]

PAIRWISE = [
    ("8B + BM25", "70B + BM25"),
    ("8B + BM25", "GPT-5.2 + BM25"),
    ("70B + BM25", "GPT-5.2 + BM25"),
    ("8B + BM25", "8B + hybrid"),
    ("70B + BM25", "70B + hybrid"),
]

TECH_MD = RESULTS / "nfcorpus_v4_technique_breakdown.md"
TECH_JSON = RESULTS / "nfcorpus_v4_technique_breakdown.json"
PAIR_MD = RESULTS / "nfcorpus_v4_pairwise_stats.md"
PAIR_JSON = RESULTS / "nfcorpus_v4_pairwise_stats.json"


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def fmt_rate(num: int, den: int) -> str:
    return f"{num}/{den} ({pct(num, den):.1f}%)"


def load_techniques() -> Dict[str, str]:
    out = {}
    for row in load_jsonl(META):
        out[row["doc_id"]] = row.get("technique", "")
    return out


def load_runs() -> Dict[str, List[Dict]]:
    return {label: load_jsonl(path) for label, path in RUNS}


def summarize_techniques(run_rows: Dict[str, List[Dict]], technique_by_doc: Dict[str, str]) -> List[Dict]:
    all_query_rows = run_rows["8B + BM25"]
    by_technique: Dict[str, Dict] = {}
    counts = Counter(technique_by_doc.get(row["target_doc_id"], "unknown") for row in all_query_rows)

    for technique, total in sorted(counts.items()):
        row = {"technique": technique, "n_queries": total}
        for label, rows in run_rows.items():
            selected = [r for r in rows if technique_by_doc.get(r["target_doc_id"], "unknown") == technique]
            yes = sum(r.get("judge_verdict") is True for r in selected)
            exposure = sum(bool(r.get("any_poison_retrieved")) for r in selected)
            row[label] = {
                "judge_yes": yes,
                "judge_rate": pct(yes, len(selected)),
                "exposure": exposure,
                "exposure_rate": pct(exposure, len(selected)),
            }
        by_technique[technique] = row
    return list(by_technique.values())


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2.0 * tail)
    return p


def pairwise_stats(run_rows: Dict[str, List[Dict]]) -> List[Dict]:
    out = []
    for left, right in PAIRWISE:
        lrows = {r["query_id"]: r for r in run_rows[left]}
        rrows = {r["query_id"]: r for r in run_rows[right]}
        query_ids = sorted(set(lrows) & set(rrows))
        a = b = c = d = invalid = 0
        for qid in query_ids:
            lv = lrows[qid].get("judge_verdict")
            rv = rrows[qid].get("judge_verdict")
            if lv is None or rv is None:
                invalid += 1
                continue
            lyes = lv is True
            ryes = rv is True
            if lyes and ryes:
                a += 1
            elif lyes and not ryes:
                b += 1
            elif not lyes and ryes:
                c += 1
            else:
                d += 1
        n = a + b + c + d
        out.append(
            {
                "left": left,
                "right": right,
                "both_yes": a,
                "left_only_yes": b,
                "right_only_yes": c,
                "both_no": d,
                "paired_n": n,
                "invalid_or_missing": invalid,
                "exact_mcnemar_p": exact_mcnemar_pvalue(b, c),
            }
        )
    return out


def render_technique_md(rows: List[Dict]) -> str:
    run_labels = [label for label, _ in RUNS]
    header = ["Technique", "N"] + [f"{label} Judge" for label in run_labels] + [f"{label} Exposure" for label in run_labels]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        cells = [row["technique"], str(row["n_queries"])]
        for label in run_labels:
            cells.append(fmt_rate(row[label]["judge_yes"], row["n_queries"]))
        for label in run_labels:
            cells.append(fmt_rate(row[label]["exposure"], row["n_queries"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_pairwise_md(rows: List[Dict]) -> str:
    header = "| Left | Right | Paired N | Invalid/Missing | Both YES | Left-only YES | Right-only YES | Both NO | Exact McNemar p |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {left} | {right} | {paired_n} | {invalid_or_missing} | {both_yes} | {left_only_yes} | {right_only_yes} | {both_no} | {p:.4f} |".format(
                left=row["left"],
                right=row["right"],
                paired_n=row["paired_n"],
                invalid_or_missing=row["invalid_or_missing"],
                both_yes=row["both_yes"],
                left_only_yes=row["left_only_yes"],
                right_only_yes=row["right_only_yes"],
                both_no=row["both_no"],
                p=row["exact_mcnemar_p"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    technique_by_doc = load_techniques()
    runs = load_runs()
    tech_rows = summarize_techniques(runs, technique_by_doc)
    pair_rows = pairwise_stats(runs)

    TECH_JSON.write_text(json.dumps(tech_rows, indent=2), encoding="utf-8")
    TECH_MD.write_text(render_technique_md(tech_rows), encoding="utf-8")
    PAIR_JSON.write_text(json.dumps(pair_rows, indent=2), encoding="utf-8")
    PAIR_MD.write_text(render_pairwise_md(pair_rows), encoding="utf-8")

    print(f"Wrote {TECH_MD}")
    print(f"Wrote {TECH_JSON}")
    print(f"Wrote {PAIR_MD}")
    print(f"Wrote {PAIR_JSON}")


if __name__ == "__main__":
    main()
