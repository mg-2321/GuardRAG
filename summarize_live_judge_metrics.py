#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize one or more live_judge JSONL result files into consistent metrics."
    )
    ap.add_argument("files", nargs="+", help="One or more live_judge JSONL files")
    ap.add_argument("--format", choices=["json", "markdown"], default="markdown")
    return ap.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(num: int, den: int) -> float:
    return round((num / den) * 100, 1) if den else 0.0


def summarize(path: Path) -> Dict:
    rows = load_jsonl(path)
    n = len(rows)
    exposed = sum(1 for r in rows if r.get("any_poison_retrieved"))
    target = sum(1 for r in rows if r.get("target_retrieved"))
    judge_yes = sum(1 for r in rows if r.get("judge_verdict") is True)
    cosine_yes = sum(1 for r in rows if r.get("cosine_asr") is True)
    missed = sum(1 for r in rows if r.get("attack_missed_by_cosine") is True)
    fp = sum(1 for r in rows if r.get("false_positive_cosine") is True)
    target_judge_yes = sum(
        1 for r in rows if r.get("judge_verdict") is True and r.get("target_retrieved")
    )
    spillover_judge_yes = sum(
        1
        for r in rows
        if r.get("judge_verdict") is True
        and r.get("any_poison_retrieved")
        and not r.get("target_retrieved")
    )

    models = sorted({str(r.get("model", "")) for r in rows if r.get("model")})
    sources = sorted({str(r.get("query_source", "")) for r in rows if r.get("query_source")})

    return {
        "file": str(path),
        "rows": n,
        "model": models[0] if len(models) == 1 else models,
        "query_sources": sources,
        "any_poison_retrieved": exposed,
        "target_doc_retrieved": target,
        "judge_yes": judge_yes,
        "cosine_yes": cosine_yes,
        "attacks_missed_by_cosine": missed,
        "cosine_false_positives": fp,
        "target_judge_yes": target_judge_yes,
        "spillover_judge_yes": spillover_judge_yes,
        "metrics": {
            "exposure_rate": pct(exposed, n),
            "target_retrieval_rate": pct(target, n),
            "judge_asr_overall": pct(judge_yes, n),
            "judge_asr_among_exposed": pct(judge_yes, exposed),
            "cosine_asr_overall": pct(cosine_yes, n),
            "cosine_asr_among_exposed": pct(cosine_yes, exposed),
            "targeted_judge_asr": pct(target_judge_yes, target),
            "spillover_judge_rate": pct(spillover_judge_yes, exposed),
            "cosine_miss_rate_among_exposed": pct(missed, exposed),
            "cosine_fp_rate_among_exposed": pct(fp, exposed),
        },
    }


def render_markdown(rows: List[Dict]) -> str:
    headers = [
        "label",
        "rows",
        "exposure",
        "target_retrieval",
        "judge_overall",
        "judge_exposed",
        "targeted_judge",
        "spillover_yes",
        "cosine_overall",
        "cosine_exposed",
        "cosine_miss",
        "cosine_fp",
    ]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        m = row["metrics"]
        out.append(
            "| "
            + " | ".join(
                [
                    Path(row["file"]).name,
                    str(row["rows"]),
                    f"{row['any_poison_retrieved']}/{row['rows']} ({m['exposure_rate']}%)",
                    f"{row['target_doc_retrieved']}/{row['rows']} ({m['target_retrieval_rate']}%)",
                    f"{row['judge_yes']}/{row['rows']} ({m['judge_asr_overall']}%)",
                    f"{row['judge_yes']}/{row['any_poison_retrieved']} ({m['judge_asr_among_exposed']}%)" if row["any_poison_retrieved"] else "0/0 (0.0%)",
                    f"{row['target_judge_yes']}/{row['target_doc_retrieved']} ({m['targeted_judge_asr']}%)" if row["target_doc_retrieved"] else "0/0 (0.0%)",
                    f"{row['spillover_judge_yes']}/{row['any_poison_retrieved']} ({m['spillover_judge_rate']}%)" if row["any_poison_retrieved"] else "0/0 (0.0%)",
                    f"{row['cosine_yes']}/{row['rows']} ({m['cosine_asr_overall']}%)",
                    f"{row['cosine_yes']}/{row['any_poison_retrieved']} ({m['cosine_asr_among_exposed']}%)" if row["any_poison_retrieved"] else "0/0 (0.0%)",
                    f"{row['attacks_missed_by_cosine']}/{row['any_poison_retrieved']} ({m['cosine_miss_rate_among_exposed']}%)" if row["any_poison_retrieved"] else "0/0 (0.0%)",
                    f"{row['cosine_false_positives']}/{row['any_poison_retrieved']} ({m['cosine_fp_rate_among_exposed']}%)" if row["any_poison_retrieved"] else "0/0 (0.0%)",
                ]
            )
            + " |"
        )
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    summaries = [summarize(Path(p)) for p in args.files]
    if args.format == "json":
        print(json.dumps(summaries, indent=2))
    else:
        print(render_markdown(summaries))


if __name__ == "__main__":
    main()
