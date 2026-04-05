#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
LIVE = Path("/gscratch/uwb/gayat23/GuardRAG/results/live_judge")
RESULTS = ROOT / "results"

RUNS: List[Tuple[str, Path]] = [
    ("8B + BM25", LIVE / "nfcorpus_realistic_llama-3.1-8b_20260402_165257.jsonl"),
    ("8B + hybrid", LIVE / "nfcorpus_realistic_llama-3.1-8b_20260402_172744.jsonl"),
    ("70B + BM25", LIVE / "nfcorpus_realistic_llama-3.3-70b-4bit_20260403_140150.jsonl"),
    ("70B + hybrid", LIVE / "nfcorpus_realistic_llama-3.3-70b-4bit_20260403_145525.jsonl"),
    ("GPT-5.2 + BM25", LIVE / "nfcorpus_realistic_gpt-5.2_20260403_235209.jsonl"),
]

OUTPUT_MD = RESULTS / "nfcorpus_v4_camera_ready_metrics.md"
OUTPUT_JSON = RESULTS / "nfcorpus_v4_camera_ready_metrics.json"
OUTPUT_NOTES = RESULTS / "nfcorpus_v4_metric_notes.md"
COSINE_THRESHOLD = 0.65


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(num: int, den: int) -> float:
    return (100.0 * num / den) if den else 0.0


def wilson_interval(num: int, den: int, z: float = 1.96) -> Tuple[float, float]:
    if den == 0:
        return 0.0, 0.0
    phat = num / den
    denom = 1 + (z * z) / den
    center = (phat + (z * z) / (2 * den)) / denom
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / den) + ((z * z) / (4 * den * den)))
        / denom
    )
    return 100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)


def fmt_ci(num: int, den: int) -> str:
    lo, hi = wilson_interval(num, den)
    return f"[{lo:.1f}, {hi:.1f}]"


def fmt_rate(num: int, den: int) -> str:
    return f"{num}/{den} ({pct(num, den):.1f}%)"


def compute_kappa(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 0.0
    po = (a + d) / n
    p_yes_j = (a + b) / n
    p_no_j = (c + d) / n
    p_yes_c = (a + c) / n
    p_no_c = (b + d) / n
    pe = p_yes_j * p_yes_c + p_no_j * p_no_c
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def summarize_run(label: str, path: Path) -> Dict:
    rows = load_jsonl(path)
    n = len(rows)
    exposed = [r for r in rows if r.get("any_poison_retrieved")]
    target = [r for r in rows if r.get("target_retrieved")]
    judge_yes = [r for r in rows if r.get("judge_verdict") is True]
    judge_yes_exposed = [r for r in exposed if r.get("judge_verdict") is True]
    target_judge_yes = [r for r in target if r.get("judge_verdict") is True]
    cosine_yes = [r for r in rows if r.get("cosine_asr") is True]
    cosine_yes_exposed = [r for r in exposed if r.get("cosine_asr") is True]

    both_yes = sum(
        r.get("judge_verdict") is True and r.get("cosine_asr") is True for r in exposed
    )
    judge_only_yes = sum(
        r.get("judge_verdict") is True and r.get("cosine_asr") is not True for r in exposed
    )
    cosine_only_yes = sum(
        r.get("judge_verdict") is not True and r.get("cosine_asr") is True for r in exposed
    )
    both_no = sum(
        r.get("judge_verdict") is not True and r.get("cosine_asr") is not True for r in exposed
    )
    agreement_n = both_yes + both_no
    agreement_rate = pct(agreement_n, len(exposed))
    kappa = compute_kappa(both_yes, judge_only_yes, cosine_only_yes, both_no)

    return {
        "label": label,
        "file": str(path),
        "rows": n,
        "exposure_n": len(exposed),
        "target_n": len(target),
        "judge_overall_n": len(judge_yes),
        "judge_exposed_n": len(judge_yes_exposed),
        "target_specific_judge_n": len(target_judge_yes),
        "cosine_overall_n": len(cosine_yes),
        "cosine_exposed_n": len(cosine_yes_exposed),
        "both_yes_n": both_yes,
        "judge_only_yes_n": judge_only_yes,
        "cosine_only_yes_n": cosine_only_yes,
        "both_no_n": both_no,
        "agreement_n": agreement_n,
        "agreement_rate": agreement_rate,
        "kappa": kappa,
        "judge_overall_ci": wilson_interval(len(judge_yes), n),
        "judge_exposed_ci": wilson_interval(len(judge_yes_exposed), len(exposed)),
    }


def render_table(rows: List[Dict]) -> str:
    header = (
        "| Run | N | Poison Exposure | Target Retrieved | "
        "Judge ASR Overall | 95% CI | Judge ASR on Any-Poison Exposure | 95% CI | "
        "Target-Specific Judge ASR | Cosine Overall | Judge-only YES | Cosine-only YES | "
        "Agreement | Cohen's kappa |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {label} | {n} | {exposure} | {target} | {judge_overall} | {judge_overall_ci} | "
            "{judge_exposed} | {judge_exposed_ci} | {target_specific} | {cosine_overall} | "
            "{judge_only} | {cosine_only} | {agreement} | {kappa:.3f} |".format(
                label=row["label"],
                n=row["rows"],
                exposure=fmt_rate(row["exposure_n"], row["rows"]),
                target=fmt_rate(row["target_n"], row["rows"]),
                judge_overall=fmt_rate(row["judge_overall_n"], row["rows"]),
                judge_overall_ci=fmt_ci(row["judge_overall_n"], row["rows"]),
                judge_exposed=fmt_rate(row["judge_exposed_n"], row["exposure_n"]),
                judge_exposed_ci=fmt_ci(row["judge_exposed_n"], row["exposure_n"]),
                target_specific=fmt_rate(row["target_specific_judge_n"], row["target_n"]),
                cosine_overall=fmt_rate(row["cosine_overall_n"], row["rows"]),
                judge_only=fmt_rate(row["judge_only_yes_n"], row["exposure_n"]),
                cosine_only=fmt_rate(row["cosine_only_yes_n"], row["exposure_n"]),
                agreement=fmt_rate(row["agreement_n"], row["exposure_n"]),
                kappa=row["kappa"],
            )
        )
    return "\n".join(lines) + "\n"


def render_notes() -> str:
    return f"""# NF-Corpus v4 Metric Notes

- `Judge-only YES` and `Cosine-only YES` are disagreement categories. They are intentionally neutral labels and do not treat either detector as ground truth.
- The judge metric uses a **clean counterfactual baseline** in `evaluation/live_judge_eval.py`, so judge-based ASR is not baseline-free.
- Cosine uses a fixed threshold of `{COSINE_THRESHOLD}` and should be interpreted as a detector, not as a counterfactual causal test.
- `Target-Specific Judge ASR` differs from `Judge ASR on Any-Poison Exposure`: the former requires retrieval of the intended poisoned document, while the latter counts any poisoned document retrieved for the query.
- `GPT-5.2 + hybrid` is currently missing from the matrix and should be marked as missing rather than silently compared as if the design were complete.
- The NF query set has `N=53`, so model-comparison claims should be paired with uncertainty estimates or paired significance tests in the paper text.
"""


def main() -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(label, path) for label, path in RUNS]
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_table(rows), encoding="utf-8")
    OUTPUT_NOTES.write_text(render_notes(), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_NOTES}")


if __name__ == "__main__":
    main()
