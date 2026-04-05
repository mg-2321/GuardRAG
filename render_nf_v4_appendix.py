#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN = RESULTS / "nfcorpus_v4_camera_ready_metrics.md"
NOTES = RESULTS / "nfcorpus_v4_metric_notes.md"
TECH = RESULTS / "nfcorpus_v4_technique_breakdown.md"
PAIR = RESULTS / "nfcorpus_v4_pairwise_stats.md"
OUT = RESULTS / "nfcorpus_v4_appendix.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    sections = [
        "# NF-Corpus v4 Appendix",
        "## Main Metrics",
        read_text(MAIN),
        "## Metric Notes",
        read_text(NOTES),
        "## Per-Technique Breakdown",
        read_text(TECH),
        "## Pairwise Model/ Retriever Comparisons",
        read_text(PAIR),
    ]
    OUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
