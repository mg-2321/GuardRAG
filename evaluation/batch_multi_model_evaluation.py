#!/usr/bin/env python3
"""
Batch Multi-Model Evaluation for All Corpora.
Runs comprehensive_multi_model_evaluation.py for each corpus in sequence.
Tests BM25, 6 Dense models, 1 SPLADE, and 5 Hybrid configurations.

Usage:
  # All corpora, results to gscratch
  python evaluation/batch_multi_model_evaluation.py \
      --output-base /gscratch/uwb/gayat23/guardrag_results \
      --log /gscratch/uwb/gayat23/guardrag_results/logs/eval.log

  # Single corpus quick test
  python evaluation/batch_multi_model_evaluation.py \
      --corpora nfcorpus --sample-size 100 \
      --output-base /gscratch/uwb/gayat23/guardrag_results \
      --log /gscratch/uwb/gayat23/guardrag_results/logs/nfcorpus_test.log
"""

import subprocess
import sys
import json
import logging
from pathlib import Path
import argparse

# Default output base — override with --output-base
DEFAULT_OUTPUT_BASE = '/gscratch/uwb/gayat23/guardrag_results'

# Merged corpus paths (clean + poisoned combined) — stored in gscratch
MERGED_CORPORA_DIR = '/gscratch/uwb/gayat23/GuardRAG/stage3_results/merged_corpora'

# Corpus configurations — use merged corpora (clean + IPI_ poisoned docs)
CORPUS_CONFIGS = {
    'nfcorpus': {
        'corpus':   f'{MERGED_CORPORA_DIR}/nfcorpus_merged.jsonl',
        'metadata': 'IPI_generators/ipi_nfcorpus_realistic_sbert_pubmedbert/ipi_nfcorpus_realistic_sbert_pubmedbert/nfcorpus_realistic_attack_metadata_v2.jsonl',
        'queries':  'data/corpus/beir/nfcorpus/queries.jsonl',
        'subdir':   'nfcorpus_multi_model',
    },
    'fiqa': {
        'corpus':   f'{MERGED_CORPORA_DIR}/fiqa_merged.jsonl',
        'metadata': 'IPI_generators/ipi_fiqa_realistic_e5-base-v2/fiqa_realistic_attack_metadata_v2.jsonl',
        'queries':  'data/corpus/beir/fiqa/queries.jsonl',
        'subdir':   'fiqa_multi_model',
    },
    'hotpotqa': {
        'corpus':   f'{MERGED_CORPORA_DIR}/hotpotqa_merged.jsonl',
        'metadata': 'IPI_generators/ipi_hotpotqa_realistic_sbert_bge-m3/hotpotqa_realistic_attack_metadata_v2.jsonl',
        'queries':  'data/corpus/beir/hotpotqa/queries.jsonl',
        'subdir':   'hotpotqa_multi_model',
    },
    'scifact': {
        'corpus':   f'{MERGED_CORPORA_DIR}/scifact_merged.jsonl',
        'metadata': 'IPI_generators/ipi_scifact_realistic_sbert_specter/scifact_realistic_attack_metadata_v2.jsonl',
        'queries':  'data/corpus/beir/scifact/queries.jsonl',
        'subdir':   'scifact_multi_model',
    },
    'nq': {
        'corpus':   f'{MERGED_CORPORA_DIR}/nq_merged.jsonl',
        'metadata': 'IPI_generators/ipi_nq_realistic_e5-base-v2/nq_realistic_attack_metadata_v2.jsonl',
        'queries':  'data/corpus/beir/nq/queries.jsonl',
        'subdir':   'nq_multi_model',
    },
}


def setup_logging(log_path: str) -> logging.Logger:
    """Set up logging to both terminal (INFO level) and a file (DEBUG level)."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('batch_eval')
    logger.setLevel(logging.DEBUG)

    # Terminal handler — only INFO and above (clean, no per-query noise)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    # File handler — everything including subprocess output
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def evaluate_corpus(
    corpus_name: str,
    config: dict,
    output_base: Path,
    sample_size: int = None,
    log_path: str = None,
) -> bool:
    """Run multi-model evaluation for a single corpus."""
    print(f"\n{'='*80}")
    print(f"MULTI-MODEL EVALUATION: {corpus_name.upper()}")
    print(f"{'='*80}")

    corpus_path   = Path(config['corpus'])
    metadata_path = Path(config['metadata'])
    queries_path  = Path(config['queries'])
    output_dir    = output_base / config['subdir']
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, label in [(corpus_path, 'corpus'), (queries_path, 'queries'), (metadata_path, 'metadata')]:
        if not path.exists():
            print(f"Skipping {corpus_name}: {label} not found: {path}")
            return False

    cmd = [
        sys.executable, '-u',  # -u = unbuffered stdout so logs appear in real time
        'evaluation/comprehensive_multi_model_evaluation.py',
        '--corpus',      str(corpus_path),
        '--queries',     str(queries_path),
        '--metadata',    str(metadata_path),
        '--output',      str(output_dir),
        '--corpus-name', corpus_name,
    ]
    if sample_size:
        cmd += ['--sample-size', str(sample_size)]

    print(f"Output dir:  {output_dir}")
    print(f"Retrievers:  BM25 | Dense (mpnet, e5, bge) | SPLADE | Hybrid (e5 α=0.3/0.5/0.8) = 8 total")
    print()

    try:
        if log_path:
            # Tee subprocess output: write to log AND pass through to terminal
            log_file = Path(log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as lf:
                lf.write(f"\n{'='*80}\n")
                lf.write(f"CORPUS: {corpus_name}  CMD: {' '.join(cmd)}\n")
                lf.write(f"{'='*80}\n")
                lf.flush()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                    lf.flush()
                proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        else:
            subprocess.run(cmd, check=True)

        print(f"\n  {corpus_name} complete")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  {corpus_name} FAILED: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n  Interrupted for {corpus_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch multi-model retriever evaluation')
    parser.add_argument('--corpora', nargs='+', default=None,
                        choices=list(CORPUS_CONFIGS.keys()),
                        help='Corpora to evaluate (default: all 5)')
    parser.add_argument('--output-base', default=DEFAULT_OUTPUT_BASE,
                        help=f'Base output directory for results and caches '
                             f'(default: {DEFAULT_OUTPUT_BASE})')
    parser.add_argument('--log', default=None,
                        help='Path to log file — subprocess output is written here '
                             'in addition to terminal (e.g. /gscratch/.../eval.log)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Queries per corpus for quick testing')
    parser.add_argument('--list', action='store_true',
                        help='List available corpora and exit')

    args = parser.parse_args()

    if args.list:
        print("Available corpora:")
        for name in CORPUS_CONFIGS:
            print(f"  - {name}")
        return

    output_base     = Path(args.output_base)
    corpora_to_eval = args.corpora or list(CORPUS_CONFIGS.keys())

    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"BATCH MULTI-MODEL EVALUATION: {len(corpora_to_eval)} CORPORA")
    print(f"{'='*80}")
    print(f"Corpora:     {', '.join(corpora_to_eval)}")
    print(f"Output base: {output_base}")
    print(f"Log file:    {args.log or '(terminal only)'}")
    print(f"Sample size: {args.sample_size or 'ALL'}")
    print()

    results = {}
    for corpus_name in corpora_to_eval:
        config  = CORPUS_CONFIGS[corpus_name]
        success = evaluate_corpus(
            corpus_name=corpus_name,
            config=config,
            output_base=output_base,
            sample_size=args.sample_size,
            log_path=args.log,
        )
        results[corpus_name] = success

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {status:<8} {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    else:
        print("\nAll corpora evaluated successfully.")


if __name__ == "__main__":
    main()
