#!/bin/bash
# Author: Gayatri Malladi
#
# Unified corpus-generation launcher for RIPE-II.
# Supports:
#   1. Canonical blackbox main corpora via materialize_* scripts
#   2. Graybox / whitebox corpora via the generic semantic generator
#
# Examples:
#   CORPORA=all ATTACKER_SETTING=blackbox sbatch run_corpus_generation.sh
#   CORPORA=fiqa,nq ATTACKER_SETTING=graybox NUM_ATTACKS=200 sbatch run_corpus_generation.sh
#   CORPORA=nfcorpus ATTACKER_SETTING=whitebox DOC_POISON_RATE=0.05 sbatch run_corpus_generation.sh
#
# Environment variables:
#   CORPORA           Comma-separated corpus list or "all"
#   ATTACKER_SETTING  blackbox | graybox | whitebox
#   MODE              realistic | hard | stress   (generic generator only)
#   NUM_ATTACKS       Exact attack count override  (generic generator only)
#   DOC_POISON_RATE   Poison rate override         (generic generator only)
#   SELECTION_MODE    random | curated             (generic generator only)
#   MAX_QUERIES       Passed to main blackbox materializers when supported
#   MIN_OVERLAP       Passed to NQ/FIQA/HotpotQA blackbox materializers
#   TOP_CANDIDATES    Passed to NQ/FIQA/HotpotQA blackbox materializers
#   OUTPUT_ROOT       Base directory for graybox/whitebox outputs

#SBATCH --job-name=corpus_gen
#SBATCH --output=/gscratch/uwb/gayat23/GuardRAG/logs/corpus_gen_%j.out
#SBATCH --error=/gscratch/uwb/gayat23/GuardRAG/logs/corpus_gen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=ckpt-gpu
#SBATCH --account=uwb-ckpt
#SBATCH --gres=gpu:1

set -euo pipefail

PROJ=/mmfs1/home/gayat23/projects/guardrag-thesis
PYTHON=/gscratch/uwb/gayat23/conda/envs/guardrag/bin/python
GSRATCH_IPI=/gscratch/uwb/gayat23/GuardRAG/IPI_generators

CORPORA=${CORPORA:-all}
ATTACKER_SETTING=${ATTACKER_SETTING:-blackbox}
MODE=${MODE:-realistic}
NUM_ATTACKS=${NUM_ATTACKS:-}
DOC_POISON_RATE=${DOC_POISON_RATE:-}
SELECTION_MODE=${SELECTION_MODE:-curated}
MAX_QUERIES=${MAX_QUERIES:-}
MIN_OVERLAP=${MIN_OVERLAP:-}
TOP_CANDIDATES=${TOP_CANDIDATES:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-$GSRATCH_IPI}

export HF_HOME=/gscratch/uwb/gayat23/hf_cache
export HUGGING_FACE_HUB_TOKEN=$(cat /gscratch/uwb/gayat23/hf_cache/token 2>/dev/null)
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

split_csv() {
    local raw="$1"
    raw="${raw// /}"
    local IFS=','
    read -r -a items <<< "$raw"
    printf '%s\n' "${items[@]}"
}

all_corpora() {
    printf '%s\n' nfcorpus scifact fiqa nq msmarco hotpotqa
}

domain_for() {
    case "$1" in
        nfcorpus|scifact) echo "biomedical" ;;
        fiqa) echo "financial" ;;
        msmarco) echo "web" ;;
        nq|hotpotqa) echo "general" ;;
        *) echo "general" ;;
    esac
}

qrels_for() {
    local corpus="$1"
    if [[ "$corpus" == "msmarco" ]]; then
        echo "$PROJ/data/corpus/beir/$corpus/qrels/dev.tsv"
    else
        echo "$PROJ/data/corpus/beir/$corpus/qrels/test.tsv"
    fi
}

run_blackbox_main() {
    local corpus="$1"
    echo "------------------------------------------------------------"
    echo "Building canonical blackbox main corpus: $corpus"
    echo "------------------------------------------------------------"
    case "$corpus" in
        nfcorpus)
            "$PYTHON" corpus_generation/materialize_nfcorpus_main.py
            ;;
        scifact)
            local args=()
            [[ -n "$MAX_QUERIES" ]] && args+=(--max-queries "$MAX_QUERIES")
            "$PYTHON" corpus_generation/materialize_scifact_main.py "${args[@]}"
            ;;
        fiqa)
            local args=()
            [[ -n "$MAX_QUERIES" ]] && args+=(--max-queries "$MAX_QUERIES")
            [[ -n "$MIN_OVERLAP" ]] && args+=(--min-overlap "$MIN_OVERLAP")
            [[ -n "$TOP_CANDIDATES" ]] && args+=(--top-candidates "$TOP_CANDIDATES")
            "$PYTHON" corpus_generation/materialize_fiqa_main_candidate.py "${args[@]}"
            ;;
        nq)
            local args=()
            [[ -n "$MAX_QUERIES" ]] && args+=(--max-queries "$MAX_QUERIES")
            [[ -n "$MIN_OVERLAP" ]] && args+=(--min-overlap "$MIN_OVERLAP")
            [[ -n "$TOP_CANDIDATES" ]] && args+=(--top-candidates "$TOP_CANDIDATES")
            "$PYTHON" corpus_generation/materialize_nq_main.py "${args[@]}"
            ;;
        msmarco)
            local args=()
            [[ -n "$MAX_QUERIES" ]] && args+=(--max-queries "$MAX_QUERIES")
            "$PYTHON" corpus_generation/materialize_msmarco_main.py "${args[@]}"
            ;;
        hotpotqa)
            local args=()
            [[ -n "$MAX_QUERIES" ]] && args+=(--max-queries "$MAX_QUERIES")
            [[ -n "$MIN_OVERLAP" ]] && args+=(--min-overlap "$MIN_OVERLAP")
            [[ -n "$TOP_CANDIDATES" ]] && args+=(--top-candidates "$TOP_CANDIDATES")
            "$PYTHON" corpus_generation/materialize_hotpotqa_main.py "${args[@]}"
            ;;
        *)
            echo "Unknown corpus for blackbox main build: $corpus" >&2
            exit 1
            ;;
    esac
}

run_generic_threat_model() {
    local corpus="$1"
    local domain
    local out_dir
    local qrels
    local args

    domain=$(domain_for "$corpus")
    qrels=$(qrels_for "$corpus")
    out_dir="$OUTPUT_ROOT/ipi_${corpus}_${ATTACKER_SETTING}_main"

    args=(
        --corpus "$PROJ/data/corpus/beir/$corpus/corpus.jsonl"
        --queries "$PROJ/data/corpus/beir/$corpus/queries.jsonl"
        --out "$out_dir"
        --dataset "$corpus"
        --mode "$MODE"
        --domain "$domain"
        --attacker-setting "$ATTACKER_SETTING"
        --selection-mode "$SELECTION_MODE"
        --qrels "$qrels"
        --skip-bad-jsonl-lines
    )

    [[ -n "$NUM_ATTACKS" ]] && args+=(--num-attacks "$NUM_ATTACKS")
    [[ -n "$DOC_POISON_RATE" ]] && args+=(--doc-poison-rate "$DOC_POISON_RATE")

    echo "------------------------------------------------------------"
    echo "Building $ATTACKER_SETTING corpus: $corpus"
    echo "Mode       : $MODE"
    echo "Domain     : $domain"
    echo "Output dir : $out_dir"
    echo "------------------------------------------------------------"

    "$PYTHON" corpus_generation/ipi_generator_v4_semantic_dense.py "${args[@]}"
}

echo "============================================================"
echo "RIPE-II Corpus Generation"
echo "============================================================"
echo "Corpora          : $CORPORA"
echo "Attacker setting : $ATTACKER_SETTING"
echo "Mode             : $MODE"
echo "Node             : $(hostname)"
echo "Start            : $(date)"
echo "============================================================"

cd "$PROJ"
mkdir -p /gscratch/uwb/gayat23/GuardRAG/logs

if [[ "$CORPORA" == "all" ]]; then
    mapfile -t CORPUS_LIST < <(all_corpora)
else
    mapfile -t CORPUS_LIST < <(split_csv "$CORPORA")
fi

for corpus in "${CORPUS_LIST[@]}"; do
    if [[ "$ATTACKER_SETTING" == "blackbox" ]]; then
        run_blackbox_main "$corpus"
    else
        run_generic_threat_model "$corpus"
    fi
done

echo
echo "Done: $(date)"

