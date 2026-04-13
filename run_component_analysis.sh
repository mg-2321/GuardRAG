#!/bin/bash
# Author: Gayatri Malladi
#
# Unified component-analysis launcher for RIPE-II.
# Covers retrieval, reranking, optional generation, and chunking sweeps.
#
# Examples:
#   CORPORA=nq RETRIEVERS=bm25,dense_e5,hybrid,splade RERANKERS=none,cross_encoder sbatch run_component_analysis.sh
#   CORPORA=msmarco CHUNK_SIZES=0,512,1024 RETRIEVERS=bm25 MODEL=llama-3.1-8b sbatch run_component_analysis.sh

#SBATCH --job-name=component_analysis
#SBATCH --output=/gscratch/uwb/gayat23/GuardRAG/logs/component_analysis_%j.out
#SBATCH --error=/gscratch/uwb/gayat23/GuardRAG/logs/component_analysis_%j.err
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
OUTPUT_DIR=/gscratch/uwb/gayat23/GuardRAG/results/component_study

CORPORA=${CORPORA:-nfcorpus}
TIER=${TIER:-main_candidate}
RETRIEVERS=${RETRIEVERS:-bm25,dense_e5,hybrid,splade}
RERANKERS=${RERANKERS:-none,cross_encoder}
CHUNK_SIZES=${CHUNK_SIZES:-0}
CHUNK_OVERLAP=${CHUNK_OVERLAP:-200}
TOP_K=${TOP_K:-10}
CAND_POOL=${CAND_POOL:-20}
LIMIT=${LIMIT:-0}
MODEL=${MODEL:-}
QUERY_PROCESSORS=${QUERY_PROCESSORS:-}
RERANKER_MODEL=${RERANKER_MODEL:-cross-encoder/ms-marco-MiniLM-L-12-v2}
RERANKER_TOP_N=${RERANKER_TOP_N:-10}

export HF_HOME=/gscratch/uwb/gayat23/hf_cache
export HUGGING_FACE_HUB_TOKEN=$(cat /gscratch/uwb/gayat23/hf_cache/token 2>/dev/null)
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=$(cat /gscratch/uwb/gayat23/.openai_api_key 2>/dev/null)
export ANTHROPIC_API_KEY=$(cat /gscratch/uwb/gayat23/.anthropic_api_key 2>/dev/null)
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export GUARDRAG_VECTOR_BACKEND=${GUARDRAG_VECTOR_BACKEND:-faiss_hnsw}

split_csv() {
    local raw="$1"
    raw="${raw// /}"
    local IFS=','
    read -r -a items <<< "$raw"
    printf '%s\n' "${items[@]}"
}

mkdir -p "$OUTPUT_DIR" /gscratch/uwb/gayat23/GuardRAG/logs
cd "$PROJ"

echo "============================================================"
echo "RIPE-II Component Analysis"
echo "============================================================"
echo "Corpora      : $CORPORA"
echo "Tier         : $TIER"
echo "Retrievers   : $RETRIEVERS"
echo "Rerankers    : $RERANKERS"
echo "Chunk sizes  : $CHUNK_SIZES"
echo "Model        : ${MODEL:-retrieval-only}"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "============================================================"

mapfile -t CORPUS_LIST < <(split_csv "$CORPORA")
mapfile -t RETRIEVER_LIST < <(split_csv "$RETRIEVERS")
mapfile -t RERANKER_LIST < <(split_csv "$RERANKERS")
mapfile -t CHUNK_LIST < <(split_csv "$CHUNK_SIZES")

for corpus in "${CORPUS_LIST[@]}"; do
    for retriever in "${RETRIEVER_LIST[@]}"; do
        for reranker in "${RERANKER_LIST[@]}"; do
            for chunk_size in "${CHUNK_LIST[@]}"; do
                ce_suffix=""
                chunk_suffix=""
                args=(
                    --corpus "$corpus"
                    --tier "$TIER"
                    --retriever "$retriever"
                    --top-k "$TOP_K"
                    --candidate-pool "$CAND_POOL"
                    --chunk-size "$chunk_size"
                    --chunk-overlap "$CHUNK_OVERLAP"
                )

                if [[ "$LIMIT" != "0" ]]; then
                    args+=(--limit "$LIMIT")
                fi
                if [[ -n "$QUERY_PROCESSORS" ]]; then
                    args+=(--query-processors "$QUERY_PROCESSORS")
                fi
                if [[ -n "$MODEL" ]]; then
                    args+=(--model "$MODEL")
                fi
                if [[ "$reranker" != "none" && -n "$reranker" ]]; then
                    ce_suffix="_ce"
                    args+=(--reranker "$reranker" --reranker-model "$RERANKER_MODEL" --reranker-top-n "$RERANKER_TOP_N")
                fi
                if [[ "$chunk_size" != "0" ]]; then
                    chunk_suffix="_chunk${chunk_size}"
                fi

                output="$OUTPUT_DIR/${corpus}_${TIER}_${retriever}${ce_suffix}${chunk_suffix}.jsonl"
                args+=(--output "$output")

                echo "------------------------------------------------------------"
                echo "Corpus     : $corpus"
                echo "Retriever  : $retriever"
                echo "Reranker   : $reranker"
                echo "Chunk size : $chunk_size"
                echo "Output     : $output"
                echo "------------------------------------------------------------"

                "$PYTHON" -u evaluation/run_component_study.py "${args[@]}"
            done
        done
    done
done

echo
echo "Done: $(date)"

