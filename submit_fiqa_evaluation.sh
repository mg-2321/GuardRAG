#!/bin/bash
#SBATCH --job-name=fiqa-retriever-eval
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/fiqa_retriever_eval_%j.out
#SBATCH --error=logs/fiqa_retriever_eval_%j.err

set -euo pipefail

echo "=========================================="
echo "FIQA RETRIEVER EVALUATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="
echo ""

# Activate conda environment from gscratch
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /gscratch/uwb/gayat23/conda/envs/guardrag

# Use gscratch caches
export HF_HOME="/gscratch/uwb/gayat23/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="/gscratch/uwb/gayat23/datasets"
export PYTHONUNBUFFERED=1

echo "Environment:"
python --version
echo ""

cd /mmfs1/home/gayat23/projects/guardrag-thesis

echo "=========================================="
echo "Running FIQA Comparative Retriever Evaluation"
echo "=========================================="

python -u evaluation/run_comparative_retriever_evaluation.py \
  --corpus IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl \
  --metadata IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl \
  --queries data/corpus/beir/fiqa/queries.jsonl \
  --corpus-name fiqa \
  --output-dir evaluation/comparative_results

echo ""
echo "=========================================="
echo "FIQA Evaluation Complete"
echo "=========================================="
