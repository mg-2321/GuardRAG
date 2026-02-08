#!/bin/bash
#SBATCH --job-name=stages-2-5-seq
#SBATCH --partition=compute
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stages_2_5_sequential_%j.out
#SBATCH --error=logs/stages_2_5_sequential_%j.err

set -euo pipefail

echo "=========================================="
echo "RUNNING STAGES 2-5 FOR ALL CORPORA (SEQUENTIAL)"
echo "Using existing working evaluation script"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="
echo ""

# Activate conda environment from gscratch
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /gscratch/uwb/gayat23/conda/envs/guardrag

# Use gscratch caches (avoid home quota + speed up repeated loads)
export HF_HOME="/gscratch/uwb/gayat23/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="/gscratch/uwb/gayat23/datasets"
export PYTHONUNBUFFERED=1

# Verify environment
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print((torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))"
echo ""

# Create output directory
mkdir -p evaluation/comparative_results

echo "=========================================="
echo "Processing Corpus 1/4: NFCORPUS"
echo "=========================================="
python -u evaluation/run_comparative_retriever_evaluation.py \
  --corpus IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl \
  --metadata IPI_generators/ipi_nfcorpus/nfcorpus_ipi_metadata_v2.jsonl \
  --queries data/corpus/beir/nfcorpus/queries.jsonl \
  --corpus-name nfcorpus \
  --output-dir evaluation/comparative_results

echo ""
echo "=========================================="
echo "Processing Corpus 2/4: FIQA"
echo "=========================================="
python -u evaluation/run_comparative_retriever_evaluation.py \
  --corpus IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl \
  --metadata IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl \
  --queries data/corpus/beir/fiqa/queries.jsonl \
  --corpus-name fiqa \
  --output-dir evaluation/comparative_results

echo ""
echo "=========================================="
echo "Processing Corpus 3/4: SCIFACT"
echo "=========================================="
python -u evaluation/run_comparative_retriever_evaluation.py \
  --corpus IPI_generators/ipi_scifact_aligned/scifact_ipi_query_aligned.jsonl \
  --metadata IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl \
  --queries data/corpus/beir/scifact/queries.jsonl \
  --corpus-name scifact \
  --output-dir evaluation/comparative_results

echo ""
echo "=========================================="
echo "Processing Corpus 4/4: HOTPOTQA"
echo "=========================================="
python -u evaluation/run_comparative_retriever_evaluation.py \
  --corpus IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl \
  --metadata IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl \
  --queries data/corpus/beir/hotpotqa/queries.jsonl \
  --corpus-name hotpotqa \
  --output-dir evaluation/comparative_results

echo ""
echo "=========================================="
echo "ALL CORPORA COMPLETE!"
echo "=========================================="
echo "Results:"
ls -lh evaluation/comparative_results/*.json
