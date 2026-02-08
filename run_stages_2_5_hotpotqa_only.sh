#!/bin/bash
#SBATCH --job-name=stages-2-5-hotpotqa
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/stages_2_5_hotpotqa_%j.out
#SBATCH --error=logs/stages_2_5_hotpotqa_%j.err

set -euo pipefail

echo "=========================================="
echo "STAGES 2-5: HotpotQA ONLY (CONTINUATION)"
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

# Verify environment
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print((torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))"
echo ""

# Create output directory
mkdir -p evaluation/comparative_results

echo "=========================================="
echo "Processing HotpotQA (97,852 queries)"
echo "=========================================="
python -u evaluation/run_comparative_retriever_evaluation.py \
  --corpus IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl \
  --metadata IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl \
  --queries data/corpus/beir/hotpotqa/queries.jsonl \
  --corpus-name hotpotqa \
  --output-dir evaluation/comparative_results

echo ""
echo "=========================================="
echo "HOTPOTQA COMPLETE!"
echo "=========================================="
echo "Results:"
ls -lh evaluation/comparative_results/hotpotqa_retriever_comparison.json
