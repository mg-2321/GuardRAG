#!/bin/bash
#SBATCH --job-name=stages-6-7-llama70b
#SBATCH --account=uwb
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/stages_6_7_nfcorpus_scifact_%j.out
#SBATCH --error=logs/stages_6_7_nfcorpus_scifact_%j.err

set -euo pipefail

echo "=========================================="
echo "STAGES 6-7: Generation with Llama 3.3 70B (4-bit)"
echo "Corpora: NFCorpus, SciFact"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="
echo ""

# Activate conda environment from gscratch
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /gscratch/uwb/gayat23/conda/envs/guardrag

# Set model cache to gscratch
export HF_HOME="/gscratch/uwb/gayat23/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="/gscratch/uwb/gayat23/datasets"
export PYTHONUNBUFFERED=1

# Verify environment
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
echo ""

mkdir -p evaluation/stage_by_stage_results

echo "=========================================="
echo "Processing NFCorpus (3,237 queries)"
echo "=========================================="
python -u evaluation/run_stages_6_7_llama70b.py \
  --corpus nfcorpus \
  --model llama-3.3-70b-4bit \
  --retriever dense

echo ""
echo "=========================================="
echo "Processing SciFact (1,109 queries)"
echo "=========================================="
python -u evaluation/run_stages_6_7_llama70b.py \
  --corpus scifact \
  --model llama-3.3-70b-4bit \
  --retriever dense

echo ""
echo "=========================================="
echo "STAGES 6-7 COMPLETE!"
echo "=========================================="
ls -lh evaluation/stage_by_stage_results/*nfcorpus* evaluation/stage_by_stage_results/*scifact*
