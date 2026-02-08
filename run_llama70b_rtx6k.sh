#!/bin/bash
#SBATCH --job-name=llama70b-4bit
#SBATCH --account=uwb
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/llama70b_4bit_%j.out
#SBATCH --error=logs/llama70b_4bit_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gayat23@uw.edu

echo "=========================================="
echo "Llama 70B 4-bit Evaluation on RTX 6000"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate guardrag

# Set model cache to lab storage (NOT home directory!)
export HF_HOME="/gscratch/uwb/gayat23/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

# Verify environment
echo "Python: $(python --version)"
echo "Location: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Go to project directory
cd /mmfs1/home/gayat23/projects/guardrag-thesis || exit 1

echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="
echo "Model: Llama 3.1 70B (4-bit quantized)"
echo "GPUs: 3x RTX 6000 (~72GB total VRAM)"
echo "Sample: 100 queries per corpus"
echo ""

# Run evaluation with 4-bit model
python evaluation/batch_run_llama70b_all_corpora.py \
    --model llama-3.1-70b-4bit \
    --retriever bm25 \
    --sample-size 100 \
    2>&1 | tee logs/llama70b_4bit_${SLURM_JOB_ID}_detailed.log

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Job completed successfully at $(date)"
else
    echo ""
    echo "❌ Job failed with exit code $EXIT_CODE at $(date)"
fi

exit $EXIT_CODE
