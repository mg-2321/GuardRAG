#!/bin/bash
#SBATCH --job-name=ultra-lean
#SBATCH --partition=compute
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/ultra_lean_%j.out
#SBATCH --error=logs/ultra_lean_%j.err

echo "Ultra-lean installation - essentials only"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Activate environment
source $HOME/miniconda3/bin/activate guardrag

# Install ONLY what we need (specific versions, minimal deps)
echo "Installing PyTorch..."
pip install --no-cache-dir --no-deps torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "Installing minimal deps..."
pip install --no-cache-dir filelock networkx sympy

echo "Installing transformers (minimal)..."
pip install --no-cache-dir transformers==4.36.0

echo "Installing bitsandbytes..."
pip install --no-cache-dir bitsandbytes==0.41.1

echo "Installing accelerate..."
pip install --no-cache-dir accelerate==0.25.0

echo "Installing huggingface-hub..."
pip install --no-cache-dir huggingface-hub

echo ""
echo "=== Testing installation ==="
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch failed"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')" || echo "❌ Transformers failed"
python -c "import bitsandbytes; print('✅ bitsandbytes OK')" || echo "❌ bitsandbytes failed"
python -c "from huggingface_hub import snapshot_download; print('✅ HF Hub OK')" || echo "❌ HF Hub failed"

echo ""
echo "✅ Ultra-lean installation complete!"
