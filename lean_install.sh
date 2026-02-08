#!/bin/bash
#SBATCH --job-name=lean-install
#SBATCH --partition=compute
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/lean_install_%j.out
#SBATCH --error=logs/lean_install_%j.err

echo "Lean installation for guardrag environment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo ""

# Activate environment
source $HOME/miniconda3/bin/activate guardrag

# Show current state
python --version
pip --version
echo ""

# Install ONLY essentials (no cache, download one at a time)
echo "Installing PyTorch (no cache)..."
pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing Transformers..."
pip install --no-cache-dir transformers==4.36.0

echo "Installing quantization..."
pip install --no-cache-dir bitsandbytes==0.41.1 accelerate==0.25.0

echo "Installing HuggingFace Hub..."
pip install --no-cache-dir huggingface-hub

echo "Installing sentence-transformers..."
pip install --no-cache-dir sentence-transformers==2.3.0

echo "Installing BM25..."
pip install --no-cache-dir rank-bm25

echo "Installing datasets..."
pip install --no-cache-dir datasets

echo ""
echo "=== Verification ==="
python -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'✅ CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python -c "import bitsandbytes; print(f'✅ bitsandbytes available')"
python -c "import accelerate; print(f'✅ Accelerate available')"
python -c "import sentence_transformers; print(f'✅ sentence-transformers available')"

echo ""
echo "✅ Lean installation complete!"
