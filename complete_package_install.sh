#!/bin/bash
#SBATCH --job-name=pip-install
#SBATCH --account=uwb
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --output=logs/pip_install_%j.out
#SBATCH --error=logs/pip_install_%j.err

echo "Completing package installation for guardrag environment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo ""

# Activate environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate guardrag

echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo ""

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core requirements individually
echo "Installing Transformers..."
pip install transformers==4.36.0

echo "Installing accelerate..."
pip install accelerate==0.25.0

echo "Installing sentence-transformers..."
pip install sentence-transformers==2.3.0

echo "Installing other core packages..."
pip install rank-bm25==0.2.2
pip install datasets==2.16.0
pip install huggingface-hub==0.20.0

echo "Installing quantization support..."
pip install bitsandbytes==0.41.0

echo "Installing evaluation metrics..."
pip install rouge-score nltk sacrebleu pandas numpy scikit-learn

echo "Installing utilities..."
pip install tqdm jsonlines pyyaml matplotlib seaborn

echo ""
echo "Verification:"
python << 'PYEOF'
import sys
print(f"Python: {sys.version}")
print()

packages = [
    'torch', 'transformers', 'accelerate', 'sentence_transformers',
    'rank_bm25', 'datasets', 'huggingface_hub', 'bitsandbytes'
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'N/A')
        print(f"✅ {pkg}: {version}")
    except Exception as e:
        print(f"❌ {pkg}: {e}")

print()
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
PYEOF

echo ""
echo "✅ Installation complete!"
echo "Run: source ~/.bashrc && conda activate guardrag"
