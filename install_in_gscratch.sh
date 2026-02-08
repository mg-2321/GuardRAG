#!/bin/bash
#SBATCH --job-name=gscratch-install
#SBATCH --partition=compute
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/gscratch_install_%j.out
#SBATCH --error=logs/gscratch_install_%j.err

echo "Installing conda environment in gscratch (33TB storage)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Set environment location in gscratch (NOT home!)
export CONDA_ENVS_PATH="/gscratch/uwb/gayat23/conda/envs"
export CONDA_PKGS_DIRS="/gscratch/uwb/gayat23/conda/pkgs"
mkdir -p $CONDA_ENVS_PATH $CONDA_PKGS_DIRS

echo "Environment path: $CONDA_ENVS_PATH"
echo ""

# Use home miniconda but create env in gscratch
source $HOME/miniconda3/bin/activate

# Remove old environment if exists
conda env remove -n guardrag -y 2>/dev/null || echo "No existing env"

# Create new environment in gscratch
echo "Creating environment in gscratch..."
conda create --prefix /gscratch/uwb/gayat23/conda/envs/guardrag python=3.11 -y

# Activate the gscratch environment
conda activate /gscratch/uwb/gayat23/conda/envs/guardrag

echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""

# Install packages (now writing to gscratch, not home!)
echo "Installing PyTorch..."
pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing Transformers..."
pip install --no-cache-dir transformers==4.36.0

echo "Installing bitsandbytes + accelerate..."
pip install --no-cache-dir bitsandbytes==0.41.1 accelerate==0.25.0

echo "Installing huggingface-hub..."
pip install --no-cache-dir huggingface-hub

echo ""
echo "=== Verification ==="
python -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'✅ CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python -c "import bitsandbytes; print('✅ bitsandbytes OK')"
python -c "from huggingface_hub import snapshot_download; print('✅ HF Hub OK')"

echo ""
echo "✅ Installation complete in gscratch!"
echo "To activate: conda activate /gscratch/uwb/gayat23/conda/envs/guardrag"
