#!/bin/bash
#SBATCH --job-name=dl-llama70b
#SBATCH --partition=compute
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/download_llama_%j.out
#SBATCH --error=logs/download_llama_%j.err

echo "Downloading Llama 3.3 70B Instruct model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo ""

# Set HuggingFace token and cache location (use lab storage, NOT home!)
export HF_TOKEN="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"
export HF_HOME="/gscratch/uwb/gayat23/huggingface"

# Activate environment (in gscratch)
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /gscratch/uwb/gayat23/conda/envs/guardrag

echo "Logging into HuggingFace..."
echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo ""
echo "Downloading Llama 3.3 70B Instruct..."
echo "Model: meta-llama/Llama-3.3-70B-Instruct"
echo "Size: ~140GB (will take 30-60 minutes depending on network)"
echo ""

# Download model (this will cache it)
python << 'EOF'
import os
from huggingface_hub import snapshot_download

model_id = "meta-llama/Llama-3.3-70B-Instruct"
print(f"Starting download of {model_id}...")

try:
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=os.environ.get("HF_HOME", f"{os.environ['HOME']}/.cache/huggingface"),
        resume_download=True,
        max_workers=4
    )
    print(f"\n✅ Model downloaded successfully!")
    print(f"Location: {local_dir}")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    exit(1)
EOF

echo ""
echo "=== Download complete! ==="
echo "Model is cached and ready to use."
echo ""

# Show cache size
du -sh $HF_HOME/hub/*Llama* 2>/dev/null || echo "Checking cache..."
