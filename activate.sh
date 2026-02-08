#!/bin/bash
# Activate guardrag environment (located in gscratch for 33TB storage)

# Initialize conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Activate environment in gscratch (NOT home - avoid 10GB quota!)
conda activate /gscratch/uwb/gayat23/conda/envs/guardrag

# Set HuggingFace cache to lab storage (NOT home directory - saves 140GB!)
export HF_HOME="/gscratch/uwb/gayat23/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="/gscratch/uwb/gayat23/datasets"

echo "✅ guardrag environment activated (in gscratch)"
echo "📦 Model cache: $HF_HOME"
echo "Python: $(python --version)"

echo "✅ guardrag environment activated"
echo "Python: $(python --version)"
echo "Location: $(which python)"
