#!/bin/bash
# Setup HuggingFace and verify everything is ready

echo "=========================================="
echo "HuggingFace Setup and Verification"
echo "=========================================="
echo ""

# Check if conda is available
if [ ! -d "$HOME/miniconda3" ]; then
    echo "❌ Miniconda not installed yet. Please wait for installation to complete."
    exit 1
fi

# Activate conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate guardrag 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ guardrag environment not ready yet. Installation still in progress."
    exit 1
fi

echo "✅ Conda environment activated"
python --version
echo ""

# Login to HuggingFace
echo "Setting up HuggingFace authentication..."
echo ""

# Use the provided token
export HF_TOKEN="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"

# Login using token
echo "$HF_TOKEN" | $HOME/miniconda3/envs/guardrag/bin/huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

if [ $? -eq 0 ]; then
    echo "✅ Successfully logged into HuggingFace"
else
    echo "⚠️  HuggingFace login may have issues, but continuing..."
fi

echo ""
echo "=========================================="
echo "Verifying Setup"
echo "=========================================="
echo ""

# Run verification
cd /mmfs1/home/gayat23/projects/guardrag-thesis
python verify_setup.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now:"
echo "1. Request GPU node: salloc -A uwb -p gpu-rtx6k --gpus=3 --mem=120G --time=4:00:00"
echo "2. Activate environment: conda activate guardrag"
echo "3. Run test: python evaluation/run_stages_6_7_llama70b.py --corpus scifact --model llama-3.3-70b-4bit --sample-size 5"
echo ""
