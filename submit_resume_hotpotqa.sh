#!/bin/bash
#SBATCH --job-name=resume-hotpotqa-stage3
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/resume_hotpotqa_stage3_%j.out
#SBATCH --error=logs/resume_hotpotqa_stage3_%j.err

set -euo pipefail

echo "=========================================="
echo "RESUMING HOTPOTQA STAGE 3 EVALUATION"
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
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import rank_bm25; print('rank-bm25 available')"
echo ""

echo "=========================================="
echo "Starting Resume from Query 7430"
echo "=========================================="

cd /mmfs1/home/gayat23/projects/guardrag-thesis

# Run the resume script and append to log
python -u evaluation/resume_hotpotqa_stage3.py 2>&1 | tee -a evaluation/batch_stages_1_5_all_queries.log

echo ""
echo "=========================================="
echo "Resume Job Complete"
echo "=========================================="
