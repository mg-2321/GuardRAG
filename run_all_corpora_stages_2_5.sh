#!/bin/bash
#SBATCH --job-name=stages-2-5
#SBATCH --partition=compute
#SBATCH --account=uwb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stages_2_5_all_corpora_%j.out
#SBATCH --error=logs/stages_2_5_all_corpora_%j.err

echo "=========================================="
echo "RUNNING STAGES 2-5 FOR ALL CORPORA"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="
echo ""

# Activate conda environment from gscratch
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /gscratch/uwb/gayat23/conda/envs/guardrag

# Verify environment
echo "Python: $(python --version)"
echo "Location: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch check failed"
echo ""

# Define corpora to process
declare -A CORPORA=(
    ["nfcorpus"]="IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl|IPI_generators/ipi_nfcorpus/nfcorpus_ipi_metadata_v2.jsonl|data/corpus/beir/nfcorpus/queries.jsonl"
    ["fiqa"]="IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl|IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl|data/corpus/beir/fiqa/queries.jsonl"
    ["scifact"]="IPI_generators/ipi_scifact/scifact_ipi_mixed_v2.jsonl|IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl|data/corpus/beir/scifact/queries.jsonl"
    ["hotpotqa"]="IPI_generators/ipi_hotpotqa/hotpotqa_ipi_mixed_v2.jsonl|IPI_generators/ipi_hotpotqa/hotpotqa_ipi_metadata_v2.jsonl|data/corpus/beir/hotpotqa/queries.jsonl"
)

# Process each corpus
for corpus_name in "${!CORPORA[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing: $corpus_name"
    echo "=========================================="
    
    # Split the paths
    IFS='|' read -r corpus_path metadata_path queries_path <<< "${CORPORA[$corpus_name]}"
    
    echo "Corpus: $corpus_path"
    echo "Metadata: $metadata_path"
    echo "Queries: $queries_path"
    echo ""
    
    # Run stages 2-5 for this corpus
    python evaluation/run_stages_2_5_single_corpus.py \
        --corpus "$corpus_path" \
        --metadata "$metadata_path" \
        --queries "$queries_path" \
        --corpus-name "$corpus_name" \
        --output-dir evaluation/stage_by_stage_results \
        --sample-size 100
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $corpus_name completed successfully"
    else
        echo "❌ $corpus_name failed with exit code $exit_code"
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "ALL CORPORA PROCESSING COMPLETE"
echo "=========================================="
echo "Results saved to: evaluation/stage_by_stage_results/"
echo ""

# Show summary
echo "Summary of results:"
ls -lh evaluation/stage_by_stage_results/*.json
