#!/bin/bash
# Run Stages 6-7 for all corpora sequentially (one at a time)

CORPORA=("scifact" "nfcorpus" "fiqa" "hotpotqa")

echo "=================================================================================="
echo "SEQUENTIAL STAGES 6-7 EVALUATION - ALL QUERIES"
echo "=================================================================================="
echo ""
echo "Order: smallest to largest"
echo "  1. scifact: 1,109 queries (~3 hours)"
echo "  2. nfcorpus: 3,237 queries (~9 hours)"
echo "  3. fiqa: 6,648 queries (~18.5 hours)"
echo "  4. hotpotqa: 97,852 queries (~11.3 days)"
echo ""
echo "Total estimated time: ~12.6 days"
echo ""

for corpus in "${CORPORA[@]}"; do
    echo "=================================================================================="
    echo "Starting: $corpus"
    echo "=================================================================================="
    echo ""
    
    log_file="evaluation/${corpus}_stages_6_7_all_queries.log"
    
    python3 evaluation/run_stages_6_7_with_generation.py --corpus "$corpus" 2>&1 | tee "$log_file"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ $corpus completed successfully!"
    else
        echo ""
        echo "❌ $corpus failed with exit code $exit_code"
        echo "Check log: $log_file"
        exit $exit_code
    fi
    
    echo ""
    echo "Waiting 10 seconds before starting next corpus..."
    sleep 10
    echo ""
done

echo "=================================================================================="
echo "ALL CORPORA COMPLETED!"
echo "=================================================================================="

