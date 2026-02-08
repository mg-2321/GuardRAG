#!/bin/bash

# Monitor HotpotQA and FIQA jobs
echo "=== Job Status ==="
squeue -u gayat23 | grep -E "32966191|32966192|resume|fiqa"

echo ""
echo "=== HotpotQA Progress (if running) ==="
if [ -f "evaluation/organized_results/logs/batch_stages_1_5_all_queries.log" ]; then
    echo "Last 5 lines of HotpotQA log:"
    tail -5 evaluation/organized_results/logs/batch_stages_1_5_all_queries.log
    echo ""
    echo "HotpotQA Query Count:"
    grep -c "Query:" evaluation/organized_results/logs/batch_stages_1_5_all_queries.log || echo "Not started yet"
else
    echo "Log file not found - job may not have started"
fi

echo ""
echo "=== Next Steps ==="
echo "Jobs are queued and waiting for maintenance window to end (Feb 10-11)"
echo "Once maintenance ends, jobs will execute automatically"
echo ""
echo "Monitor with: watch -n 30 'squeue -u gayat23'"
