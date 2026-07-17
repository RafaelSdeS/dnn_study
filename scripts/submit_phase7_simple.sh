#!/bin/bash
#
# Simpler Phase 7 multi-node submission.
# Submits 3 jobs (one per backbone) to PCAD in parallel.
# Each runs on a separate node.
#
# Usage:
#   bash scripts/submit_phase7_simple.sh           # Submit all 3
#   bash scripts/submit_phase7_simple.sh bottleneck fire  # Submit specific models

MODELS=${@:-alexnet_bottleneck alexnet_fire alexnet_tv}

echo "Submitting Phase 7 FP32 training to PCAD..."
echo "Models: $MODELS"
echo ""

JOB_IDS=()
for model in $MODELS; do
    echo "Submitting: $model"

    job_id=$(sbatch \
        --job-name="p7_${model:0:8}" \
        --time=12:00:00 \
        --mem=32G \
        --gpus=1 \
        --partition=gpu \
        --output="runs/phase7/logs/p7_${model}_%j.log" \
        scripts/slurm/det_seg.sbatch detection "$model" phase7_detection \
        | awk '{print $NF}')

    echo "  Submitted: Job $job_id"
    JOB_IDS+=("$job_id")
done

echo ""
echo "=========================================="
echo "Jobs submitted:"
for i in "${!MODELS[@]}"; do
    echo "  ${MODELS[$i]}: ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor: squeue -u $USER"
echo "Logs: runs/phase7/logs/"
echo "=========================================="
