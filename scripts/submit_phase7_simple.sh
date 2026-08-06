#!/bin/bash
#
# Simpler Phase 7 multi-node submission.
# Submits 3 jobs (one per backbone) to PCAD in parallel.
# Each runs on a separate node.
#
# Usage:
#   bash scripts/submit_phase7_simple.sh           # Submit all 3
#   bash scripts/submit_phase7_simple.sh bottleneck fire  # Submit specific models

if [ "$#" -eq 0 ]; then
    MODELS=("alexnet_bottleneck" "alexnet_fire" "alexnet_tv")
else
    MODELS=("$@")
fi

PARTITION="tupi"   # RTX 4090 nodes; alternatives: shared (K20m, weak), grace (L40s), beagle (GTX1080Ti)

mkdir -p outputs/detection_segmentation/phase7/logs

echo "Submitting Phase 7 FP32 training to PCAD..."
echo "Models: ${MODELS[@]}"
echo "Partition: $PARTITION"
echo ""

JOB_IDS=()
for model in "${MODELS[@]}"; do
    echo "Submitting: $model"

    output=$(sbatch \
        --job-name="p7_${model:0:8}" \
        --time=12:00:00 \
        --mem=32G \
        --gpus=1 \
        --partition="$PARTITION" \
        --output="outputs/detection_segmentation/phase7/logs/p7_${model}_%j.log" \
        scripts/slurm/det_seg.sbatch detection "$model" phase7_detection 2>&1)

    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
    if [ -z "$job_id" ]; then
        echo "  ERROR: $output"
    else
        echo "  Submitted: Job $job_id"
    fi
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
echo "Logs: outputs/detection_segmentation/phase7/logs/"
echo "=========================================="
