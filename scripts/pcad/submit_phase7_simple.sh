#!/bin/bash
#
# Simpler Phase 7 multi-node submission.
# Submits 3 jobs (one per backbone) to PCAD in parallel, FP32 stage only.
# Each runs on a separate node.
#
# Usage:
#   bash scripts/pcad/submit_phase7_simple.sh                       # Submit all 3, detection
#   bash scripts/pcad/submit_phase7_simple.sh alexnet_bottleneck alexnet_fire  # Specific models
#   TASK=segmentation bash scripts/pcad/submit_phase7_simple.sh     # Segmentation instead

if [ "$#" -eq 0 ]; then
    MODELS=("alexnet_bottleneck" "alexnet_fire" "alexnet_tv")
else
    MODELS=("$@")
fi

TASK="${TASK:-detection}"   # or: TASK=segmentation
EXPERIMENT="phase7_${TASK}"
PARTITION="tupi"   # RTX 4090 nodes; alternatives: shared (K20m, weak), grace (L40s), beagle (GTX1080Ti)

mkdir -p outputs/detection_segmentation/phase7/logs

echo "Submitting Phase 7 FP32 ${TASK} training to PCAD..."
echo "Models: ${MODELS[@]}"
echo "Partition: $PARTITION"
echo ""

JOB_IDS=()
for model in "${MODELS[@]}"; do
    echo "Submitting: $model"

    # sbatch positional args are TASK STAGE MODEL EXPERIMENT (scripts/slurm/det_seg.sbatch) --
    # previously passed as (detection, $model, phase7_detection), which put the model name in
    # the STAGE slot and "phase7_detection" in the MODEL slot; argparse would have rejected
    # both immediately.
    output=$(sbatch \
        --job-name="p7_${model:0:8}" \
        --time=12:00:00 \
        --mem=32G \
        --gpus=1 \
        --partition="$PARTITION" \
        --output="outputs/detection_segmentation/phase7/logs/p7_${TASK}_${model}_%j.log" \
        scripts/slurm/det_seg.sbatch "$TASK" fp32 "$model" "$EXPERIMENT" 2>&1)

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
