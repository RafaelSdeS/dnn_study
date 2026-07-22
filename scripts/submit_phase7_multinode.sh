#!/bin/bash
#
# Multi-node Phase 7 job submission for PCAD cluster.
# Submits detection training across 3 nodes (one backbone per node).
# Optionally chains QAT/INT8 runs after FP32 completes.
#
# Usage:
#   bash scripts/submit_phase7_multinode.sh                 # FP32 only
#   bash scripts/submit_phase7_multinode.sh qat             # FP32 + QAT
#   bash scripts/submit_phase7_multinode.sh qat int8        # FP32 + QAT + INT8
#   bash scripts/submit_phase7_multinode.sh --dry-run       # Show commands without submitting

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Configuration
MODELS=("alexnet_bottleneck" "alexnet_fire" "alexnet_tv")
EXPERIMENT="phase7_detection"
PARTITION="tupi"   # RTX 4090 nodes; alternatives: shared (K20m, weak), grace (L40s), beagle (GTX1080Ti)
TIME="12:00:00"
MEM="32G"
GPUS=1

# Options
DRY_RUN=false
RUN_QAT=false
RUN_INT8=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        qat)
            RUN_QAT=true
            ;;
        int8)
            RUN_INT8=true
            RUN_QAT=true  # INT8 requires QAT first
            ;;
    esac
done

echo "=========================================="
echo "PHASE 7 MULTI-NODE JOB SUBMISSION"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
echo "Models: ${MODELS[@]}"
echo "Partition: $PARTITION | Time: $TIME | Memory: $MEM"
echo "FP32: ✓ | QAT: $([ "$RUN_QAT" = true ] && echo '✓' || echo '✗') | INT8: $([ "$RUN_INT8" = true ] && echo '✓' || echo '✗')"
echo "Dry-run: $DRY_RUN"
echo "=========================================="
echo ""

# Create log directory
mkdir -p runs/phase7/logs

echo "Submitting FP32 detection training (3 nodes, parallel)..."
echo ""

# Submit all FP32 jobs in parallel
fp32_job_ids=()
for model in "${MODELS[@]}"; do
    job_name="p7_${model:0:8}_fp32"
    log_file="runs/phase7/logs/p7_${model}_fp32_%j.log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] sbatch --job-name=$job_name --time=$TIME --mem=$MEM --gpus=$GPUS --partition=$PARTITION --output=$log_file scripts/slurm/det_seg.sbatch detection fp32 $model $EXPERIMENT"
        fp32_job_ids+=("DRY_RUN_ID")
    else
        echo "Submitting: $job_name"
        output=$(sbatch --job-name="$job_name" --time="$TIME" --mem="$MEM" --gpus="$GPUS" --partition="$PARTITION" --output="$log_file" scripts/slurm/det_seg.sbatch detection fp32 "$model" "$EXPERIMENT" 2>&1)
        job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
        if [ -z "$job_id" ]; then
            echo "  ERROR submitting $model: $output"
            fp32_job_ids+=("")
        else
            echo "  Job ID: $job_id"
            fp32_job_ids+=("$job_id")
        fi
    fi
done

# Optional: QAT runs (after FP32 completes)
if [ "$RUN_QAT" = true ]; then
    echo ""
    echo "Chaining QAT training (depends on FP32 completion)..."
    echo ""

    qat_job_ids=()
    for i in "${!MODELS[@]}"; do
        model="${MODELS[$i]}"
        depend_on="${fp32_job_ids[$i]}"

        if [ -z "$depend_on" ]; then
            echo "Skipping QAT for $model (no FP32 job ID)"
            qat_job_ids+=("")
            continue
        fi

        job_name="p7_${model:0:8}_qat"
        log_file="runs/phase7/logs/p7_${model}_qat_%j.log"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY-RUN] sbatch --job-name=$job_name --time=$TIME --mem=$MEM --gpus=$GPUS --partition=$PARTITION --output=$log_file --dependency=afterok:$depend_on scripts/slurm/det_seg.sbatch detection qat $model $EXPERIMENT"
            qat_job_ids+=("DRY_RUN_ID")
        else
            echo "Submitting: $job_name (depends on FP32 job $depend_on)"
            output=$(sbatch --job-name="$job_name" --time="$TIME" --mem="$MEM" --gpus="$GPUS" --partition="$PARTITION" --output="$log_file" --dependency="afterok:$depend_on" scripts/slurm/det_seg.sbatch detection qat "$model" "$EXPERIMENT" 2>&1)
            job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
            if [ -z "$job_id" ]; then
                echo "  ERROR: $output"
                qat_job_ids+=("")
            else
                echo "  Job ID: $job_id"
                qat_job_ids+=("$job_id")
            fi
        fi
    done

    # Optional: INT8 runs (after QAT completes)
    if [ "$RUN_INT8" = true ]; then
        echo ""
        echo "Chaining INT8 conversion (depends on QAT completion)..."
        echo ""

        for i in "${!MODELS[@]}"; do
            model="${MODELS[$i]}"
            depend_on="${qat_job_ids[$i]}"

            if [ -z "$depend_on" ]; then
                echo "Skipping INT8 for $model (no QAT job ID)"
                continue
            fi

            job_name="p7_${model:0:8}_int8"
            log_file="runs/phase7/logs/p7_${model}_int8_%j.log"

            if [ "$DRY_RUN" = true ]; then
                echo "[DRY-RUN] sbatch --job-name=$job_name --time=$TIME --mem=$MEM --gpus=$GPUS --partition=$PARTITION --output=$log_file --dependency=afterok:$depend_on scripts/slurm/det_seg.sbatch detection int8 $model $EXPERIMENT"
            else
                echo "Submitting: $job_name (depends on QAT job $depend_on)"
                output=$(sbatch --job-name="$job_name" --time="$TIME" --mem="$MEM" --gpus="$GPUS" --partition="$PARTITION" --output="$log_file" --dependency="afterok:$depend_on" scripts/slurm/det_seg.sbatch detection int8 "$model" "$EXPERIMENT" 2>&1)
                job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
                if [ -z "$job_id" ]; then
                    echo "  ERROR: $output"
                else
                    echo "  Job ID: $job_id"
                fi
            fi
        done
    fi
fi

echo ""
echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY-RUN COMPLETE (no jobs submitted)"
else
    echo "SUBMISSION COMPLETE"
    echo "Monitor with: squeue -u $USER"
    echo "Check dependencies: squeue -u $USER --long"
    echo "Logs: runs/phase7/logs/"
fi
echo "=========================================="
