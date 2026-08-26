#!/bin/bash
#
# Multi-node Phase 7 job submission for PCAD cluster.
# Submits detection (or segmentation) training across 3 nodes (one backbone per node).
# Optionally chains QAT/INT8 runs after FP32 completes.
#
# Usage:
#   bash scripts/pcad/submit_phase7_multinode.sh                 # FP32 only, from-scratch backbones
#   bash scripts/pcad/submit_phase7_multinode.sh qat             # FP32 + QAT
#   bash scripts/pcad/submit_phase7_multinode.sh qat int8        # FP32 + QAT + INT8
#   bash scripts/pcad/submit_phase7_multinode.sh segmentation qat int8  # same, segmentation task
#   bash scripts/pcad/submit_phase7_multinode.sh pretrained      # same, but init backbones from
#                                                            # their Tiny-ImageNet classification
#                                                            # checkpoints instead of random init
#                                                            # (separate run_id tree, doesn't
#                                                            # collide with the from-scratch sweep)
#   bash scripts/pcad/submit_phase7_multinode.sh --dry-run       # Show commands without submitting

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Configuration
MODELS=("alexnet_bottleneck" "alexnet_fire" "alexnet_tv")
PARTITION="tupi"   # RTX 4090 nodes; alternatives: shared (K20m, weak), grace (L40s), beagle (GTX1080Ti)
TIME="24:00:00"
MEM="32G"
GPUS=1
# Phase 4/5 large-scale sweep's Tiny-ImageNet classification checkpoints — dir name
# matches ${model} exactly for all three arches, so no per-model special-casing needed.
CKPT_ROOT="outputs/pcad/archive_legacy_phases/phase_4_5_large_scale"

# Options
TASK="detection"   # or: segmentation
DRY_RUN=false
RUN_QAT=false
RUN_INT8=false
RUN_PRETRAINED=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        detection|segmentation)
            TASK="$arg"
            ;;
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
        pretrained)
            RUN_PRETRAINED=true
            ;;
    esac
done

EXPERIMENT="phase7_${TASK}"

# Suffix for job names/log files, distinguishing a pretrained-init sweep from the
# from-scratch one (mirrors train_det_seg.py's run_id "_pretrained" suffix).
name_suffix=""
[ "$RUN_PRETRAINED" = true ] && name_suffix="_pretrained"

echo "=========================================="
echo "PHASE 7 MULTI-NODE JOB SUBMISSION"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
echo "Models: ${MODELS[@]}"
echo "Partition: $PARTITION | Time: $TIME | Memory: $MEM"
echo "FP32: ✓ | QAT: $([ "$RUN_QAT" = true ] && echo '✓' || echo '✗') | INT8: $([ "$RUN_INT8" = true ] && echo '✓' || echo '✗')"
echo "Init: $([ "$RUN_PRETRAINED" = true ] && echo 'pretrained (Tiny-ImageNet checkpoints)' || echo 'from-scratch')"
echo "Dry-run: $DRY_RUN"
echo "=========================================="
echo ""

# Create log directory
mkdir -p outputs/detection_segmentation/phase7/logs

echo "Submitting FP32 ${TASK} training (3 nodes, parallel)..."
echo ""

# Submit all FP32 jobs in parallel
fp32_job_ids=()
for model in "${MODELS[@]}"; do
    job_name="p7_${model#alexnet_}_fp32${name_suffix}"
    log_file="outputs/detection_segmentation/phase7/logs/p7_${model}_fp32${name_suffix}_%j.log"
    extra_args=""
    [ "$TASK" = "detection" ] && [ "$model" = "alexnet_tv" ] && extra_args="--skip-anchor-check"
    [ "$RUN_PRETRAINED" = true ] && extra_args="$extra_args --pretrained-ckpt ${CKPT_ROOT}/${model}/checkpoints/${model}_best.pth"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] sbatch --job-name=$job_name --time=$TIME --mem=$MEM --gpus=$GPUS --partition=$PARTITION --output=$log_file scripts/slurm/det_seg.sbatch $TASK fp32 $model $EXPERIMENT $extra_args"
        fp32_job_ids+=("DRY_RUN_ID")
    else
        echo "Submitting: $job_name"
        output=$(sbatch --job-name="$job_name" --time="$TIME" --mem="$MEM" --gpus="$GPUS" --partition="$PARTITION" --output="$log_file" scripts/slurm/det_seg.sbatch $TASK fp32 "$model" "$EXPERIMENT" "$extra_args" 2>&1)
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

        job_name="p7_${model#alexnet_}_qat${name_suffix}"
        log_file="outputs/detection_segmentation/phase7/logs/p7_${model}_qat${name_suffix}_%j.log"
        # Presence (not content) of --pretrained-ckpt must match the FP32 stage's so
        # train_det_seg.py computes the same run_id suffix and finds the right
        # upstream checkpoint; the path itself is unused for qat/int8 backbone init.
        extra_args=""
        [ "$RUN_PRETRAINED" = true ] && extra_args="--pretrained-ckpt ${CKPT_ROOT}/${model}/checkpoints/${model}_best.pth"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY-RUN] sbatch --job-name=$job_name --time=$TIME --mem=$MEM --gpus=$GPUS --partition=$PARTITION --output=$log_file --dependency=afterok:$depend_on scripts/slurm/det_seg.sbatch $TASK qat $model $EXPERIMENT $extra_args"
            qat_job_ids+=("DRY_RUN_ID")
        else
            echo "Submitting: $job_name (depends on FP32 job $depend_on)"
            output=$(sbatch --job-name="$job_name" --time="$TIME" --mem="$MEM" --gpus="$GPUS" --partition="$PARTITION" --output="$log_file" --dependency="afterok:$depend_on" --kill-on-invalid-dep=yes scripts/slurm/det_seg.sbatch $TASK qat "$model" "$EXPERIMENT" "$extra_args" 2>&1)
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

            job_name="p7_${model#alexnet_}_int8${name_suffix}"
            log_file="outputs/detection_segmentation/phase7/logs/p7_${model}_int8${name_suffix}_%j.log"
            extra_args=""
            [ "$RUN_PRETRAINED" = true ] && extra_args="--pretrained-ckpt ${CKPT_ROOT}/${model}/checkpoints/${model}_best.pth"

            if [ "$DRY_RUN" = true ]; then
                echo "[DRY-RUN] sbatch --job-name=$job_name --time=$TIME --mem=$MEM --gpus=$GPUS --partition=$PARTITION --output=$log_file --dependency=afterok:$depend_on scripts/slurm/det_seg.sbatch $TASK int8 $model $EXPERIMENT $extra_args"
            else
                echo "Submitting: $job_name (depends on QAT job $depend_on)"
                output=$(sbatch --job-name="$job_name" --time="$TIME" --mem="$MEM" --gpus="$GPUS" --partition="$PARTITION" --output="$log_file" --dependency="afterok:$depend_on" --kill-on-invalid-dep=yes scripts/slurm/det_seg.sbatch $TASK int8 "$model" "$EXPERIMENT" "$extra_args" 2>&1)
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
    echo "Logs: outputs/detection_segmentation/phase7/logs/"
fi
echo "=========================================="
