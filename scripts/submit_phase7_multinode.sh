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

set -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Configuration
MODELS=("alexnet_bottleneck" "alexnet_fire" "alexnet_tv")
EXPERIMENT="phase7_detection"
PARTITION="gpu"
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

# Function to submit a single job
submit_job() {
    local model=$1
    local stage=$2
    local depend_on=$3

    local job_name="p7_${model:0:8}_${stage}"

    # Build sbatch command
    local sbatch_cmd="sbatch"
    sbatch_cmd="$sbatch_cmd --job-name=$job_name"
    sbatch_cmd="$sbatch_cmd --time=$TIME"
    sbatch_cmd="$sbatch_cmd --mem=$MEM"
    sbatch_cmd="$sbatch_cmd --gpus=$GPUS"
    sbatch_cmd="$sbatch_cmd --partition=$PARTITION"

    if [ -n "$depend_on" ]; then
        sbatch_cmd="$sbatch_cmd --dependency=afterok:$depend_on"
    fi

    sbatch_cmd="$sbatch_cmd --output=runs/phase7/logs/p7_${model}_${stage}_%j.log"

    # Inline script (simpler than .sbatch template)
    local script="
#!/bin/bash
set -e
module load cuda/12.1
source ~/.bash_profile
cd $PROJECT_ROOT
source .venv/bin/activate
export PYTHONHASHSEED=42

echo \"Starting ${stage} training for $model on \$(hostname)\"
python scripts/train_det_seg.py detection \\
    --model $model \\
    --experiment $EXPERIMENT \\
    --runtime local \\
    --save-dir runs/phase7

echo \"${stage} complete for $model\"
"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would submit: $sbatch_cmd"
        echo "  Script: $script"
        echo ""
    else
        # Write script to temp file and submit
        local script_file=$(mktemp)
        echo "$script" > "$script_file"
        chmod +x "$script_file"

        echo "Submitting: $job_name"
        local job_id=$(eval "$sbatch_cmd $script_file" | awk '{print $NF}')
        echo "  Job ID: $job_id"
        echo ""

        rm -f "$script_file"

        # Return job ID for dependency chaining
        echo "$job_id"
    fi
}

# Create log directory
mkdir -p runs/phase7/logs

echo "Submitting FP32 detection training (3 nodes, parallel)..."
echo ""

# Submit all FP32 jobs in parallel (no dependencies)
fp32_job_ids=()
for model in "${MODELS[@]}"; do
    job_id=$(submit_job "$model" "fp32")
    fp32_job_ids+=("$job_id")
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
        job_id=$(submit_job "$model" "qat" "$depend_on")
        qat_job_ids+=("$job_id")
    done

    # Optional: INT8 runs (after QAT completes)
    if [ "$RUN_INT8" = true ]; then
        echo ""
        echo "Chaining INT8 conversion (depends on QAT completion)..."
        echo ""

        for i in "${!MODELS[@]}"; do
            model="${MODELS[$i]}"
            depend_on="${qat_job_ids[$i]}"
            submit_job "$model" "int8" "$depend_on"
        done
    fi
fi

echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY-RUN COMPLETE (no jobs submitted)"
else
    echo "SUBMISSION COMPLETE"
    echo "Monitor with: squeue -u $USER"
    echo "Logs: runs/phase7/logs/"
fi
echo "=========================================="
