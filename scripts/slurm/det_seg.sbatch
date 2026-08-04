#!/bin/bash
#SBATCH --job-name=phase7-det-seg
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=tupi
#SBATCH --signal=B:USR1@300

set -e

# Load environment (skip module load; venv has CUDA built-in)
cd $(git rev-parse --show-toplevel)
source .venv/bin/activate

# Set seed for reproducibility
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

# Training parameters (override as needed)
TASK=${1:-detection}
STAGE=${2:-fp32}
MODEL=${3:-alexnet_bottleneck}
EXPERIMENT=${4:-phase7_detection}
EXTRA_ARGS=${5:-}

echo "===================================================="
echo "Phase 7 ${TASK} training [${STAGE}]"
echo "Model: ${MODEL}"
echo "Experiment: ${EXPERIMENT}"
echo "===================================================="

# Run training. Slurm sends SIGUSR1 to this batch shell 300s before the wall-clock
# limit (see --signal above). The resume checkpoint is already up to date (written
# every epoch regardless), so just requeue and let slurmd kill the child — forwarding
# the signal to Python (no handler installed) only kills it early for nothing, and its
# own `wait` returning 138 would abort this shell under `set -e` before requeue ran.
python scripts/train_det_seg.py ${TASK} \
    --model ${MODEL} \
    --stage ${STAGE} \
    --experiment ${EXPERIMENT} \
    --runtime local \
    --save-dir outputs/detection_segmentation/phase7 \
    ${EXTRA_ARGS} &
child=$!
trap 'scontrol requeue "$SLURM_JOB_ID"; sleep 600' USR1
wait "$child"

echo "===================================================="
echo "Training complete. Results in outputs/detection_segmentation/phase7/"
echo "===================================================="
