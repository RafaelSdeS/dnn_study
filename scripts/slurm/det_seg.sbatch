#!/bin/bash
#SBATCH --job-name=phase7-det-seg
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=tupi

set -e

# Load environment
module load cuda/12.1
source ~/.bash_profile
cd $(git rev-parse --show-toplevel)
source .venv/bin/activate

# Set seed for reproducibility
export PYTHONHASHSEED=42

# Training parameters (override as needed)
TASK=${1:-detection}
STAGE=${2:-fp32}
MODEL=${3:-alexnet_bottleneck}
EXPERIMENT=${4:-phase7_detection}

echo "===================================================="
echo "Phase 7 ${TASK} training [${STAGE}]"
echo "Model: ${MODEL}"
echo "Experiment: ${EXPERIMENT}"
echo "===================================================="

# Run training
python scripts/train_det_seg.py ${TASK} \
    --model ${MODEL} \
    --stage ${STAGE} \
    --experiment ${EXPERIMENT} \
    --runtime local \
    --save-dir runs/phase7

echo "===================================================="
echo "Training complete. Results in runs/phase7/"
echo "===================================================="
