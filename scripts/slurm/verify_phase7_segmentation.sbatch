#!/bin/bash
#SBATCH --job-name=verify-seg
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=tupi
#SBATCH --output=outputs/detection_segmentation/phase7/logs/verify_phase7_segmentation_%j.log

set -e
cd $(git rev-parse --show-toplevel)
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== ml.det_seg_trainer smoke tests (detection + segmentation, 10-image overfit) ==="
python -m ml.det_seg_trainer

echo ""
echo "=== pytest tests/ ==="
pytest tests/
