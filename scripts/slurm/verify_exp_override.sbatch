#!/bin/bash
#SBATCH --job-name=verify-override
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared
#SBATCH --output=slurm-%j.out

set -e
cd $(git rev-parse --show-toplevel)
source .venv/bin/activate

for m in alexnet_bottleneck alexnet_fire alexnet_tv; do
  echo "=== $m ==="
  python scripts/train_det_seg.py detection --model $m --stage fp32 --experiment phase7_detection --dry-run --save-dir /tmp/verify_override
  cat /tmp/verify_override/ssd_${m}_fp32_phase7_detection/config.yaml
done
