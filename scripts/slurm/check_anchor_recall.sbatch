#!/bin/bash
#SBATCH --job-name=anchor-recall-check
#SBATCH --time=00:40:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=tupi

set -e
cd $(git rev-parse --show-toplevel)
source .venv/bin/activate

python scripts/check_anchor_recall.py --model alexnet_bottleneck alexnet_fire alexnet_tv --img-size 512
