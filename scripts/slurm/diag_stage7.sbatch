#!/bin/bash
#SBATCH --job-name=diag-stage7
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=tupi
#SBATCH --output=slurm-%j.out

set -e
cd $(git rev-parse --show-toplevel)
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/diag_stage7.py
