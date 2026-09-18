#!/usr/bin/env bash
set -euo pipefail

# .venv (not conda) -- conda init lives in ~/.bashrc, which a non-interactive batch shell never
# sources, so `command -v conda` always misses here. Same failure ("python: command not found")
# and same fix as scripts/slurm/train.sbatch/det_seg.sbatch already needed.
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m scripts.profile_hardware "$@"
