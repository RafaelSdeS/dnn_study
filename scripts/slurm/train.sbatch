#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${TRAIN_REPO_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-alexnet_rafael}"

cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
elif [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/bin/activate" ]]; then
    source "$CONDA_PREFIX/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m scripts.train "$@"
