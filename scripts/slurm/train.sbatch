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

# Slurm sends SIGUSR1 to this batch shell shortly before the wall-clock limit (see
# configs/slurm/tupi_4090.yaml's `signal: "B:USR1@300"`). Forward it to the Python
# child so Trainer's existing signal handler (scripts/train.py) can stop after the
# current epoch and write its resume checkpoint, then requeue the job so it picks
# back up via auto_resume_path on the next node.
python -m scripts.train "$@" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true' USR1
wait "$child"
