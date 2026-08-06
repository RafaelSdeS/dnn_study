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

# Only matters for --finetune-epochs runs (long enough to hit a wall-clock limit); a plain
# --dry-run/--evaluate call just exits before the signal ever fires. Same requeue pattern as
# train.sbatch: forward Slurm's pre-timeout SIGUSR1 to the child so scripts.prune_channels'
# own handler can request_stop() and checkpoint, then requeue onto the next node.
python -m scripts.prune_channels "$@" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true' USR1
wait "$child"
