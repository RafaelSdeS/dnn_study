#!/usr/bin/env bash
set -euo pipefail

# ponytail: cd via git rev-parse (not $TRAIN_REPO_ROOT/$PWD) + .venv (not conda) — matches
# scripts/slurm/det_seg.sbatch's already-proven pattern. The old TRAIN_REPO_ROOT+conda-activate
# combo caused every real PCAD attempt (jobs 812276-812278) to fail before Python ever ran: a
# doubled repo path from a bad TRAIN_REPO_ROOT, then "python: command not found" because conda
# never activates in a non-interactive batch shell (conda init lives in ~/.bashrc, which batch
# shells don't source).
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Only matters for --finetune-epochs runs (long enough to hit a wall-clock limit); a plain
# --dry-run/--evaluate call just exits before the signal ever fires. Same requeue pattern as
# train.sbatch: forward Slurm's pre-timeout SIGUSR1 to the child so scripts.phase9.prune_channels'
# own handler can request_stop() and checkpoint, then requeue onto the next node.
python -m scripts.phase9.prune_channels "$@" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true' USR1
wait "$child"
