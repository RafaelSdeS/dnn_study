#!/usr/bin/env bash
set -euo pipefail

# .venv (not conda) -- conda init lives in ~/.bashrc, which a non-interactive batch shell never
# sources, so `command -v conda` always misses here. Same failure ("python: command not found")
# and same fix as scripts/slurm/det_seg.sbatch/prune_channels.sbatch already needed (jobs
# 812276-812278, and 821240/821241 on 2026-09-13). git rev-parse, not $TRAIN_REPO_ROOT/$PWD, for
# the same reason: a bad TRAIN_REPO_ROOT doubles the repo path instead of failing loudly.
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Slurm sends SIGUSR1 to this batch shell shortly before the wall-clock limit (see
# configs/slurm/tupi_4090.yaml's `signal: "B:USR1@900"`). Forward it to the Python
# child so Trainer's existing signal handler (scripts/train.py) can stop after the
# current epoch and write its resume checkpoint, then requeue the job so it picks
# back up via auto_resume_path on the next node.
python -m scripts.train "$@" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true' USR1
wait "$child"
