#!/usr/bin/env bash
set -euo pipefail

# .venv (not conda) -- conda init lives in ~/.bashrc, which a non-interactive batch shell never
# sources, so `command -v conda` always misses here. Same failure ("python: command not found")
# and same fix as scripts/slurm/train.sbatch/det_seg.sbatch already needed.
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Executes a notebook headlessly for the models scripts/train.py's generic loop
# can't drive (Phase 8's vit_tiny/deit_tiny -- DistillationTrainer, see
# notebooks/phase_8_efficient_vit/vit_qat_phase8.ipynb).
# --inplace only rewrites cell outputs, not source, so a requeued re-run still
# re-executes the same cells; each training cell's own auto_resume_path() check
# (same pattern as scripts/train.py) picks up mid-run instead of restarting.
#
# Same SIGUSR1 forwarding as train.sbatch (configs/slurm/tupi_4090.yaml's
# `signal: "B:USR1@300"`) -- nbconvert itself doesn't handle SIGUSR1 specially,
# so this just terminates the current cell early; Trainer's per-epoch
# _resume.pth checkpoint (ml/trainer.py) means at most one in-progress epoch is
# lost, not more.
NOTEBOOK="${NOTEBOOK_PATH:?NOTEBOOK_PATH env var must be set}"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    "$NOTEBOOK" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true' USR1
wait "$child"
