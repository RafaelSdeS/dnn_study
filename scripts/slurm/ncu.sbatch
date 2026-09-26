#!/usr/bin/env bash
#SBATCH --job-name=ncu-phase6
#SBATCH --partition=tupi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/pcad/logs/phase_6_ncu/%x-%j.out
#SBATCH --error=outputs/pcad/logs/phase_6_ncu/%x-%j.err
set -euo pipefail

# Nsight Compute over scripts/phase6/ncu_targets.py, one .ncu-rep per target. PCAD has no ncu of
# its own: NCU defaults to a world-executable copy in another user's home (CUDA 11.8's, 2022.3,
# which predates the torch cu121 in .venv) -- override NCU=... if a newer one turns up. Counter
# access (ERR_NVGPUCTRPERM) is the other thing that can fail here; the log says which.
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

NCU="${NCU:-/home/users/fpetcheverria/cuda-11.8/nsight-compute-2022.3.0/ncu}"
OUT=outputs/pcad/phase_6_hardware_profiling/ncu
mkdir -p "$OUT"

nvidia-smi -L
"$NCU" --version

for t in conv2 conv3 conv5 winograd3; do
  "$NCU" --set full --profile-from-start off -f -o "$OUT/$t" python -m scripts.phase6.ncu_targets "$t"
  "$NCU" -i "$OUT/$t.ncu-rep" --page details > "$OUT/$t.details.txt"
done
