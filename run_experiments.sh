#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

TIMEOUT=86400  # 24 h — notebooks do full training runs

for nb in \
    notebooks/baselines_qat.ipynb \
    notebooks/alexnet_qat.ipynb \
    notebooks/compensation_qat.ipynb
do
    echo "=== Running $nb ==="
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=$TIMEOUT \
        "$nb"
    echo "=== Done: $nb ==="
done
