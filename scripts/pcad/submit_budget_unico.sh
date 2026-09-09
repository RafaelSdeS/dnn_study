#!/bin/bash
#
# Fase 2 do plano de avaliacao Winograd (Winograd-FPGA repo,
# docs/plano_avaliacao_redes_winograd.md): submete UM job por modelo do
# experimento budget_unico (configs/experiments/budget_unico.yaml) -- todos do
# zero, mesmo lr/epochs/seed (uniform_hparams: true), stages fp32 + qat_wino.
#
# Usage:
#   bash scripts/pcad/submit_budget_unico.sh                  # todos os 9 modelos do yaml
#   bash scripts/pcad/submit_budget_unico.sh vgg_style alexnet_stacked   # so' estes
#
# Cada modelo e' seu proprio job (nao um array), como scripts/pcad/submit_phase7_simple.sh --
# GoogLeNet/ResNet-50 desta lista de 9 nao fazem parte ainda (Fase 1 do plano nao
# terminou de registra-los em ml/model_registrations.py), senao entrariam aqui tambem.
#
# Wall-time via configs/slurm/tupi_4090.yaml (RTX 4090, 24h): dimensionado pelo proprio
# yaml para "large-scale FP32(<=1000ep)+QAT(<=100ep)", que cobre com folga o budget de
# 100+15 epocas daqui. --signal/--requeue replicam esse yaml (train.sbatch depende do
# SIGUSR1 pra checkpointar antes do wall-clock e dar scontrol requeue).

set -euo pipefail

if [ "$#" -eq 0 ]; then
    MODELS=(
        vgg_style
        alexnet_smallkernel
        alexnet_small_kernel_with_bn
        alexnet_stacked
        alexnet_fire
        alexnet_fire_bypass
        alexnet_bottleneck
        alexnet_final_fire_residual
        alexnet_final_bottleneck_residual
    )
else
    MODELS=("$@")
fi

EXPERIMENT="budget_unico"
PARTITION="${PARTITION:-tupi}"   # tupi_4090.yaml; ver configs/slurm/ para beagle/single_gpu
LOG_DIR="outputs/pcad/${EXPERIMENT}/logs"
mkdir -p "$LOG_DIR"

echo "Submitting ${EXPERIMENT} to PCAD (partition: ${PARTITION})"
echo "Models: ${MODELS[@]}"
echo ""

JOB_IDS=()
for model in "${MODELS[@]}"; do
    echo "Submitting: $model"
    output=$(sbatch \
        --job-name="budu_${model:0:12}" \
        --time=24:00:00 \
        --mem=32G \
        --gpus=1 \
        --cpus-per-task=8 \
        --partition="$PARTITION" \
        --requeue \
        --signal="B:USR1@300" \
        --output="${LOG_DIR}/${model}_%j.log" \
        scripts/slurm/train.sbatch --experiment "$EXPERIMENT" --runtime pcad --model "$model" 2>&1)

    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
    if [ -z "$job_id" ]; then
        echo "  ERROR: $output"
    else
        echo "  Submitted: Job $job_id"
    fi
    JOB_IDS+=("$job_id")
done

echo ""
echo "=========================================="
echo "Jobs submitted:"
for i in "${!MODELS[@]}"; do
    echo "  ${MODELS[$i]}: ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: ${LOG_DIR}/"
echo "Sync wandb after: wandb sync outputs/pcad/${EXPERIMENT}/*/wandb/ (offline mode, see configs/runtime/pcad.yaml)"
echo "=========================================="
