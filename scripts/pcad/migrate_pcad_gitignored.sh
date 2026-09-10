#!/usr/bin/env bash
# Run this after `git pull`, from the repo root, on ANY machine that has
# gitignored artifacts (*.pth, *.log, wandb/) sitting in old folder names —
# PCAD (outputs/pcad/), the local box (outputs/local/), and/or wherever the
# Phase 1-4/8 notebooks were run (top-level checkpoints/).
# Every move is a no-op if its source doesn't exist. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

move() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    [ "$src" = "$dst" ] && return 0
    mkdir -p "$dst"
    echo "Merging $src -> $dst"
    # -n: don't clobber files git already placed at the destination
    cp -rn "$src"/. "$dst"/ 2>/dev/null || true
    find "$src" -type f -exec rm -f {} \;
    find "$src" -depth -type d -empty -delete
}

# --- 2026-09 reorg: one canonical slug (phase_N_description) per phase ---
move "outputs/pcad/phase6"                              "outputs/pcad/phase_6_hardware_profiling/runs"
move "outputs/pcad/phase6_backfill"                     "outputs/pcad/phase_6_hardware_profiling/backfill"
move "outputs/pcad/phase8"                              "outputs/pcad/phase_8_efficient_vit"
move "outputs/pcad/phase_8_efficient_vit_hybrid_attention" "outputs/pcad/phase_8_efficient_vit"
move "outputs/pcad/results"                             "outputs/pcad/aggregates"
move "outputs/pcad/logs/phase6"                         "outputs/pcad/logs/phase_6_hardware_profiling"
move "outputs/pcad/logs/phase8"                         "outputs/pcad/logs/phase_8_efficient_vit"
move "outputs/pcad/logs/phase9_fire_bypass"             "outputs/pcad/logs/phase_9_bypass_ablation"
move "outputs/pcad/logs/phase9_fire_bypass_large_scale" "outputs/pcad/logs/phase_9_bypass_ablation_large_scale"
move "outputs/pcad/logs/phase9_pruning"                 "outputs/pcad/logs/phase_9_pruning"
move "outputs/local/phase6"                             "outputs/local/phase_6_hardware_profiling"
move "outputs/local/phase8_convstem"                    "outputs/local/phase_8_efficient_vit_convstem"
move "outputs/local/phase9_pruning"                     "outputs/local/phase_9_pruning"
move "outputs/local/phase9_compression"                 "outputs/local/phase_9_compression"
move "outputs/local/results"                            "outputs/local/aggregates"

# Detection/segmentation left the top level: it was always a PCAD run.
move "outputs/detection_segmentation/phase7"            "outputs/pcad/phase_7_detection_segmentation"
move "runs/phase7"                                      "outputs/pcad/phase_7_detection_segmentation"
rmdir "outputs/detection_segmentation" "runs" 2>/dev/null || true

# Phase 7 run dirs carry the experiment name (phase7_* -> phase_7_*), and so do
# the files inside them.
for d in outputs/pcad/phase_7_detection_segmentation/*_phase7_detection \
         outputs/pcad/phase_7_detection_segmentation/*_phase7_segmentation; do
    [ -d "$d" ] || continue
    new="${d/_phase7_detection/_phase_7_detection}"; new="${new/_phase7_segmentation/_phase_7_segmentation}"
    move "$d" "$new"
    for f in "$new"/*phase7_*; do
        [ -e "$f" ] || continue
        n="${f/phase7_detection/phase_7_detection}"; n="${n/phase7_segmentation/phase_7_segmentation}"
        mv -n "$f" "$n"
    done
done

# Notebook-era checkpoints stopped being a top-level dir; they are run output.
move "checkpoints/phase_1_baseline_training"                          "outputs/notebooks/phase_1_baseline"
move "checkpoints/phase_2_kernel_restriction_training"                "outputs/notebooks/phase_2_kernel_restriction"
move "checkpoints/phase_3_compensation_and_hybrids_training"          "outputs/notebooks/phase_3_compensation_and_hybrids"
move "checkpoints/phase_4_compression_and_final_architecture_training" "outputs/notebooks/phase_4_compression_and_final_architecture"
move "checkpoints/phase_8_efficient_vit_hybrid_attention_training"    "outputs/notebooks/phase_8_efficient_vit"

# --- pre-2026-09 names, kept so an old machine can catch up in one pass ---
move "outputs/pcad/alexnet_3x3_fc"   "outputs/pcad/archive_legacy_phases/phase_2_kernel_restriction/alexnet_3x3_fc"
move "outputs/pcad/alexnet_3x3_gap"  "outputs/pcad/archive_legacy_phases/phase_2_kernel_restriction/alexnet_3x3_gap"
move "outputs/pcad/large_scale"      "outputs/pcad/archive_legacy_phases/phase_4_5_large_scale"
move "outputs/pcad/phase9_fire_bypass"             "outputs/pcad/phase_9_bypass_ablation/fire_bypass"
move "outputs/pcad/phase9_fire_bypass_large_scale" "outputs/pcad/phase_9_bypass_ablation/fire_bypass_large_scale"
move "outputs/pcad/figures"          "results/figures_generated/phase_9_bypass_ablation"
move "outputs/pcad/figures_generated" "results/figures_generated/phase_9_bypass_ablation"
move "checkpoints/baselines_qat_phase1"      "outputs/notebooks/phase_1_baseline"
move "checkpoints/alexnet_qat_phase2"        "outputs/notebooks/phase_2_kernel_restriction"
move "checkpoints/compensation_phase3"       "outputs/notebooks/phase_3_compensation_and_hybrids"
move "checkpoints/compression_phase4_1"      "outputs/notebooks/phase_4_compression_and_final_architecture"
move "checkpoints/final_architecture_phase4" "outputs/notebooks/phase_4_compression_and_final_architecture"
rmdir "checkpoints" 2>/dev/null || true

# No move for outputs/pcad/logs itself: scripts/cluster.py already writes SLURM
# logs straight to outputs/pcad/logs/<experiment>/, which is the final location.

echo "Done. Any leftover empty old folders can be removed manually."
