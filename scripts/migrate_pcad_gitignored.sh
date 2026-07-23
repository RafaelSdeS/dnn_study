#!/usr/bin/env bash
# Run this after `git pull`, from the repo root, on ANY machine that has
# gitignored artifacts (*.pth, *.log, wandb/) sitting in the old pre-reorg
# folder names — PCAD (outputs/pcad/) and/or wherever the Phase 1-4
# notebooks were run (top-level checkpoints/). Each move is a no-op if its
# source folder doesn't exist on this machine.
# Safe to re-run — skips anything already moved or missing.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="outputs/pcad"

move() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        echo "Merging $src -> $dst"
        mkdir -p "$dst"
        # -n: don't clobber files git already placed at the destination
        cp -rn "$src"/. "$dst"/ 2>/dev/null || true
        find "$src" -type f -exec rm -f {} \;
        find "$src" -type d -empty -delete
    fi
}

move "$ROOT/alexnet_3x3_fc"                    "$ROOT/archive_legacy_phases/phase_2_kernel_restriction/alexnet_3x3_fc"
move "$ROOT/alexnet_3x3_gap"                   "$ROOT/archive_legacy_phases/phase_2_kernel_restriction/alexnet_3x3_gap"
move "$ROOT/large_scale"                       "$ROOT/archive_legacy_phases/phase_4_5_large_scale"
move "$ROOT/phase6"                            "$ROOT/phase_6_hardware_profiling/runs"
move "$ROOT/phase6_backfill"                   "$ROOT/phase_6_hardware_profiling/backfill"
move "$ROOT/phase9_fire_bypass"                "$ROOT/phase_9_bypass_ablation/fire_bypass"
move "$ROOT/phase9_fire_bypass_large_scale"    "$ROOT/phase_9_bypass_ablation/fire_bypass_large_scale"
move "$ROOT/figures"                           "$ROOT/figures_generated"
move "$ROOT/logs"                              "$ROOT/logs_by_phase/all_slurm_jobs"

# Top-level checkpoints/ (Phase 1-4 notebook runs — *.pth is gitignored)
move "checkpoints/baselines_qat_phase1"        "checkpoints/phase_1_baseline_training"
move "checkpoints/alexnet_qat_phase2"          "checkpoints/phase_2_kernel_restriction_training"
move "checkpoints/compensation_phase3"         "checkpoints/phase_3_compensation_and_hybrids_training"
move "checkpoints/compression_phase4_1"        "checkpoints/phase_4_compression_and_final_architecture_training"
move "checkpoints/final_architecture_phase4"   "checkpoints/phase_4_compression_and_final_architecture_training"

# Top-level runs/phase7/ (Phase 7 detection/segmentation — *.pth is gitignored)
move "runs/phase7"                             "outputs/detection_segmentation/phase7"
rmdir "runs" 2>/dev/null || true

echo "Done. Any leftover empty old folders can be removed manually."
