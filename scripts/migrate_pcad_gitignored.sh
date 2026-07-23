#!/usr/bin/env bash
# Run this ON PCAD after `git pull`, from the repo root.
# Moves gitignored files (*.pth, *.log, wandb/) left behind in old
# outputs/pcad/ folder names into the new reorganized structure.
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

echo "Done. Any leftover empty old folders under $ROOT can be removed manually."
