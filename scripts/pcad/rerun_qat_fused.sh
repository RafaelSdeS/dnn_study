#!/usr/bin/env bash
# Wave 1 of the fused-QAT rerun (docs/logs/PHASE11_LOG.md, "QAT fusion bug"). Every run below trained QAT
# with no Conv-(BN-)ReLU fusion (ml/quantization.py:prepare_qat_model fused the caller's original, not the
# copy); its FP32 checkpoint is unaffected and is reused. Per <experiment>/<model>:
#   1. move the unfused QAT run's gitignored artifacts -- qat_<m>_best.pth/_resume.pth (whose presence makes
#      scripts/train.py skip QAT), the INT8 qat_<m>.pth, the QAT/INT8 logits and log -- into
#      outputs/pcad/archive_unfused_qat/<experiment>/<model>/, and copy the summary there;
#   2. resubmit the model: FP32 "resumes" at 500/500 (a no-op that reloads the best checkpoint, ml/trainer.py),
#      then QAT (100 ep) + INT8 run again from it, same protocol as every Phase 11 run.
# Tracked files (qat_<m>.pth.gz, *_meta.json, the summary) are overwritten in place; git keeps the old ones.
# Idempotent: an already-archived run is skipped. Run on PCAD from the repo root, after `git pull`, clean code.
#   scripts/pcad/rerun_qat_fused.sh            # all runs below
#   scripts/pcad/rerun_qat_fused.sh RUN...     # only these <experiment>/<model> (the wave-0 gate)
set -euo pipefail
ARCHIVE=outputs/pcad/archive_unfused_qat

# The default-init alexnet_tv_3x3 (factorial cell alexnet_fx_k3_s4_pk3n3_fc_d) lives in archive_old_init;
# give it a live run dir of its own experiment (configs/experiments/phase_11_reuse_old_init.yaml).
if [[ ! -d outputs/pcad/phase_11_reuse_old_init/alexnet_tv_3x3 ]]; then
  mkdir -p outputs/pcad/phase_11_reuse_old_init
  cp -a outputs/pcad/archive_old_init/phase_11_kernel_size_comparison/alexnet_tv_3x3 outputs/pcad/phase_11_reuse_old_init/
fi

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  mapfile -t RUNS <<'EOF'
phase_11_kernel_size_comparison/alexnet_tv_scratch
phase_11_kernel_size_comparison/alexnet_tv_3x3
phase_11_kernel_size_comparison/alexnet_tv_2x2
phase_11_kernel_size_comparison/vgg16
phase_11_kernel_size_comparison/vgg16_2x2
phase_11_mixed_kernel_comparison/alexnet_2x2_gap
phase_11_mixed_kernel_comparison/alexnet_3x3_gap
phase_11_mixed_kernel_comparison/alexnet_smallkernel
phase_11_mixed_kernel_comparison/alexnet_tv_mixed_alt
phase_11_mixed_kernel_comparison/alexnet_tv_mixed_early2
phase_11_mixed_kernel_comparison/alexnet_tv_mixed_early3
phase_11_head_bn_ablation/alexnet_mixed
phase_11_head_bn_ablation/alexnet_mixed_fc
phase_11_head_bn_ablation/alexnet_smallkernel_fc
phase_11_head_bn_ablation/alexnet_stacked
phase_11_head_bn_ablation/alexnet_stacked_gap
phase_11_head_bn_ablation/alexnet_stacked_gap_nobn
phase_11_head_bn_ablation/alexnet_tv_mixed_alt_gap
phase_11_head_bn_ablation/alexnet_tv_mixed_early2_gap
phase_11_head_bn_ablation/alexnet_tv_mixed_early3_gap
phase_11_geometry_controls/alexnet_3x3_fc
phase_11_geometry_controls/alexnet_adapted_2x2_fc
phase_11_geometry_controls/alexnet_adapted_2x2_gap
phase_11_geometry_controls/alexnet_adapted_orig_fc
phase_11_geometry_controls/alexnet_adapted_orig_gap
phase_11_geometry_factorial/alexnet_geo_s4_p3_fc
phase_11_geometry_factorial/alexnet_geo_s2_p3_fc
phase_11_geometry_factorial/alexnet_geo_s4_p2_fc
phase_11_geometry_factorial/alexnet_geo_s2_pk3n2_fc
phase_11_geometry_factorial/alexnet_geo_s2_pk2n3_fc
phase_11_geometry_factorial/alexnet_geo_s4_p3_gap
phase_11_geometry_factorial/alexnet_geo_s2_p2_drop_fc
phase_11_geometry_factorial/alexnet_geo_s4_p3_fc_k3
phase_11_geometry_factorial/alexnet_tv
phase_11_geometry_factorial/alexnet_adapted_orig_fc_pt
phase_11_geometry_seeds_s43/alexnet_3x3_fc
phase_11_geometry_seeds_s43/alexnet_3x3_gap
phase_11_geometry_seeds_s43/alexnet_adapted_orig_fc
phase_11_geometry_seeds_s43/alexnet_adapted_orig_gap
phase_11_geometry_seeds_s44/alexnet_3x3_fc
phase_11_geometry_seeds_s44/alexnet_3x3_gap
phase_11_geometry_seeds_s44/alexnet_adapted_orig_fc
phase_11_geometry_seeds_s44/alexnet_adapted_orig_gap
phase_11_reuse_old_init/alexnet_tv_3x3
EOF
fi

for run in "${RUNS[@]}"; do
  exp=${run%/*}; m=${run#*/}; src=outputs/pcad/$run; dst=$ARCHIVE/$run
  [[ -f "$src/checkpoints/${m}_best.pth" ]] || { echo "SKIP $run: no FP32 checkpoint"; continue; }
  if [[ -e "$dst/checkpoints/qat_${m}_best.pth" ]]; then echo "SKIP $run: already archived"; continue; fi
  mkdir -p "$dst/checkpoints" "$dst/results" "$dst/logs"
  for f in "checkpoints/qat_${m}_best.pth" "checkpoints/qat_${m}_resume.pth" "checkpoints/qat_${m}.pth" \
           "results/${m}_qat_val_logits.npz" "results/${m}_int8_val_logits.npz" "logs/qat_${m}.log"; do
    if [[ -e "$src/$f" ]]; then mv "$src/$f" "$dst/$f"; fi
  done
  cp "$src/results/${m}_summary.json" "$dst/results/"
  python -m scripts.cluster submit --experiment "$exp" --model "$m" --runtime pcad --slurm tupi_4090
done
