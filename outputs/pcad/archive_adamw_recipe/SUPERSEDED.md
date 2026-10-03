# Superseded: Phase 11 runs on the old recipe (archived 2026-10-03)

Every run under this directory was trained before Phase 11's single literature-referenced protocol (commits 8fb9d38 +
b752d6c, 2026-10-03): AdamW lr 3e-4 / wd 5e-4, batch 64, RandomResizedCrop(0.7-1) + 15° rotation + AutoAugment,
PyTorch's default init in some cells, QAT 100 epochs at 1e-5 with fbgemm's 7-bit activations, accuracy reported on the
90/10 validation split. **No number here is comparable with a current Phase 11 run** (those carry
`quant_protocol: "2026-10-03"` in their summary), and none of these experiment or model names is used by the current
design.

- Current Phase 11 runs: `outputs/pcad/phase_11_*`, one directory per experiment of `configs/experiments/phase_11_*.yaml`,
  models named by what they are (`ml/model_registrations.py`, `CELL_FACTORS`).
- `scripts/build_runs_index.py` indexes runs below this file with `superseded=archive_adamw_recipe` and no phase;
  `scripts/phase11/*` read only `outputs/pcad/phase_11_*`; `scripts/train.py` refuses to resume a run directory whose
  `resolved_config.json` has another protocol (`refuse_foreign_run_dir`).
- Known laptop/PCAD difference, on purpose: 15 runs (alexnet_adapted_orig_gap, alexnet_mixed, alexnet_mixed_fc,
  alexnet_smallkernel_fc, alexnet_stacked, alexnet_tv_2x2, alexnet_tv_3x3, alexnet_tv_scratch, vgg16, alexnet_2x2_gap,
  alexnet_3x3_gap, alexnet_smallkernel, alexnet_tv_mixed_alt/_early2/_early3) had their QAT redone on PCAD on 2026-09-30
  to 10-02 (fused-QAT fix, code d2f4fdc/eb2557d) and those files were never committed: PCAD holds the redone QAT/INT8
  (summary, QAT meta, layer_stats, resolved_config, `qat_*.pth.gz` + the kernel_size_comparison aggregate CSV -- 74
  files, shown as modified there), git the pre-fix one (unfused QAT). Same FP32 model in both; both versions superseded.
- Checkpoints and logs (gitignored) exist only on PCAD, at this same path under `~/dnn_study`. Older superseded Phase 11
  states, PCAD only: `archive_old_init/` (default init), `archive_unfused_qat/` (QAT fused on the caller's model),
  `archive_quantized_logits_qat/` (8-bit logits grid).
- Why each change: `docs/logs/PHASE11_LOG.md`, "Literature-standard INT8 + held-out test set", "One recipe, every number
  from a reference" and "Reduced design, descriptive names". The curated tables/figures built from these runs:
  `results/archive_adamw_recipe/`.
