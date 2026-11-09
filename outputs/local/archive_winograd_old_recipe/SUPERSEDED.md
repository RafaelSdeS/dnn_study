# Superseded: Winograd-FPGA accuracy runs on the old recipe (archived 2026-11-09)

`budget_unico` (14 `*_fpga` nets, FP32 + `qat_wino`, no packing) and `wino_f{23,43,63}_pack` (`qat_wino` alone at the packed
iso-DSP point, 14 nets + 2 Phase 11 cells) were trained on Winograd-FPGA's recipe: AdamW, FP32 of up to 100 epochs, 15 QAT
epochs, accuracy on the 90/10 validation split, the old INT8 definition. **No number here is comparable with a Phase 14
run** (Phase 11's one recipe, 500 ep FP32, QAT 50 ep, accuracy on the official test split).

- Replaced by `outputs/pcad/phase_14_winograd_{fp32,f23_pack,f43_pack,f63_pack}/` (48 of 48 arms, 2026-11-09): median
  arm - FP32 on the test set F23 -0.16, F43 -6.30, F63 -40.09 pp (here: +0.05, -5.1, -18.9). Table:
  `results/phase_14_winograd_numerics/`; log: `docs/logs/PHASE14_LOG.md`.
- `scripts/build_runs_index.py` indexes runs below this file with `superseded=archive_winograd_old_recipe` and no phase.
  The yamls stay in `configs/experiments/` (`budget_unico.yaml`, `wino_f*_pack.yaml`) for provenance; do not resubmit them
  into these directories.
- `scripts/phase14/f63_collapse.py` still reads this directory's `budget_unico/<net>/checkpoints/<net>_best.pth` as the
  old FP32 base (PCAD only; the laptop keeps no checkpoints).
- This is the laptop's share: the 2026-09-18 local M7 runs (`wino_f{23,43,63}_pack`, `_pcad_fp32`). Their gitignored files (26 checkpoints) were copied to PCAD's `~/dnn_study` on 2026-10-30 (md5 identical) and are here too.
