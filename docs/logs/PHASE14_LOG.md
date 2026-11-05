# Phase 14 — Winograd-FPGA numerics on the one recipe

Slug `phase_14_winograd_numerics`. An extra study, kept apart from Phases 11–13 (nothing there reads these runs).

## Why (2026-11-05)

The article on Phases 11–13 (`report/artigo_kernel.tex`, "2×2 ou 3×3?") cites the accuracy cost of the Winograd-FPGA
accelerator numerics: F(2,3) +0.05, F(4,3) −5.1, F(6,3) −18.9 pp vs FP32, median over 14 nets (PHASE11_LOG "M7
extension"). Those numbers come from the old protocol: `budget_unico` trained the FP32 nets up to 100 epochs with AdamW
and early stopping, and `wino_f{23,43,63}_pack` fine-tuned `qat_wino` 15 epochs at lr 5e-5, measured on the validation
split. The user asked for everything on one recipe, so this phase reruns them on `_protocols/no_patience.yaml`.

## Design

- **FP32** — `phase_14_winograd_fp32`: the 14 `*_fpga` nets of `budget_unico` (the M7 pair `alexnet_fire_bypass_fpga` /
  `vgg_style_fpga` included), stage `fp32` only, exactly the Phase 11 recipe (500 epochs, SGD lr 0.01, batch 128,
  crop-flip-AutoAugment, no early stopping). They have no fbgemm QAT path; their quantized contract is `qat_wino`.
- **Arms** — `phase_14_winograd_f{23,43,63}_pack`: stage `qat_wino` only, on the reused FP32, so the three arms differ in the
  numerics alone. `qat_wino` = the recipe's QAT (50 epochs, same SGD, lr 1e-4 cosine to 1e-6, Wu et al. 2020 App. A.2),
  packed, `(u_w, v_w, k_dsp) = (9, 8, 2)` as in M7. 16 nets: the 14 above plus the one-recipe twins of M7's two Phase 11
  nets, whose FP32 already exists — `alexnet_k3x3_stride4_3pool3x3_map1_fcdrop_nobn` (`phase_11_dropout_pretraining`,
  for `alexnet_tv_3x3`) and `vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn` (`phase_11_vgg_kernel_head`, for `vgg16`).
- `tests/test_config.py::test_phase_14_runs_the_phase_11_recipe_on_the_winograd_numerics` holds the four files to that.

## Code changes it needed

- `scripts/train.py`, `qat_wino` stage: passes `eta_min` (it was dropped, so the cosine went to the FP32 floor, 0), and
  evaluates on the held-out test set like every other stage (`test_qat_wino_*` in the summary, `test_eval` and
  `agreement_fp32_qat_wino_test` in `<model>_qat_wino_summary.json`). Old yamls: unchanged behaviour (eta_min default 0).
- `ml/winograd_bridge.py`: every `*_fpga` net starts from `models/baselines.py:he_init`, the one recipe's init (they kept
  the sibling repo's before). Loading a checkpoint is unaffected.

## How it runs (PCAD)

- Bridge: `WINOGRAD_FPGA_ROOT=~/winograd_bridge_m7/avaliacao_redes` (f44da9a). Its default numerics are M7's (fde71c7; f44da9a
  only adds the opt-in `hw_params`). Never `~/winograd_bridge` (156e5a1): older, no packing. The export goes on each queue line.
- FP32 reuse: when a net's FP32 finishes, copy `checkpoints/<m>_best.pth`, `results/<m>_fp32_{val,test}_logits.npz` and
  `results/<m>_summary.json` into each arm's `outputs/pcad/phase_14_winograd_f*_pack/<m>/` (the loop is in the arm yamls),
  then queue its three arms. The two Phase 11 nets' arms go first: they are the canary of the `qat_wino` path.
- Cost, from the old runs' time per epoch: ~115 GPU-h FP32 + ~100 GPU-h arms; `--slurm tupi_4090`.

## When the arms land

- A short `scripts/phase14/winograd_numerics.py`: arm − FP32 on the test set per net, median per variant, CSV.
- Update the article's "2×2 ou 3×3?" paragraph (its `\pendente` mark).
- Archive `budget_unico/` and `wino_f*_pack/` (PCAD and `outputs/local`) to `archive_winograd_adamw_recipe/` with a
  `SUPERSEDED.md`. The old yamls stay: the sibling study still runs them from `~/dnn_study_m7`.

## Status

- 2026-11-05: configs, code and tests in; submission queued on PCAD (`~/queue_phase14.txt`).
