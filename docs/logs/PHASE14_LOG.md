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
- 2026-11-05: smoke job 835396 (`alexnet_fire_fpga`, FP32, `--smoke`) COMPLETED in 4 min 45 s on tupi1 with the M7 bridge (it only tests the FP32 stage; the `qat_wino` path
  is tested by the canary arms). Feeder started 21:33 (`QOS=normal`, MAX 8): first the 4 Phase 13 3x3 seeds (835433, 835435-835437), then the 3 canary arms of the Phase 11
  `alexnet_k3x3_stride4...` cell (835438-835440, F23 / F43 / F63).
- 2026-11-06 09:20: `~/queue_phase14.txt` reordered (backup `.bak_20261106`, content checked): the 14 FP32 first -- heavy nets (`resnet18`, `vgg13`, `vgg_style`, `repvgg_a0`, `wrn_16_4`,
  `wrn_28_2`, `googlenet`) on `tupi_4090` alternating with the 7 `alexnet_*_fpga` on `tupi_beagle` (partition `tupi,beagle`; beagle was idle while tupi was saturated by other users) --
  and the 3 `vgg16` canary arms last. The FP32 are the critical path: the 42 remaining arms (14 nets x 3) are queued by hand after each FP32 lands. Nothing of Phase 14 has finished
  beyond the smoke. `exclude: tupi4` stays (PHASE13_LOG "2026-11-06"): the 4090 nodes tupi3 / tupi5 / tupi6 had crashes too, so a Phase 14 run killed by a signal is rerun, not resumed.
- 2026-11-06 (later): the 3 canary arms of `alexnet_k3x3_stride4_3pool3x3_map1_fcdrop_nobn` are done and synced (`test_qat_wino_top1`, `agreement_fp32_qat_wino_test` and `git_dirty: false`
  present in each `qat_wino` summary): **F23 35.32 (agreement 0.613), F43 32.00 (0.521), F63 21.05 (0.306)**. This Phase 11 cell has FP32 31.52 but QAT 35.11 / INT8 35.08 (QAT adds +3.6 pp), so
  read the arms against INT8 as well: +0.24 / −3.08 / −14.03 pp, the same order as M7's median (F23 ~0, F43 −5.1, F63 −18.9 vs FP32). One net, not yet a median. The 3 `vgg16...` arms still wait in the queue
  (their dirs hold only the copied FP32 files, not synced); the 14 FP32 are running / queued (`resnet18_fpga` tupi3, `alexnet_stacked_fpga` tupi6, the rest pending on the 10-job QOS cap).
- 2026-11-07: **8 of the 14 FP32 are done** (jobs COMPLETED; test top-1): `resnet18_fpga` 59.22, `repvgg_a0_fpga` 59.09, `vgg13_fpga` 58.37, `alexnet_stacked_fpga` 58.00, `vgg_style_fpga` 54.97,
  `alexnet_bottleneck_fpga` 54.59, `alexnet_fire_fpga` 54.57, `alexnet_3x3_fc_fpga` **37.86**. The last is about 17-20 pp under the other AlexNet-class nets, but it is not a failed
  training: 500 epochs, train acc 99.80 %, val 38.27 % (best 38.46 at epoch 408), val loss rising 3.51 -> 3.64, 57.6 M parameters, 96 % of them in the three Linear layers (37.8 M + 16.8 M + 0.8 M; layer stats) -- an FC-head net that overfits (train acc 99.8 % against val 38 %), not a
  broken run; the old `budget_unico` run (AdamW recipe) gave 36.24. Its three arms stay in the queue. The other 6: `wrn_16_4` (835506, tupi3), `wrn_28_2` (835527, tupi1), `alexnet_fire_bypass` (835526, tupi5) and
  `alexnet_final_bottleneck_residual` (835535, beagle) running; `googlenet` (835536) and `alexnet_final_fire_residual` (835537) pending.
  **Second wave for the 8:** each net's `<m>_best.pth`, `<m>_fp32_{val,test}_logits.npz` and `<m>_summary.json` copied (`cp -n`) into the three arm dirs and compared with `cmp` (24 of 24 identical);
  24 arm lines (`--slurm tupi_4090`, `WINOGRAD_FPGA_ROOT` exported on the line) appended to `~/queue_phase14.txt` (backup `.bak_20261107`), which now has 25 lines with the `vgg16` F63 arm at the head.
  The `vgg16` F23 / F43 arms (835538 / 835539) are pending on priority. The feeder (PID 1912681, `QOS=normal`, MAX 8) is alive; the `normal` QOS is full with the FP32 jobs, so the arms start as they end.
  Still by hand: the same copy and 3 queue lines for each of the 6 nets above when its FP32 lands. No arm beyond the 3 canaries has finished.
- 2026-11-07 (later): the second wave of the 6 remaining nets is automated for this session: an idempotent pass (`ssh … 'bash -s' < phase14_second_wave.sh`, kept in the Claude scratchpad, not in the
  repo) run by a self-paced loop every ~30 min. A net is seeded only when its `train-<m>` job left `squeue`, `sacct` says COMPLETED and its summary has `test_fp32_top1`; it then `cp -n`s the FP32
  files into the 3 arm dirs (`cmp`-checked) and appends the 3 arm lines to `~/queue_phase14.txt` unless they are already in it / `.done` / `.failed`. A FAILED / TIMEOUT / CANCELLED FP32 job is
  reported, never resubmitted. If the session closes the loop stops; the pass is idempotent, so the copy can be redone by hand.
- 2026-11-07 (later): the 8 finished `phase_14_winograd_fp32` runs are synced to the laptop (no `*.pth`, no logs; 58 MB) and `runs_index.csv` has their 8 rows (the 54 archive rows merged back).
  `scripts/phase14/winograd_numerics.py` (+ `tests/test_phase14_winograd.py`, 2 passed on a fake tree) writes `results/phase_14_winograd_numerics/{runs,medians}.csv`: per net and variant the arm's
  test top-1 minus the FP32 top-1 recomputed from `<m>_fp32_test_logits.npz`, and the median per variant with its `n`. Not run on the real runs yet (only the 3 canary arms exist).
