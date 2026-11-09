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
- 2026-11-07 (evening): the arms wait for a free 4090, not for our own queue: tupi had 5 nodes allocated to other users (1 down), `beagle` was busy with the 16 h `alexnet_final_bottleneck_residual_fpga` FP32,
  and the other PCAD GPUs do not fit (`grace` = 2 aarch64 nodes with L40S, the x86 `.venv` does not run there; `shared` = K20m, sm_35, below what torch 2.5.1+cu121 supports). A first arm job, vgg16 F23
  (835538), had a Slurm start estimate of 2026-11-08 14:05. An arm takes about 28 min on a 4090 (canary), so the user OK'd moving runs: the 6 pending arm jobs (835538 835539 835553 835558 835564 835570)
  went from QOS `normal` to QOS `short` with `scontrol update ... QOS=short TimeLimit=02:00:00` (priority component 3000 -> 30000, start reason `Priority` -> `Resources` for the first one), partition
  `tupi` unchanged, `requeue` and the pre-timeout stop kept so an arm that overruns 2 h resumes from its epoch checkpoint. The 33 queue lines now call `~/submit_arm.sh <experiment> <model>` (outside the repo;
  backup `queue_phase14.txt.bak_20261107b`): same `scripts.cluster submit --slurm tupi_4090` as before, then the same `scontrol update`, best effort (QOS `short` caps submits at 25 per user; a job that does not
  fit stays in `normal` and the loop's pass retries). No repo file changed for this.
- 2026-11-07 (evening, later): the second-wave pass no longer depends on the laptop. It runs on Hestia (no GPU, up 12 days) in `tmux` session `phase14`: `~/phase14_loop.sh` calls the pass (`~/phase14_second_wave.sh`,
  piped over ssh, nothing installed on PCAD's login node) every 30 min, logs to `~/phase14_loop.log` on Hestia, and stops by itself on `ALLDONE`. Hestia reaches PCAD with a new dedicated key
  `~/.ssh/id_pcad_phase14` (ed25519, no passphrase, host alias `pcad-phase14`), authorized with a full shell on the PCAD account at the user's choice (backup `~/.ssh/authorized_keys.bak_20261107` on PCAD;
  revoke by deleting the `hestia-phase14-loop` line there). The old `~/.ssh/id_pcad_backup` (rrsync, read-only) is untouched. The laptop loop was stopped so two writers never touch the queue. Pass fix found by
  the dry run: arm jobs are named `train-<net>` like the FP32 job, so a net already seeded now counts as done before squeue / sacct are consulted (else `ALLDONE` would wait for every arm to leave the queue).
  First pass from Hestia: 4 nets already queued, `alexnet_final_bottleneck_residual` and `alexnet_final_fire_residual` waiting for their FP32.
- 2026-11-07 (night): **14 of 14 FP32 done** (all 500 of 500 epochs, best epoch 408-497, `quant_protocol` 2026-10-07). The last 6, test top-1: `googlenet_fpga` 61.85, `wrn_28_2_fpga` 60.32,
  `wrn_16_4_fpga` 58.79, `alexnet_fire_bypass_fpga` 55.19, `alexnet_final_fire_residual_fpga` 53.69, `alexnet_final_bottleneck_residual_fpga` 53.11 (835535, beagle, 13 h 48 min). Synced (6 files per
  run: no `*.pth`, no logs); `runs_index.csv` +6 rows with the 54 archive rows merged back from git; `build_cross_phase_results` unchanged (it reads `results/phase_*` only). The Hestia loop
  stopped itself on `ALLDONE` (~21:11): every net's arms are seeded and queued. `scripts.phase14.winograd_numerics` run for the first time on the 3 canary arms:
  `results/phase_14_winograd_numerics/{runs,medians}.csv`, n = 1 per variant, F23 +3.79 / F43 +0.47 / F63 −10.48 pp against the FP32 recomputed from the logits (31.53).
  Arms at 21:38: 3 of 48 done, 2 running (`resnet18_fpga` F43 / F63), 29 pending (21 QOS `short`, 8 `normal`), 14 lines still in `~/queue_phase14.txt`.
  **The 28 min per arm was the AlexNet canary's, not every net's:** `resnet18_fpga`'s arm runs ~209 s per epoch, ~2.9 h for 50, over QOS `short`'s 2 h. Its F23 arm (835558) took the
  pre-timeout stop at epoch 30 of 50 (21:30) and was requeued (`Restarts=1`; it resumes from `qat_wino_resnet18_fpga_resume.pth`), as set up above. A clean stop, not a signal death, so resuming
  is within the rule. Expect the arms of the heavier nets to pass through the queue twice; left as is.
- 2026-11-08 (morning): **26 of 48 arms done**, all 50 of 50 epochs (the heavy nets' arms after one requeue + resume each). The queue had stalled since 2026-11-07 16:08: the `scontrol update` in
  `~/submit_arm.sh` silently left 8 arms (835594-835630: `vgg_style` F63, `alexnet_3x3_fc` x3, `wrn_16_4` x3, `alexnet_fire_bypass` F23) in QOS `normal`, pending on `Priority` (tupi full, tupi4 down),
  which filled the feeder's `MAX=8` and blocked the 14 remaining arm lines and the 7 Stage B lines behind them. The same `scontrol update ... QOS=short TimeLimit=02:00:00` by hand (user OK'd) moved all 8;
  the feeder resumes on its next pass. The 26 finished arm dirs synced (no `*.pth`, no logs; 234 files), the 3 arm aggregate CSVs taken from PCAD (the laptop copies differed only in float formatting),
  `winograd_numerics` rerun, `runs_index.csv` +46 rows with the 54 archive rows merged back, `build_cross_phase_results` unchanged. Medians, arm − FP32 on the test set:
  **F23 −0.16 pp (n = 9, −0.38..+3.79), F43 −4.63 (n = 9, −8.21..+0.47), F63 −50.82 (n = 8, −58.79..−10.48)**. F63 puts the deep nets at chance: `resnet18_fpga` 0.56, `vgg13_fpga` 0.48,
  `vgg16_k3x3...` 0.43 (agreement with FP32 ≈ 0.005); the AlexNet-class nets keep 11-24 %.
  **The F63 collapse is the arm's numerics on this recipe's nets, not a pipeline bug.** `scripts/phase14/f63_collapse.py` (-> `results/phase_14_winograd_numerics/f63_collapse.json`;
  `resnet18_fpga` on the CPU, bridge f44da9a copied from PCAD, the arm yamls' packing, observers calibrated on 400 train images with BN frozen as the arm's first epoch does, no `qat_wino` training)
  against the same net's old `budget_unico` AdamW FP32, whose F63 arm still reached 37.5 % (val, 15 epochs):
  (1) Saturation is negligible in every arm, F63 included (`sat_frac` ≤ 0.005 in the summaries' calibration dump).
  (2) Both bases are at chance in F63 **before** any `qat_wino` epoch (first 1000 test images: new 0.3 %, old 0.1 %; F43 new 5.9 %, old 37.9 %; F23 53.5 / 51.7 vs FP32 56.5 / 53.7).
  (3) Each F63 conv alone, its observer set from the FP32 net's own input, is about as noisy on both bases: cosine with the FP32 conv 0.46-0.96 new, 0.75-0.96 old (F43 0.56-0.99 / 0.90-0.99;
  the new base's worst is `layer4.1.conv2`, whose input has crest factor absmax / rms 65 vs 13).
  (4) What differs is the error compounding into the observers: calibrated on the Winograd net's own activations, the F63 `act_absmax` sits 14-51x above the clean signal from `layer1.1.conv2` on
  in the new base, 1.0-3.7x in the old (F43: up to 2.6x vs ≤ 1.2x). So in the new base only 2-57 % of the nonzero inputs keep a nonzero INT8 code (old 91-99 %), and of the transformed tiles U, which
  the packing shifts down by the analytic worst case `su` (|x| = 127, `u_w` = 9), under 2 % survive (old 26-79 %): 13 of the 19 convs pass almost no signal, and the arm stays at chance
  (train loss ≈ ln 200 from epoch 1 to 50). V (weights) survives in both (≥ 0.91). Why the error compounds so much faster on this recipe's nets is not isolated; their higher activation crest factor
  in several layers fits, untested. The bridge notes recalibrated scales (`hw_params`) as the one measured lever for packed F43 / F63.
  For the article: on deep nets F63 reads as a collapse, not a −18.9 pp median; F23 ≈ lossless and F43 ≈ −5 pp hold.
- 2026-11-09: **48 of 48 arms done** (16 nets × F23 / F43 / F63, all 50 of 50 epochs; the last job, a `wrn_28_2_fpga` arm, ended 2026-11-09 01:16; every job of the last 2 days is COMPLETED, exit 0, in `sacct`).
  The 22 new arm dirs synced (no `*.pth`, no logs; the three arm trees are 229 MB each), the 3 arm aggregate CSVs taken from PCAD (17 lines each), `winograd_numerics` rerun, `runs_index.csv` +22 rows
  (no key lost), `build_cross_phase_results` unchanged, `tests/test_phase14_winograd.py` 3 passed. Medians, arm − FP32 on the test set, n = 16 each:
  **F23 −0.16 pp (−0.69..+3.79; 5 nets above zero), F43 −6.30 (−8.57..+0.47), F63 −40.09 (−59.84..−10.48)**. Against the 26-arm medians (−0.16 / −4.63 / −50.82) F43 got worse and F63 better
  because the added nets are slower to collapse for F63 (4 AlexNet-class nets and `googlenet` at 12-49 %; `vgg_style` and the two WRNs are at chance) but sit at −6 to −9 pp in F43. F63 arm top-1: six nets under 1 % (`vgg_style` 0.41, `vgg16...` 0.43, `vgg13` 0.48, `wrn_28_2` 0.49, `wrn_16_4` 0.51, `resnet18` 0.56; agreement with FP32 ≈ 0.005),
  `repvgg_a0` 4.21, AlexNet-class 11.21-27.81, `googlenet` 48.51 (best, agreement 0.54). The collapse measured on `resnet18` (2026-11-08) therefore holds for the plain-conv deep nets as a class; `googlenet` and the AlexNets survive.
  The article's `\pendente` in "2x2 ou 3x3?" carries the 48-arm numbers, its text still the budget_unico ones. Phase 13 Stage B (7 jobs, 835668-835674, `phase_13_bottleneck_fire`) is pending on `tupi` (`Priority`); nothing else of ours runs on PCAD.
