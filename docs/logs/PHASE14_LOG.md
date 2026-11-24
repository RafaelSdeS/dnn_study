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
  in several layers fits, untested. (2026-11-11: it does not explain it -- `googlenet_fpga`'s max crest is 30.7, above `vgg13_fpga`'s 22.7, and it does not compound.) The bridge notes recalibrated scales (`hw_params`) as the one measured lever for packed F43 / F63.
  For the article: on deep nets F63 reads as a collapse, not a −18.9 pp median; F23 ≈ lossless and F43 ≈ −5 pp hold.
- 2026-11-09: **48 of 48 arms done** (16 nets × F23 / F43 / F63, all 50 of 50 epochs; the last job, a `wrn_28_2_fpga` arm, ended 2026-11-09 01:16; every job of the last 2 days is COMPLETED, exit 0, in `sacct`).
  The 22 new arm dirs synced (no `*.pth`, no logs; the three arm trees are 229 MB each), the 3 arm aggregate CSVs taken from PCAD (17 lines each), `winograd_numerics` rerun, `runs_index.csv` +22 rows
  (no key lost), `build_cross_phase_results` unchanged, `tests/test_phase14_winograd.py` 3 passed. Medians, arm − FP32 on the test set, n = 16 each:
  **F23 −0.16 pp (−0.69..+3.79; 5 nets above zero), F43 −6.30 (−8.57..+0.47), F63 −40.09 (−59.84..−10.48)**. Against the 26-arm medians (−0.16 / −4.63 / −50.82) F43 got worse and F63 better
  because the added nets are slower to collapse for F63 (4 AlexNet-class nets and `googlenet` at 12-49 %; `vgg_style` and the two WRNs are at chance) but sit at −6 to −9 pp in F43. F63 arm top-1: six nets under 1 % (`vgg_style` 0.41, `vgg16...` 0.43, `vgg13` 0.48, `wrn_28_2` 0.49, `wrn_16_4` 0.51, `resnet18` 0.56; agreement with FP32 ≈ 0.005),
  `repvgg_a0` 4.21, AlexNet-class 11.21-27.81, `googlenet` 48.51 (best, agreement 0.54). The collapse measured on `resnet18` (2026-11-08) therefore holds for the plain-conv deep nets as a class; `googlenet` and the AlexNets survive.
  The article's `\pendente` in "2x2 ou 3x3?" carries the 48-arm numbers, its text still the budget_unico ones. Phase 13 Stage B (7 jobs, 835668-835674, `phase_13_bottleneck_fire`) is pending on `tupi` (`Priority`); nothing else of ours runs on PCAD.
- 2026-11-09 (later): the article's "2x2 ou 3x3?" paragraph rewritten on the 48 arms (medians, the F63 collapse of the deep nets, the observer-compounding explanation kept and limited to resnet18; the `\pendente` is gone, a limitation added); PDF rebuilt (19 pages).
  `budget_unico` (14 runs) and `wino_f{23,43,63}_pack` archived with their aggregate CSV: `outputs/pcad/archive_winograd_old_recipe/` and, for the 2026-09-18 local M7 runs, `outputs/local/archive_winograd_old_recipe/`, each with a
  `SUPERSEDED.md`; `build_runs_index` marks 126 rows `superseded=archive_winograd_old_recipe`; `f63_collapse.py` reads the old base from the new path. The configs stay for provenance. PCAD still has the gitignored files at the old
  paths (move them there after it pulls). The `f63_collapse` rerun on another net was not done: it needs the bridge (f44da9a), the new and the old FP32 checkpoint of that net and the dataset on one machine.
- 2026-11-10: PCAD pulled main (`4c114ce`): 405 untracked `phase_14_winograd_*_pack` files and 3 aggregate CSVs moved aside, `git pull --ff-only`, all cmp-identical to the checkout, copies deleted. `budget_unico`'s 75 gitignored checkpoints (+ logs, + the PCAD-only `vgg13_fpga_broken_novbn_20260914`) moved to `outputs/pcad/archive_winograd_old_recipe/budget_unico/` (`mv`, same filesystem); `wino_f*_pack` had no checkpoints. `f63_collapse.py`'s default old path now exists on PCAD. The rerun on another net (`vgg13_fpga`, `googlenet_fpga`) is not done: it needs a CPU Slurm job (the user's go).
- 2026-11-11: **`f63_collapse` on `vgg13_fpga` and `googlenet_fpga`** (CPU, QOS `short`, tupi2: 836097 8:25 / 836098 7:01, COMPLETED; bridge f44da9a, clean). The 2026-11-10 submission
  (835987 / 835988) had failed: `source` under `/bin/sh`, and `--new/--old` default to `resnet18_fpga`'s dirs -- both now passed explicitly (`sbatch --wrap ". .venv/bin/activate && ... --net <n>
  --new outputs/pcad/phase_14_winograd_fp32/<n>/checkpoints --old outputs/pcad/archive_winograd_old_recipe/budget_unico/<n>/checkpoints --out ~/f63_out/f63_collapse_<n>.json"`). The JSONs copied to
  `results/phase_14_winograd_numerics/f63_collapse_<net>.json` (md5 = PCAD's), `~/f63_out` removed. All three nets are at chance in F63 at epoch 0 (new 0.1-0.3 %, old 0.1-0.6 %); what differs is the compounding:
  `vgg13_fpga` compounds like `resnet18_fpga` (observers 11.6-34.2x the clean signal in 5 of 10 convs from the 6th, `x_kept` 21-60 %, `u_kept` 0.1-2.4 %; old base ≤ 4.45x) and stays at chance (arm 0.48 %);
  `googlenet_fpga`, same recipe, does not (≤ 4.32x everywhere, `x_kept` ≥ 89 %, `u_kept` ≥ 29 %) and its arm recovers to 48.51 %. So the epoch-0 accuracy does not predict the arm; whether the signal
  crosses the net does. Article: the "A diferença não é falha de treino" paragraph now covers the three nets; the limitation keeps only "escalas recalibradas não testadas". Next (user's plan, same day):
  `f63_collapse --hw-params f63=f63c` (the calibrated netlist scales, `gate_flags/params_f63c.txt`) on the three nets, and one `qat_wino` arm of `resnet18_fpga` with `hw_params: f63c`.
- 2026-11-11 (later): submitted on PCAD (4636092). `f63_collapse --hw-params f63=f63c` (F63 only, the netlist's calibrated scales; `sv` from
  `sv_hw`, `su` from the netlist; tests/test_phase14_winograd.py covers it) on the three nets, CPU, QOS `short`: 836176 (`resnet18_fpga`),
  836177 (`vgg13_fpga`), 836178 (`googlenet_fpga`) -> `~/f63_out/f63_collapse_<net>_f63c.json`. The QAT arm `phase_14_winograd_f63c_pack`
  (`resnet18_fpga`, the f63 arm with `hw_params: f63c`, outside the 48-arm medians; FP32 copied in as its yaml says) is in
  `~/queue_followups.txt` (`feed_queue.sh`, QOS normal, `--slurm tupi_4090`), fifth of nine. The epoch-0 run says whether the signal now
  crosses the nets; the arm, whether resnet18 leaves chance.
- 2026-11-13: **`f63c` at epoch 0: with the netlist's calibrated scales the signal crosses all three nets.** 836176-836178 COMPLETED 2026-11-11 19:17-19:20
  (tupi6, QOS short, 1-3 min each, bridge f44da9a, clean); the three JSONs synced to `results/phase_14_winograd_numerics/f63_collapse_<net>_f63c.json` (md5 = PCAD's).
  Same measurement as 2026-11-08 / 11-11 (400 calibration train images, first 1000 test images, observers calibrated on the Winograd net's own activations,
  no `qat_wino` epoch), F63 only, `su` / `sv` / `align_base` from `gate_flags/params_f63c.txt` instead of the analytic worst-case shift. Top-1, first 1000 test images:

  | net | base | FP32 | F63 default | F63 `f63c` | F43 default | F23 default |
  |---|---|---|---|---|---|---|
  | `resnet18_fpga` | new (one recipe) | 56.5 | 0.3 | **23.7** | 5.9 | 53.5 |
  | | old (AdamW) | 53.7 | 0.1 | 37.0 | 37.9 | 51.7 |
  | `vgg13_fpga` | new | 56.5 | 0.1 | **16.4** | 3.3 | 54.8 |
  | | old | 48.7 | 0.6 | 28.4 | 21.8 | 46.2 |
  | `googlenet_fpga` | new | 60.9 | 0.3 | **38.1** | 35.4 | 60.5 |
  | | old | 54.8 | 0.5 | 37.8 | 38.8 | 54.3 |

  Per Winograd conv, new base, default -> `f63c` (range over the convs): `absmax_ratio` 1.0-51.2 -> 0.48-3.04 (resnet18), 1.0-34.2 -> 0.66-1.95 (vgg13),
  0.92-4.32 -> 0.66-2.11 (googlenet); `u_kept` 0.00-0.79 -> 0.66-0.96 over the three nets, convs passing under 5 % of U 13 of 19 -> 0 (resnet18) and 5 of 10 -> 0 (vgg13);
  `x_kept` ≥ 0.92; `cos_clean` 0.46-0.96 -> 0.80-0.99 (resnet18); `v_kept` about unchanged (0.91-0.99). No conv of any net or base has an observer over 4x the
  clean signal any more (default: 15 / 6 / 2 convs on the new base). So the analytic worst-case `su` was what starved U and started the compounding;
  with the calibrated shifts no conv starves, the observers stay near the clean signal, and on the new base F63 at epoch 0 is now above the default
  F43 at epoch 0 on all three nets (23.7 vs 5.9, 16.4 vs 3.3, 38.1 vs 35.4). The new base still sits under the old on resnet18 and vgg13 (23.7 vs 37.0,
  16.4 vs 28.4): some of the base difference is left, googlenet shows none. Epoch 0 on 1000 images, not an arm: where QAT takes `f63c` is the arm's
  question (`phase_14_winograd_f63c_pack`, `resnet18_fpga`, job 836205, pending on tupi by priority). The article's limitation "escalas recalibradas
  não testadas" can now say they were tested at epoch 0; its F(6,3) paragraph waits for the arm. `~/f63_out` on PCAD left in place.
  2026-11-14 (user OK): 836205, pending on Priority since 11-11, moved to QOS short (`scontrol update JobId=836205 QOS=short TimeLimit=04:00:00`;
  short's cap is now 4 h, priority 10000; the job sits at priority 32023, top of tupi). Revert: `QOS=normal TimeLimit=1-00:00:00`.
- 2026-11-16: **the `f63c` arm: with the netlist's calibrated scales, `resnet18_fpga` in F63 reaches 54.66 %.** 836205 COMPLETED 2026-11-14 12:46
  (tupi1, QOS short, 3:39 h, 50 of 50 `qat_wino` epochs, best epoch 41, git 05e6277); the run dir synced (no `*.pth`, no logs) with
  `outputs/pcad/aggregates/phase_14_winograd_f63c_pack_comparison.csv`; its copied FP32 logits are md5-identical to `phase_14_winograd_fp32`'s.
  Test top-1, `resnet18_fpga`, one seed: FP32 59.22, F23 59.00 (−0.22), F43 55.26 (−3.96), F63 default 0.56 (−58.66), **F63 `f63c` 54.66 (−4.56)**;
  agreement with FP32 0.67 (F43 0.64, F63 default 0.005). So the F63 collapse was the analytic worst-case shifts, not the tile size: with the
  calibrated ones F63 lands 0.6 pp under F43 (F43 still on the default scales -- `f43` has no calibrated set here). The arm stays outside the
  48-arm medians (`winograd_numerics` reads only `f{23,43,63}_pack`; not rerun, its inputs are unchanged); `runs_index` +2 rows (the FP32 copy
  and the arm, as for every arm). Article: abstract, the methods' Winograd paragraph (one extra arm), a new paragraph after "A diferença ... não é
  falha de treino" (epoch-0 numbers of 2026-11-13 + this arm), "2x2 ou 3x3?", the conclusion, the limitation ("só treinadas em uma rede e não
  testadas em F(4,3)") and future work; `tab:blocos` Winograd 62 -> 63. `~/f63_out` on PCAD still in place.
- 2026-11-16 (later, user's plan `~/.claude/plans/ticklish-wondering-blanket.md`, front B): **the calibrated scales on every net, F(4,3) too.**
  Correction to the entry above: `f43` does have a calibrated set. The bridge's `gate_flags/params_f43v3.txt` is Thomas's F(4,3) synthesis with the
  scale constants calibrated on 2026-10-26 (its header: "Netlist com as constantes de escala CALIBRADAS", tag iscas-2027; `ALIGN_BASE` 9 vs 12 in
  `params_f43`, `SU_FLAT`/`SV_FLAT` lower). `ml/yolov2.py` already names it as f43's calibrated line next to `f63c`, and
  `make_wino_conv("f43", pack=True, hw_params="f43v3")` builds and runs (checked on Hestia; it warns that `SU_FLAT` differs from the analytic `su` by up
  to 3 bits, as `f63c` does). New arm `phase_14_winograd_f43c_pack` (= `f43_pack` + `hw_params: f43v3`, the same 16 nets); `phase_14_winograd_f63c_pack`
  goes from `resnet18_fpga` to the same 16 nets (`resnet18_fpga` done, not resubmitted). `tests/test_config.py` now checks both calibrated arms with
  one rule: the base yaml plus `hw_params`. `scripts/phase14/winograd_numerics.py`: `ARMS` = the 3 analytic arms + `f43c` / `f63c`, each with its own
  rows and median (the 48-arm medians unchanged); figure 01 keeps 3 panels and draws a calibrated arm hollow in its base variant's panel, its median
  dotted. `VARIANTS` stays the 3 analytic arms (`f63_collapse` reads it). Not rerun yet: the outputs change once the arms land. 31 arms,
  ~3.5-4.5 h each (~125 GPU-h), queued behind the 8 Phase 11 FC + BN seeds (PHASE11_LOG "FC + BN column at 3 seeds").
- 2026-11-18: every one of the 31 calibrated arm dirs (`phase_14_winograd_{f43c,f63c}_pack`) holds its net's FP32 `*_best.pth` on PCAD; the feeder
  (`~/queue_followups.txt`) has submitted `f43c` on `vgg_style_fpga` (838470), `alexnet_stacked_fpga` (838483), `repvgg_a0_fpga` (838528), `vgg13_fpga`
  (838534), all pending on tupi behind the 4 Phase 11 FC + BN seed-44 runs; 27 arms left in the feeder (12 `f43c`, 15 `f63c`). None done yet.
- 2026-11-22: **all 31 calibrated arms are done: with the netlists' calibrated scales F(4,3) costs −0.83 pp (median over 16 nets) and F(6,3) −4.58, against −6.30 and −40.09 on the analytic scales.**
  `phase_14_winograd_{f43c,f63c}_pack`, 16 nets each (`resnet18_fpga` f63c from 2026-11-16); `qat_wino_summary.json` dated 2026-11-17..22 on PCAD, the last job
  (838976, `vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn`) ended 07:27. `squeue` empty; `~/queue_followups.txt` and `~/queue_phase14.txt` fully in `.done` (50 / 73 lines).
  Synced to the laptop: `rsync -rnc` of both pack dirs against PCAD (no `*.pth`/logs) lists 0 differences; `phase_14_winograd_{f43c,f63c}_pack_comparison.csv` hold the same
  16 rows as PCAD's. `scripts.phase14.winograd_numerics`, `build_runs_index` and `build_cross_phase_results` outputs regenerated (80 rows in `runs.csv` = 5 arms x 16 nets).
  Median d (arm − FP32, test top-1, pp), min..max: F23 −0.16 (−0.69..+3.79) · F43 −6.30 (−8.57..+0.47) · **F43c −0.83 (−1.54..+3.19)** · F63 −40.09 (−59.84..−10.48) · **F63c −4.58 (−6.60..+0.28)**.
  Per net, the calibrated arm beats its analytic arm on 16 of 16 (F43c > F43, F63c > F63); F23 ≥ F43c on 15 of 16 and F43c ≥ F63c on 16 of 16 (median F63c − F43c −3.62 pp).
  No net collapses on `f63c` (worst `wrn_28_2_fpga` −6.60; the 6 nets below −5 pp are `alexnet_final_fire_residual`, `alexnet_fire_bypass`, `alexnet_stacked`, `vgg16_k3x3…_bn`, `vgg_style`,
  `wrn_28_2`); the worst `f43c` is `vgg13_fpga` (−1.54). One seed per arm, so the medians carry no seed spread. The 11 jobs / array tasks that did not end COMPLETED since 2026-11-10 (8 TIMEOUT at 12 h, 2 `f63_collapse` failing in 1 s, 1 guard)
  are all superseded: the TIMEOUTs are the 12 h chunks of the `tr_C4_V10` / `tr_C4q_s1` / `tr_C4q_s2` arrays (each array's last task COMPLETED), `f63_collapse_{vgg13,googlenet}_fpga` re-ran 2026-11-11 (COMPLETED),
  `train-mobilenetv2` was re-run as `-scratch`. Not done: the article (`sec:numerica`, "2x2 ou 3x3?", abstract, limitation, conclusion, future work, `tab:blocos`) still carries the 17 pre-calibration numbers (TODO).
- 2026-11-22 (later): article updated with the 31 calibrated arms (`report/artigo_kernel/`, TODO): F43c −0,8 / F63c −4,6 pp (16 nets) in the abstract, contribution bullet, methods, `sec:numerica`
  (+ figure caption: hollow circles = calibrated, dotted line = their median), "2x2 ou 3x3?", limitations, conclusion; `tab:blocos` Winograd 63 -> 94. Future work: the "train the calibrated scales on F(4,3) and the other nets" item is gone (done).
- 2026-11-23: **why the packed F(6,3) error compounds in some nets and not in others -- probes added, the runs are pending.**
  *Mechanism, read in `qat_wino.py` (bridge f44da9a):* `su` is a constant per (variant, `u_w`) -- the analytic worst case with the input range fixed at 127 -- and does NOT follow the observer; the observer is an EMA (momentum 0.05)
  of the batch max |x|. So the loop is: error -> rare large activations -> observer max up -> x_q uses few INT8 levels -> U = Bt x B small -> the fixed `su` zeroes it (`u_kept`) -> more error.
  *Already in the tracked JSONs (new base, F63, default scales):* ln(observer / clean max) per Winograd conv grows +0.206 on `resnet18_fpga` (x1.23), +0.411 on `vgg13_fpga` (x1.51), -0.033 on `googlenet_fpga` (x0.97);
  `u_kept` < 0.2 from the 5th / 4th conv, never on googlenet; with `f63c` the observer / clean ratio tops at 3.0 / 1.9 / 2.1. Over the 16 nets the F63 arm is 0.4-0.6 % on the six deep plain-3x3 nets, 4.2 `repvgg_a0`,
  11-28 on the AlexNet-class nets (Fire / Bottleneck 21-28, `alexnet_stacked` 11.2), 48.5 `googlenet`. A correlation only: the number of 3x3 convs varies too.
  *Hypotheses* (plan `~/.claude/plans/glimmering-doodling-cascade.md`): A the max estimator is the weak link (error makes outliers; test `p999_ratio` vs `max_ratio` and a 99.9th-percentile observer), B the observer loop is necessary
  (observers calibrated on the FP32 activations), C the count of consecutive Winograd convs (Winograd on every second conv), D 1x1 reduce convs filter the error (relative drift at every conv, exact and Winograd).
  *Decision rules, fixed before the numbers:* `p999` takes resnet18 / vgg13 to >= ~20 % at epoch 0 (as `f63c`: 23.7 / 16.4) -> A; `clean` survives -> the loop is the cause, collapses -> the worst-case `su` per layer suffices;
  `every2` leaves the per-conv growth at ~x1.2 -> a local property of the layer, not of the chain; nothing separates -> keep the question open and report the table. Epoch 0 on 1000 images does not predict the arm
  (googlenet 0.3 -> 48.5 %), so the per-layer trajectories (`absmax_ratio`, `u_kept`, `drift`) are the criterion; saying where QAT goes needs an arm (not planned).
  *Code (5c8351c):* `scripts/phase14/f63_collapse.py` gets `--wino-on all|every2` (the others swapped back to exact FP32 convs), `--observer own|clean|p999`, `--variants`, `--new-only`; per layer `cin`, `drift`, `max_ratio`,
  `p999_ratio`, `obs_over_p999`; per variant `conv_gain` (relative drift at the input and output of every Conv2d); `main()` seeds torch (post_shift draws a 1M-element `randint` sample on large layers); defaults unchanged.
  `tests/test_phase14_winograd.py::test_f63_probe_helpers` (no bridge needed). Smoke-tested on a random net with real `qat_wino` layers (bridge 3a15cd0 on Hestia -- not for results).
  *Jobs (PCAD, CPU, no training):* `~/f63_probe/run.sh <net>` runs `default`, `every2`, `clean`, `p999`, `f63c` in sequence on one node -> `~/f63_probe/f63_probe_<net>_<mode>.json`. 840195 (resnet18) sat PENDING on `tupi` (64 of 64 CPUs taken,
  the tupi3-6 nodes have 8) and was cancelled; resubmitted with `-p shared,tupi --exclude=tupi4 -q short`: 840197 resnet18 (limit 1:30 -- a user cannot raise it; may time out on the last modes), 840199 vgg13, 840200 googlenet
  (3:00). 840197 runs on `bali2`, an **Intel Xeon E5-2650 (2012, AVX only)**; the 2026-11-08 / 11 JSONs ran on tupi nodes, so a mismatch in the regression may be the CPU, not the code (rerun `default` on a tupi node to tell).
  *Regression:* the `default` JSON must match `results/phase_14_winograd_numerics/f63_collapse.json` (`new.f63`): `absmax_ratio`, `x_kept`, `u_kept`, `v_kept`, crests to 1e-4, `cos_clean` / top-1 loosely (post_shift's random draw).
  *When they land:* rsync the JSONs to `results/phase_14_winograd_numerics/f63_probe_<net>_<mode>.json` (md5 = PCAD's), a table per net and mode, the decision rules above, then `sec:numerica` (replace "Não medimos por que...") and the future-work item;
  delete `~/f63_probe` on PCAD.
- 2026-11-23 (housekeeping): PCAD pulled main (`6f806a8`, then `5c8351c`): 376 untracked clashing files (FC + BN seeds 43 / 44, `f43c` / `f63c`) moved aside, `git merge --ff-only`, all 376 `cmp`-identical, copies deleted; the 3 aggregate CSVs
  PCAD had modified (re-evaluation, last-digit noise) are kept in `~/dnn_study_prepull_untracked_20261123/csv_modified/`. Article audit (read-only, ~190 numbers against `all_runs.csv` / `runs.csv` / `medians.csv`): no wrong result number;
  two imprecisions fixed in the article and four stale lines in `docs/plans/BEST_MODELS.md` (300a7eb). The 27 files that existed only on Hestia (Phase 7 SSD checkpoints, Phase 9 / legacy resume files, logs; 1.16 GB) copied to the same
  relpaths on PCAD, md5-identical.
- 2026-11-24: **the F(6,3) probes, closed (plan `~/.claude/plans/sleepy-toasting-cat.md`).** 840197 / 840199 / 840200 COMPLETED on `bali2` (1:09 / 1:04 / 1:03), all 15 JSONs written.
  *The regression above could not pass, and not because of a bug.* In all six pairs (3 nets x `default` / `f63c`) the new JSON equals the tracked one up to some layer and then drifts (resnet18 `default` from the 7th
  Winograd conv, `layer1.1.conv2`: `absmax_ratio` 14.36 vs 14.39, growing to 60.6 vs 51.2 at `layer4.1.conv1`, top-1 0.6 vs 0.3; vgg13 `f63c` equal to 2.7e-6 everywhere); `cos_clean` and the crests, which see no draw, agree to <= 1e-5.
  Cause: post_shift's 1M-element `torch.randint` sample (`qat_wino.py` l.507) can round one post_shift the other way, and every later layer inherits it; the tracked runs set no torch seed, and **this torch (2.5.1) starts each
  process from a different default seed** (`torch.initial_seed()` differs per run, checked), so they cannot be replayed bit for bit. The 1e-4 rule is dropped. In its place, fixed before the numbers:
  R1 (code) -- the code of 4636092 (which wrote the tracked JSONs) and the current code, same node, torch seed 0 set after the same imports (the `model_registrations` import draws from the RNG), F63 on both bases:
  every field the old code writes must be exactly equal, else 5c8351c changed the measurement; R2 (CPU) -- the current code at seed 0 on `draco1` (Xeon E5-2640 v2) vs `bali2`'s seed-0 JSON on the three nets. Both are
  AVX-only Xeons (tupi1/2 are E5-2620 v4 with AVX2, tupi3-6 i9-14900KF), so R2 does not test against the CPUs the tracked JSONs ran on (tupi had 0 of 64 CPUs free for hours, so a tupi run was left out; the tracked JSONs,
  which ran on AVX2 machines, serve as that comparison below). Draw spread: torch seeds 1 and 2 of all five modes on `bali2` (the seed-0 CPU), the calibration images staying seed 0's; only what holds over every draw counts.
  *Code (305920b):* `f63_collapse --torch-seed` (the global torch seed only) and `torch_seed` + `capture_provenance()` (host, CPU model, git hash, Slurm job) in every JSON. `scripts/phase14/f63_probe_summary.py` -> `results/phase_14_winograd_numerics/f63_probe_summary.csv`:
  per net x mode x draw, top-1, growth per Winograd conv = exp(least-squares slope of ln `absmax_ratio`; reproduces the +0.206 / +0.411 / -0.033 above), max `absmax_ratio`, min `x_kept` / `u_kept`, median `conv_gain`
  drift_out / drift_in over Winograd / exact 1x1 / exact 3x3 convs; the tracked `f63_collapse*.json` enter as draw "unseeded". Test `test_f63_probe_summary_row`.
  *Jobs (QOS `short`, 8 CPUs):* `~/f63_probe/run.sh <net> <seed>` 840357 / 840358 (resnet18 s1 / s2), 840360 / 840361 (vgg13), 840363 / 840364 (googlenet) on `-w bali2`; `~/f63_probe/check.sh <net>` 840359 (resnet18, R1 + R2),
  840362 (vgg13), 840365 (googlenet) on `-w draco1`. Seed-0 JSONs renamed `f63_probe_<net>_<mode>_s0.json`.
  **R1 and R2: exactly equal.** R1 (840359, 11:06-11:33): the code of 4636092 and the current code give the same top-1 and the same value in all 133 fields of each base (new 0.6 %, old 0.3 %; FP32 56.5 / 53.7), so the probes
  did not change the measurement. R2 (840359 / 840362 / 840365): draco1 and bali2 at seed 0 are bit-identical on the three nets (228 / 120 / 264 layer fields, top-1 and every `conv_gain` row). Kept as
  `f63_probe_<net>_default_s0_draco1.json` (resnet18's has both bases). With R2 equal, the two googlenet seed jobs pending on bali2 (840363 / 840364) were moved to draco1 (`scontrol update ReqNodeList`, 11:35).
  **Seeds: the draw almost never matters; the machine does.** 840357 / 58 / 60 / 61 (bali2, 1:10-1:14) and 840363 / 64 (draco1, 0:19) COMPLETED, 30 JSONs (md5 = PCAD's, provenance 305920b clean). Seeds 0 / 1 / 2 are
  bit-identical in 14 of the 15 net x mode cells; the exception is resnet18 `f63c` seed 2 (from `layer1.1.conv2`; top-1 23.5 vs 24.4). Yet the tracked JSONs differ from seed 0 in 5 of 6 pairs, and even `w_crest`, a function
  of the same weights only, differs by 8e-8 to 1.1e-7 in 3-5 layers of every pair, while bali2 and draco1 are bit-identical: the AVX2 machines the tracked runs used (laptop, tupi2, tupi6) and the AVX-only Xeons compute
  different floats (presumably torch's AVX2 kernels vs its default ones; the thread count may count too), and a ~1e-7 difference sometimes flips a rounding that every later layer inherits. So the machine, not the draw,
  explains most of the 2026-11-23 mismatch. Its size: resnet18 `default` max `absmax_ratio` 51.2 (laptop) vs 60.6, top-1 0.3 vs 0.6; vgg13 top-1 0.1 vs 0.8; resnet18's old base max 3.74 vs 5.13; `f63c` top-1 23.5-24.4 /
  16.4 / 38.1-38.4. No conclusion below changes over the 3-5 draws of a cell (seeds, draco1, tracked).
  *Table* (`results/phase_14_winograd_numerics/f63_probe_summary.csv`, `python -m scripts.phase14.f63_probe_summary`; min-max over the draws; top-1 at epoch 0 on 1000 test images; growth per Winograd conv; gain = median
  drift_out / drift_in of `conv_gain`):

  | net | mode | top-1 % | growth | max obs. ratio | min `u_kept` | gain F63 | gain exact 1x1 | gain exact 3x3 |
  |---|---|---|---|---|---|---|---|---|
  | resnet18 | default | 0.3-0.6 | 1.23-1.24 | 51-61 | < 1e-4 | 1.32 | 1.11 | - |
  | resnet18 | every2 | 2.0 | 0.89 | 1.84 | 0.33 | 1.17 | 0.96 | 0.67 |
  | resnet18 | clean | 0.6 | 0.99 | 1.17 | 0.32 | 0.92 | 0.99 | - |
  | resnet18 | p999 | 7.8 | 0.96 | 1.00 | 0.51 | 0.85 | 0.98 | - |
  | resnet18 | f63c | 23.5-24.4 | 0.94 | 3.04 | 0.68-0.69 | 0.80 | 0.96 | - |
  | vgg13 | default | 0.1-0.8 | 1.51 | 34.1-34.2 | 6e-4 | 1.27 | - | - |
  | vgg13 | every2 | 0.8 | 1.15 | 1.79 | 0.36 | 1.34 | - | 0.59 |
  | vgg13 | clean | 0.9 | 1.00 | 1.06 | 0.37 | 0.93 | - | - |
  | vgg13 | p999 | 6.1 | 0.97 | 1.00 | 0.52 | 0.82 | - | - |
  | vgg13 | f63c | 16.4 | 0.94 | 1.95 | 0.66 | 0.80 | - | - |
  | googlenet | default | 0.2-0.3 | 0.95-0.97 | 4.32 | 0.29-0.32 | 1.10 | 0.85 | - |
  | googlenet | every2 | 5.3 | 0.96 | 1.27 | 0.36 | 1.04 | 0.82 | 0.92 |
  | googlenet | clean | 1.1 | 1.00 | 1.11 | 0.34 | 1.09 | 0.84 | - |
  | googlenet | p999 | 22.3 | 0.98 | 1.00 | 0.53 | 1.01 | 0.82 | - |
  | googlenet | f63c | 38.1-38.4 | 0.97 | 2.11 | 0.68-0.70 | 0.96 | 0.82 | - |

  *Decision rules (fixed 2026-11-23):* **A no** -- `p999` leaves resnet18 / vgg13 at 7.8 / 6.1 % (googlenet 22.3): the 99.9th-percentile observer stops the growth (ratio <= 1, `u_kept` >= 0.51) but the max estimator is not
  the weak link by itself. **B: the loop is necessary for the compounding** -- by top-1 `clean` collapses (0.6 / 0.9 / 1.1 %), by the trajectories (the criterion fixed above) it survives (observers <= 1.17x, `u_kept` >= 0.32,
  growth ~1.0); the per-layer error with the worst-case `su` alone still leaves epoch 0 at chance, as it does googlenet `default`, which QAT takes to 48.5 %. **C: a property of the chain, not of the layer** -- with F63 on every
  second conv the growth falls from x1.23 / x1.51 to x0.89 / x1.15 and the max observer ratio from 61 / 34 to 1.8 / 1.8; the convs swapped back to exact 3x3 shrink the relative drift (median x0.67 / x0.59) while each F63
  conv grows it (x1.17-1.34). **D supported** -- googlenet's exact 1x1 convs shrink the drift (x0.82-0.85 in every mode); after its three stem convs, each of its other 19 F63 convs comes right after an exact 1x1 (module
  order), against 15 of 19 right after another F63 conv in resnet18 and 9 of 9 in vgg13, and its F63 convs grow the drift less (x1.10 vs 1.32 / 1.27).
  *Answer:* the error compounds where F63 convs follow one another with nothing between them that shrinks it, and the observers, calibrated on the error-carrying activations, follow it up; googlenet puts an exact 1x1
  before each one. Any of `every2`, `clean`, `p999` stops the compounding; none takes resnet18 / vgg13 out of chance at epoch 0, and whether QAT would (as for googlenet) needs an arm -- the article's future work, not planned.
  *Article* (`sec:numerica`): "Não medimos por que..." replaced by a paragraph with this answer; the numbers it cites that change with the machine now hold for every draw ("14 a mais de 50 vezes", "1\% a 57\%",
  "no máximo 5 vezes na anterior", `f63c` "cerca de 24\%, 16\% e 38\% ... contra menos de 1\%"); the future-work item now asks whether exact convs between the Winograd ones avoid the collapse after QAT. Compiled, no error.
