# Phase 13 — hybrids (`phase_13_hybrids`)

An extra study, kept apart from Phase 11 (user, 2026-10-22: "deixe a fase 11 como está. Esses modelos híbridos são coisas
a mais"): it reuses Phase 11's recipe and reads 6 of its runs as references, but no Phase 11 yaml, figure, table or log
reads or lists these runs. Analysis: `scripts/phase13/hybrids.py` -> `results/figures_generated/phase_13_hybrids/`,
`results/phase_13_hybrids/`.

## Design and why (2026-10-22): do the techniques that worked add up? 18 runs

**Why.** At 93 of 152 Phase 11 runs the ingredients that lift the 64px GAP + BN AlexNet are measured one at a time: two convs
per stage (+5.3 pp 3x3, +8.0 pp 2x2), AlexNet's 11x11 conv1 on a small body (+1.4 / +3.7 pp), and the two families that do
well both have a stride-1 conv1 (Fire 54.6%, SmallKernelWithBN 57.2%). Two were never isolated: the residual skip
(AlexNetResidual = the stacked net + skips, but with a 3-layer FC head and no Dropout, which no stacked cell has) and stride
2 -> 1 (no matched pair). The user asked for "hybrid" nets mixing what worked; a single hand-picked net would give one number
and explain nothing, so it is a factorial on one base, every contrast a matched pair, every cell reported -- no net picked on
the test set (Cawley & Talbot, JMLR 2010). User's choices: both body kernels, stride 1 as a 4th factor, end of PCAD's queue.

| factor | levels |
|---|---|
| body kernel | 3x3, 2x2 (2x2 behind C2sp, a stride-1 conv1 too, as `VGGAdapted`'s stem) |
| conv1 | the body's kernel / AlexNet's 11x11 |
| body | 5 convs / two per stage (`k*stacked`) / two per stage as a residual block (`k*residual`) |
| conv1 stride | 2 (last map 8x8) / 1 (16x16) |

2 x 2 x 3 x 2 = 24 cells; 6 were trained in Phase 11 at stride 2 (`k3x3`, `k2x2`, `k11-3`, `k11-2`, `k3x3stacked`,
`k2x2stacked`, GAP + BN, seed 42) and are read from there, never retrained; 18 are new
(`configs/experiments/phase_13_hybrids.yaml`, seed 42). Residual exists only on the two-conv body, so it is not a 2^4.

**Decisions and their references.**
- *Residual block.* He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition" (CVPR 2016): two convs + BN, the skip added
  before the last ReLU, identity where the shape holds and a 1x1 conv + BN where channels or stride change (option B, Sec. 3.3 /
  Table 3) -- 4 projections (stages 1-4), identity at stage 5, ~0.19M parameters and ~15M MACs over the stacked net. A one-conv
  residual unit "y = W1 x + x" showed no advantage (Sec. 3.2), hence no "5 convs + skip" cell. QAT: the add is
  `FloatFunctional.add_relu`, torchvision's `QuantizableBasicBlock` (Jacob et al. 2018). The block is `models.compensation._ResBlock`,
  the one `AlexNetResidual` uses (kernel now a parameter; tests/test_registry.py checks `k3x3residual`'s body is `AlexNetResidual`'s
  layer for layer).
- *Stacking.* Two small convs see what one larger does, with an extra non-linearity (Simonyan & Zisserman, ICLR 2015, Sec. 2.3).
- *Large conv1 on a small body.* ResNet's own pattern (7x7 stem, 3x3 body: He et al. 2016, Table 1); the 11x11 is AlexNet's
  (Krizhevsky et al. 2012), 'same' padding 5, as in `phase_11_large_stem`.
- *Stride 1.* Small-image nets keep the resolution early: He et al. 2016 Sec. 4.2's CIFAR ResNet has a stride-1 3x3 stem. Costs ~4x
  the MACs (k3x3: 167M -> 669M; stacked 451M -> 1803M), which is the factor itself.
- *Recipe.* `_protocols/no_patience.yaml`, Phase 11's, unchanged: tests/test_config.py holds `phase_13_*` to it exactly.

**Code.** `_ResBlock(kernel_size=, first_kernel=)` (defaults = before: `AlexNetResidual` unchanged); `AlexNetAdapted` puts C2sp in front
of a stride-1 2x2 conv1 (`i > 0 or stem_stride == 1`; no registered cell had stride 1); `AlexNetStacked(stem_kernel=, stem_stride=,
residual=)`, whose defaults build the same flat Sequential (the trained stacked cells load unchanged: state_dict, fuse map and module
list of 9 existing cells compared before/after). Cells registered in `ml/model_registrations.py` (fuse map by `find_fuse_groups`); the
`kernels` token names the body, as `k3x3stacked` / `k3x3deepstem` do: `alexnet_k11-2residual_stride1_2pool2x2_map16_gap_bn`.

**Analysis** (`scripts/phase13/hybrids.py`, tests/test_phase13_hybrids.py on a fake tree). `01_hybrids.png`: one panel per conv1 stride,
the three bodies per conv1, 3x3 and 2x2 side by side, FP32 / INT8, MACs under each body, VGG16 3x3 GAP + BN (Phase 11) as the
reference line, the AlexNet GAP + BN seed band in the title. `contrasts.csv`: every matched pair of the 24 cells, one factor changed
(5 convs -> two per stage 8, two per stage -> + skip 8, conv1 -> 11x11 12, kernel 3x3 -> 2x2 12, stride 2 -> 1 12), B - A at seed 42 --
whether the gains add up is read off the same factor's delta across the other factors' levels. `all_runs.csv`: every run of the 24
cells. Phase 11's `analyze_geometry.load()` globs `phase_11_*` and `design_cells()` the `phase_11_*.yaml`, so none of its figures,
tables or matched pairs sees these runs.

**Cost.** Estimated on a 4090 (stride 1 not measured): stride-2 hybrids ~6-9 h, stride-1 plain ~6-10 h, stride-1 deep ~12-20 h
(VGG16-like MACs) -- ~150-250 GPU-h. Stride 2 on `tupi_beagle` (as every AlexNet cell), stride 1 on `tupi_4090` (VGG16's cost, kept off
beagle as VGG16 is). At the end of PCAD's queue (~48 lines at ~2.4 a day): first hybrid in ~3 weeks, last in ~4.

**Checked 2026-10-22.** tests/test_registry.py, test_config.py, test_phase11_figures.py (unchanged), test_phase13_hybrids.py,
test_train_pipeline.py and test_quantization.py's slice for the 18 cells (QAT -> INT8 against the literature's definition) pass; a
1-epoch `--smoke` of the two most complex cells (`k11-2residual` and `k3x3stacked`, both stride 1) ran FP32 -> QAT -> INT8 and exited 0
(the INT8 kernel and QAT-agreement guards included), nothing written to `outputs/`. First pushed as `phase_11_hybrids` (a0a7719, with
figure 27 and contrasts inside Phase 11's analysis); separated the same day at the user's request -- Phase 11's analysis, tests and log
restored exactly to 510734c.

**Next, not in this block.** Seeds 43 / 44 once seed 42 lands, on the contrasts inside 2x the noise band (likely stacked -> residual, with
the stacked partners that have one seed). Stage B -- Bottleneck / Fire blocks on the best body (compression at equal accuracy) -- waits for
these, the `alexnet_final_*` and `alexnet_fire_bypass` runs; it needs a kernel parameter in `_AlexBottleneck` / `_FireModule`.

**Status 2026-10-27.** None of the 18 has run. They are still the last 18 lines of PCAD's `~/queue_phase11_full.txt` (35 lines: 7
Phase 11 seeds, 10 Phase 12 fine-tunes, then these), behind 7 Phase 11 jobs in Slurm; at the ~2-3 jobs a day PCAD has finished
lately, the first hybrid starts in ~1-1.5 weeks. Everything they are read against is in: the 6 Phase 11 reference cells, and for
Stage B the `alexnet_final_*` and `alexnet_fire_bypass` runs (`phase_11_families`, seed 42: FinalBottleneckResidual 55.83, FireBypass
55.19, FinalFireResidual 54.96, FinalBottleneckFire 54.40, FinalDepthwiseFire 45.76 FP32).

**Status 2026-10-31: the first hybrid.** `alexnet_k11-3stacked_stride2_2pool2x2_map8_gap_bn` (11x11 conv1 + the two-conv 3x3 body,
stride 2, seed 42; job 834529 on beagle, 15.6 h) finished 2026-10-31 04:29 and is on the laptop: 55.52 FP32 / 55.45 INT8 (test
top-1), 5.20 M params, 473 M MACs; best FP32 epoch 491 of 500, agreement_qat_int8 0.984, quant_protocol 2026-10-07, git_dirty false.
Its two matched pairs, B - A in FP32 / INT8 (negative = loses):
- conv1 3x3 -> 11x11 on the stacked body (`k3x3stacked`, 55.87 / 56.15, 5.18 M, 451 M MACs): -0.35 / -0.70 pp.
- body 5 convs -> stacked under the 11x11 conv1 (`k11-3`, 54.55 / 54.24 at seed 42; 54.14 FP32 over 3 seeds): +0.97 / +1.21 pp,
  against +2.37 / +2.50 for the same step under a 3x3 conv1 (`k3x3` 53.50 / 53.65 -> `k3x3stacked`).

One seed, and the conv1 contrast sits inside the AlexNet GAP + BN floor (±0.7-0.8 pp FP32): so far no sign that the 11x11 conv1 adds
anything on the stacked body. `scripts.phase13.hybrids` not run yet (1 of 18). Queue: `k11-2stacked` running on beagle since ~04:30,
7 pending in Slurm (the 3 stride-1 cells on `tupi_4090` held by Priority), 9 lines in the feeder file.

**2026-10-31, nyx for the weekend.** The user lent nyx for 10-31..11-02 only. The 5 last feeder lines -- `k11-2stacked`,
`k3x3residual`, `k2x2residual`, `k11-3residual`, `k11-2residual`, all stride 1 -- moved from PCAD's `~/queue_phase11_full.txt`
(backup `.bak_20261031`, list in `~/moved_to_nyx_20261031.txt`) to `~/queue_nyx.txt`, each guarded to start before Mon 2026-11-02
10:00; the first started 08:02 at 61 s per FP32 epoch (GPU-bound, 99%), ~10 h per cell with QAT, so all 5 should end Monday
afternoon. A line skipped by the guard returns to the end of PCAD's feeder. nyx's disk had 2.2 GB free (~250 MB per run).

**Status 2026-10-31 evening: 5 of 18, stride 2 nearly complete.** PCAD's tupi freed up: `k11-2stacked` (beagle, 15.1 h),
`k3x3residual`, `k2x2residual`, `k11-3residual` (tupi3 / tupi6, ~4 h each) finished the same day and are on the laptop (tracked
files, md5 identical; aggregate rows appended, PCAD's rewrite of the `k11-3stacked` row differs in float digits only, committed
text kept). All 5: quant_protocol 2026-10-07, git_dirty false, every budget used, int8_kernel_max_err_steps 1, agreement_qat_int8
0.971-0.984, best FP32 epoch 450-498 of 500 (none early best). `scripts.phase13.hybrids` rerun: figure 01 (stride-1 panels still
"pendente"), `contrasts.csv`, `all_runs.csv`; `results/runs_index.csv` 485 -> 489.

Test top-1 FP32 (INT8), seed 42, stride 2, last map 8x8; MACs per image:

| body | 3x3, conv1 3x3 | 3x3, conv1 11x11 | 2x2, conv1 2x2 | 2x2, conv1 11x11 |
|---|---|---|---|---|
| 5 convs (Phase 11) | 53.50 (53.65), 167 M | 54.55 (54.24), 189 M | 52.39 (52.36), 75 M | 55.64 (55.84), 98 M |
| two per stage | 55.87 (56.15), 451 M (Phase 11) | 55.52 (55.45), 473 M | 57.25 (57.55), 201 M (Phase 11) | 57.12 (57.06), 224 M |
| two per stage + skip | 56.24 (56.05), 465 M | 56.12 (55.83), 487 M | 57.67 (57.86), 215 M | running |

Matched pairs, B - A in FP32 (`contrasts.csv`, median over the pairs in the script's table):
- *5 convs -> two per stage*: +2.37 (3x3), +4.86 (2x2), +0.97 (11x11 + 3x3), +1.48 (11x11 + 2x2). The largest step, larger under
  a small conv1.
- *two per stage -> + residual skip*: +0.37, +0.60, +0.42 (INT8 -0.10, +0.38, +0.31). Same sign every time, each inside the seed
  floor (±0.7 pp FP32): a candidate for seeds 43 / 44.
- *conv1 -> 11x11*: +1.05 (3x3) and +3.25 (2x2) on the 5-conv body, -0.35 / -0.13 / -0.12 on the two-conv bodies. The two do not
  add up: the second conv per stage buys what the 11x11 conv1 bought.
- *body 3x3 -> 2x2*: -1.11 on the 5-conv body, +1.38 / +1.43 / +1.60 on every deeper body (and +1.09 under the 11x11 conv1).

Best so far: `k2x2residual`, 57.67 FP32 / 57.86 INT8 at 2.52 M params and 215 M MACs -- 1.8 pp above the two-conv 3x3 body at half
its MACs, every conv a stride-1 2x2 but conv1 (stride 2). One seed each.

**nyx.** At ~15:30 a lab colleague moved nyx out of the user's tailscale into his own account to run simulations over the weekend
(user agreed; runs not stopped). `ssh nyx` times out until the user moves it back, so the 5 stride-1 cells there cannot be checked
from here; the start guard runs locally, so none starts after Mon 2026-11-02 10:00 either way.

## Status 2026-11-03: 17 of 18, the stride-1 cells are in

**What landed.** The 12 remaining cells except one are on the laptop (`399e223`, `7e2b1b6`): the 7 stride-1 cells that ran on PCAD
(`k{3x3,2x2,11-3,11-2}` plain and `k{3x3,2x2,11-3}stacked`; tupi3 / tupi5 / tupi6, 4.1-4.9 h each), `k11-2residual` stride 2 (beagle), and 4 of the 5 on nyx (`k11-2stacked`, `k3x3residual`,
`k2x2residual`, `k11-3residual`, all stride 1; RTX 4060 Ti). All 17: seed 42, `quant_protocol` 2026-10-07, `git_dirty` false, best FP32 epoch
430-499 of 500 (none early best), INT8 - FP32 median -0.10 pp (-0.44 to +0.77). Hosts differ (11 x 4090, 4 x 4060 Ti, 2 x 1080 Ti):
bit-for-bit retraining of a GAP cell was measured on one GPU model only (CLAUDE.md), so a cell's last digit is not comparable across
hosts -- the contrasts below rest on tenths of a pp and more.

**The 18th: `alexnet_k11-2residual_stride1_2pool2x2_map16_gap_bn`.** It was the 5th nyx cell: FP32 finished (final val top-1 59.73),
QAT stopped at epoch 13 of 50 (log 2026-11-02 19:40) when nyx's loan ended; its checkpoints are on PCAD. Resubmitted as job 835178
(`tupi`, 2026-11-03 11:03, pending on Priority); `scripts/train.py` should resume the QAT from its checkpoint (not checked), ~2.5 h of QAT
left plus the evaluations. This run is the one cell trained on two GPU models (QAT epochs 1-13 on nyx's 4060 Ti, the rest on a 4090;
`provenance_history` in its `resolved_config.json` will list both jobs) -- keep that in mind when reading its last digit. Four matched pairs wait for it (skip at stride 1, conv1 11x11 on the
skip body at stride 1, 3x3 -> 2x2 under the 11x11 skip body at stride 1, stride on the 11x11 skip body).

Test top-1 FP32 (INT8), seed 42, **conv1 stride 1 (last map 16x16)**; MACs per image:

| body | 3x3, conv1 3x3 | 3x3, conv1 11x11 | 2x2, conv1 2x2 | 2x2, conv1 11x11 |
|---|---|---|---|---|
| 5 convs | 57.66 (57.56), 669 M | 57.29 (57.12), 757 M | 56.38 (56.22), 298 M | 57.07 (56.63), 390 M |
| two per stage | 58.53 (58.27), 1803 M | 59.16 (59.04), 1891 M | 60.35 (60.79), 803 M | 59.86 (60.63), 895 M |
| two per stage + skip | 59.16 (59.53), 1862 M | 59.11 (59.14), 1950 M | 60.75 (60.46), 862 M | pending (835178) |

`k11-2residual` stride 2, the cell the 2026-10-31 table marked running: 56.59 (56.59), 238 M MACs, 2.54 M params.

Matched pairs, B - A, 48 of the grid's 52 (`contrasts.csv`); median FP32 / INT8, pairs with B > A, range of the FP32 delta:

| change | pairs | median FP32 / INT8 | B > A | range |
|---|---|---|---|---|
| conv1 stride 2 -> 1 (4x the MACs) | 11 | +2.99 / +3.31 | 11 of 11 | +1.43 to +4.16 |
| 5 convs -> two per stage | 8 | +2.12 / +2.21 | 8 of 8 | +0.87 to +4.86 |
| body 3x3 -> 2x2 | 11 | +1.09 / +1.40 | 8 of 11 | -1.28 to +1.82 |
| two per stage -> + skip | 7 | +0.40 / +0.10 | 5 of 7 | -0.53 to +0.63 |
| conv1 -> 11x11 | 11 | -0.12 / -0.22 | 4 of 11 | -1.08 to +3.25 |

- *Stride 1 is the largest factor and never fails* (all 11 pairs, 3.0 pp at 4x the MACs). Two per stage comes second and never fails either,
  but at stride 1 its step is smaller under a 3x3 body (+0.87) than at stride 2 (+2.37).
- *Skip: the 2026-10-31 reading ("same sign every time") did not survive.* +0.37, +0.60, +0.42 at the first five became -0.53 (`k11-2`, stride
  2) and -0.05 (`k11-3`, stride 1) with the rest; INT8 flips sign too (4 of 7). Every pair is inside the ±0.7 pp floor: no measurable
  gain from the skip on this base, at 3% to 7% more MACs.
- *11x11 conv1: only helps the 5-conv body.* +1.05 / +3.25 (3x3 / 2x2, stride 2) and +0.69 (2x2, stride 1); on the 7 two-conv pairs the
  median is -0.13 (-1.08 to +0.63), and `k2x2residual` -> `k11-2residual` at stride 2 (-1.08 / -1.27) is the only pair there outside the floor.
- *3x3 -> 2x2 depends on the body.* It loses on the 5-conv body (-1.11, -1.28) and wins on **every one of the 7 two-conv pairs** (+0.47 to
  +1.82, median +1.43) with 51-55% fewer MACs.

**Do the gains add up? Partly.** From `k3x3` (stride 2, 5 convs) to `k3x3stacked` stride 1, the two steps measured separately (+2.37, +4.16) would
give 60.03; measured 58.53 (-1.50). For 2x2: 52.39 + 4.86 + 3.99 = 61.24 predicted, 60.35 measured (-0.89). Each of the two big steps keeps 37-82% of its gain
once the other is in (3x3: two per stage +2.37 -> +0.87 at stride 1, stride +4.16 -> +2.66 on the stacked body; 2x2: +4.86 -> +3.97, +3.99 -> +3.10).
The 2x2 body overlaps less. The skip and the 11x11 conv1 add nothing on top.

**Best nets (one seed, so nothing is picked on it):** FP32 `k2x2residual` stride 1, 60.75 (INT8 60.46), 2.52 M params, 862 M MACs; INT8
`k2x2stacked` stride 1, 60.79 (FP32 60.35), 2.33 M, 803 M. VGG16 3x3 GAP + BN (Phase 11): 59.35 / 59.47 at seed 42, 59.09-59.37 over 3 seeds, 14.8 M
params, 1255 M MACs. So the two best AlexNets sit 1.0-1.4 pp above it with 1/6 of the params and 64-69% of the MACs -- against an AlexNet floor
of ±0.7-0.8 pp, so a sign, not a size, until seeds 43 / 44 land. Every layer of the 2x2 two-conv nets but conv1 is a stride-1 2x2.

**Seeds 43 / 44 -- submitted 2026-11-03 (PCAD jobs 835242 / 835243 seed 43, 835244 / 835245 seed 44; `tupi_4090`, pending on Resources).** `configs/experiments/phase_13_hybrids_seed43.yaml` / `_seed44.yaml`: `k2x2stacked`
and `k2x2residual`, both stride 1, 4 runs, ~4.5-5 h each on a 4090 (the stacked stride-1 cells took 4.5-4.9 h), ~19 GPU-h. Why these two:
they hold the open contrast (skip, +0.40 FP32 / -0.33 INT8 at stride 1, sign flipping elsewhere) and they are the nets to compare with VGG16, so
their own noise is what the comparison needs. Not replicated: the 11x11 conv1 (≈0 on every two-conv body, nothing to confirm) and the 3x3
partners (`k3x3stacked` / `k3x3residual` stride 1, +4 runs, only if "2x2 beats 3x3 at stride 1" (+1.82 / +1.59) is to be quoted). Committed (ac3c926), pushed
and pulled on PCAD (`--ff-only`, no untracked file in the way) before `scripts.cluster submit-sweep`. `scripts.phase13.hybrids` (2026-11-03): figure 01 and
`contrasts.csv` stay seed 42; a cell's other seeds are small FP32 dots beside it, `contrasts_by_seed.csv` gives every matched pair at each
seed where both cells ran (the skip contrast at seeds 42 / 43 / 44 once the jobs land), and the run prints mean / SD for cells with replicates.
Checked on the fake tree (tests/test_phase13_hybrids.py, seed 43 added); not rerun on the real runs.

## Status 2026-11-04: 18 of 18

**The 18th landed.** `alexnet_k11-2residual_stride1_2pool2x2_map16_gap_bn` (PCAD job 835178, tupi1, RTX 4090, 04:46-06:23, 1 h 37): `scripts/train.py`
resumed the QAT at epoch 14 of 50 from nyx's checkpoint (log: epoch 13 on 2026-11-02 19:40 at 234 s/epoch, epoch 14 on 2026-11-04 04:51 at 87 s/epoch);
`provenance_history` lists both jobs (Nyx 4060 Ti, then tupi1 4090), `git_dirty` false. Test top-1 **60.48 FP32 / 60.76 QAT / 60.85 INT8**, 954 M MACs,
agreement QAT/INT8 0.977. INT8 - FP32 = +0.37. The cell trained on two GPU models, so its last digit is soft.

| body | 3x3, conv1 3x3 | 3x3, conv1 11x11 | 2x2, conv1 2x2 | 2x2, conv1 11x11 |
|---|---|---|---|---|
| two per stage + skip, stride 1 | 59.16 (59.53) | 59.11 (59.14) | 60.75 (60.46) | 60.48 (60.85), 954 M |

Matched pairs with the 18th (52 of the grid's 52; `contrasts.csv`), B - A, median FP32 / INT8:

| change | pairs | median FP32 / INT8 | B > A | range FP32 |
|---|---|---|---|---|
| conv1 stride 2 -> 1 | 12 | +3.03 / +3.40 | 12 of 12 | +1.43 to +4.16 |
| 5 convs -> two per stage | 8 | +2.12 / +2.21 | 8 of 8 | +0.87 to +4.86 |
| body 3x3 -> 2x2 | 12 | +1.23 / +1.50 | 9 of 12 | -1.28 to +1.82 |
| two per stage -> + skip | 8 | +0.41 / +0.16 | 6 of 8 | -0.53 to +0.63 |
| conv1 -> 11x11 | 12 | -0.12 / -0.19 | 4 of 12 | -1.08 to +3.25 |

New pairs: skip at stride 1 on the 11x11 / 2x2 body +0.62 / +0.22; 11x11 conv1 on the 2x2 skip body at stride 1 -0.27 / +0.39 (stride 2: -1.08 / -1.27);
3x3 -> 2x2 under the 11x11 skip body at stride 1 +1.37 / +1.71; stride 2 -> 1 on that body +3.89 / +4.26. None changes a reading of 2026-11-03: stride 1 never
fails (12 of 12), two per stage never fails, 2x2 beats 3x3 on **all 8 two-conv pairs** (+0.47 to +1.82), the skip stays inside the ±0.7 pp floor, the 11x11 conv1
is still ~0 on the two-conv bodies. The only new skip pair, +0.62, is the largest of the 8 and still at the floor. Seeds 43 / 44 of the two best nets:
835242 / 835243 running (tupi6, tupi1), 835244 / 835245 pending.

