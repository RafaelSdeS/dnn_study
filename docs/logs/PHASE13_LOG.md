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


## 2026-11-05: the 3x3 side of the two-conv contrast gets seeds too

"2x2 beats 3x3 on all 8 two-conv pairs" is a line of the article's abstract, and the replicates above only cover the 2x2
nets. `phase_13_hybrids_seed43/_seed44.yaml` gain `alexnet_k3x3stacked_stride1_...` and `alexnet_k3x3residual_stride1_...`
(4 runs, ~20 GPU-h at seed 42's times), so the two stride-1 pairs compare at 3 seeds. The stride-2 pairs would mean
seed files for Phase 11's `phase_11_stacked` cells, a Phase 11 design change -- left out. Seed state today: 835242
(k2x2stacked s43) done; 835243 / 835244 hit the 24 h wall and were resubmitted (835347, 835352); 835245 running.


## 2026-11-06: where the seeds are, and what the crashes say about tupi4

**Seeds** (PCAD; nothing synced to the laptop yet). Stride 1, GAP + BN, test top-1 FP32 / INT8; seed 42 for reference: `k2x2stacked` 60.35 / 60.79, `k2x2residual` 60.75 / 60.46.

| run | job | state |
|---|---|---|
| `k2x2stacked` s43 | 835242 | done, 60.65 / 60.56 |
| `k2x2residual` s43 | 835347 | done, 60.38 / 60.48 (fresh run; 835243 froze, see below) |
| `k2x2residual` s44 | 835245 | done, 59.92 / 60.09 |
| `k2x2stacked` s44 | 835352 | **running on beagle** since 2026-11-06 09:20:27 (wall 11-07 09:20) |
| `k3x3stacked` s43 / s44 | 835433 / 835436 | running (tupi1, in QAT) / pending |
| `k3x3residual` s43 / s44 | 835435 / 835437 | pending |

The three finished 2x2 runs had `git_dirty` false, the current quant protocol, best FP32 epoch 456-492 of 500 (no early best) and `agreement_qat_int8` 0.976-0.984. The
residual net's FP32 spans 0.83 pp across its three seeds. `k2x2stacked` seed 44 took three tries: 835244 froze mid-epoch with the PCAD freeze of 2026-11-04 ~10:14 (as
835243 did on tupi1) and was killed at the wall; a fresh 835348 died on tupi3 at FP32 epoch 41/500 with "Illegal instruction" (exit 4:0); 835352 excluded tupi1-4 and sat pending
for 21 h because only tupi5/6 were left, until it was also allowed on beagle (idle), with MinMemoryNode=15360. A beagle run is a different GPU (GTX 1080 Ti): the accuracy
protocol is the same (exact INT8, TF32 off), the latency is recorded with its host; if it reaches the wall it requeues and resumes, which is not bit-reproducible. The
interrupted attempts are in `outputs/pcad/archive_interrupted_seeds/` on PCAD, with a `SUPERSEDED.md`.

**Signals by node.** `exclude: tupi4` dates from 2026-10-24 (3 of its 10 jobs crashed). `sacct -a`, jobs of >= 30 min since 2026-10-01, every user, FAILED by exit code
(11 = SIGSEGV, 6 = SIGABRT, 4 = SIGILL -- read as the signal; verified only for 835348, "Illegal instruction"):

| node | CPU | long jobs | killed by a signal |
|---|---|---|---|
| tupi1 | Xeon | 163 | 0 |
| tupi2 | Xeon | 188 | 0 |
| tupi3 | i9 | 117 | 3 (SIGSEGV) |
| **tupi4** | i9 | 126 | **10** (9 SIGSEGV, 1 SIGABRT) |
| tupi5 | i9 | 215 | 5 (1 SIGSEGV, 4 SIGILL) |
| tupi6 | i9 | 220 | 1 (SIGILL) |

tupi4: 10 of 126 against 9 of 903 elsewhere (Fisher one-sided p = 2e-5; SIGSEGV alone 9/126 vs 4/903, two-sided p = 2e-6). The i9 nodes together 19 of 678 against 0 of 351 on the
Xeons (one-sided p = 3e-4). Job 835348 (17 min, so under the 30 min cut) is a fourth tupi3 case. Our three tupi4 crashes: 832921 (`TypeError: list_iterator doesn't define
__round__` inside `PIL.Image.point`, after 1 h 45, impossible from Pillow's code), 832753 (SIGSEGV right after "QAT training complete", 5 h 49) and 832951 (SIGSEGV 38 s after a
resume, after the FP32 summary). After 2026-10-24 the node only ran 18 short Winograd-FPGA jobs (clean), so it has no long-job evidence since the exclude.

*Caveats.* The counts include other users' code and only the exit code, not their error text; tupi1/2 carry a different job mix; the signals are small numbers. *Cause:* not
verified. The pattern (random SIGSEGV / SIGILL / a Python TypeError that cannot happen, under load, on the i9 desktop parts, worst on one node) fits Raptor Lake Vmin-shift
degradation -- Intel's 0x12B microcode stops further damage but does not repair a chip that already degraded (Intel scopes it to Raptor Lake-S desktop parts such as the
i9-14900KF; the sources do not mention segfaults, only crashes and instability) -- but bad RAM would look the same. Only someone with the node can tell
(`dmesg` is restricted for us): dmesg/MCE, BIOS and microcode version, a memtest.

**Decision (2026-11-06): keep.** `exclude: tupi4` stays in `tupi_4090.yaml` / `tupi_beagle.yaml`. The 1-epoch smoke on tupi4 that the close-out plan asked for is dropped
as a criterion: it cannot catch a crash that came at epoch 242 or after 5 h. tupi3 is not excluded: its 2.6% and tupi5's 2.3% cannot be told apart from each other with these
numbers. Rule for the i9 nodes: a long run that dies by a signal is rerun from scratch, not resumed. Open: ask the GPPD admin about tupi4 (and tupi3 / tupi5). The re-evaluation
pilot (Bloco C) is postponed until the Phase 13 seeds and Phase 14 are done, and then fits QOS `short` (2 h, priority 1000).

Sources for the Intel part: Tom's Hardware, "Intel finds root cause of CPU crashing and instability errors, prepares new and final microcode update" (2024); WePC,
"Intel confirms 0x12B as the final patch for 13th and 14th gen instability issues" (2024).

## 2026-11-06 (later): five seed runs synced

Synced from PCAD (only runs with a complete summary): seed 43 `k2x2residual` / `k2x2stacked` / `k3x3stacked`, seed 44 `k2x2residual` / `k3x3residual`, all stride 1.
Still running: `k2x2stacked` seed 44 (835352, beagle), and the 3x3 counterparts `k3x3residual` seed 43 / `k3x3stacked` seed 44 (835435 / 835436, tupi1 / tupi2).
`scripts.phase13.hybrids` rerun; test top-1 per cell, mean ± std over the seeds in (n):

| cell (stride 1) | FP32 | INT8 | seeds |
|---|---|---|---|
| `k2x2residual` | 60.35 ± 0.42 | 60.34 ± 0.22 | 42, 43, 44 (3) |
| `k2x2stacked` | 60.50 ± 0.21 | 60.68 ± 0.16 | 42, 43 (2) |
| `k3x3residual` | 59.63 ± 0.66 | 59.76 ± 0.32 | 42, 44 (2) |
| `k3x3stacked` | 58.69 ± 0.23 | 58.61 ± 0.48 | 42, 43 (2) |
| VGG16 3x3 GAP + BN (Phase 11 reference) | 59.27 ± 0.16 | 59.45 ± 0.16 | 3 |

Medians over every matched pair of the 24-cell grid (FP32 / INT8, pp): conv1 stride 2 → 1 **+3.03 / +3.40** (12 pairs); 5 convs → two per stage +2.12 / +2.21 (8); 3×3 → 2×2 body
+1.23 / +1.50 (12); residual skip +0.41 / +0.16 (8); conv1 → 11×11 −0.12 / −0.19 (12). Nothing moved the ranking of the factors. The best two-conv 2×2 nets (60.35-60.50 FP32)
stay ahead of VGG16's 59.27 at about 2/3 of its MACs and 1/6 of its parameters. The seed spread (±0.2-0.7) is as wide as the skip's effect, so "skip = noise" stands; the 2×2 vs
3×3 gap at stride 1 (60.35-60.50 vs 58.69-59.63) is still one seed short on the 3x3 side — wait for the three runs above before quoting it with its floor.

## 2026-11-07: all eight seed runs are in

The last three synced from PCAD (summary complete): seed 43 `k3x3residual`, seed 44 `k2x2stacked` (835352, beagle, 21 h 27 min, finished 06:47) and seed 44 `k3x3stacked`. Every cell of the
{2×2, 3×3} × {stacked, residual} stride-1 block now has seeds 42, 43, 44. `scripts.phase13.hybrids` rerun; `runs_index.csv` rebuilt with the 54 archive rows merged back from git (the
sparse checkout hides them); `build_cross_phase_results` unchanged. Nothing committed.

Test top-1, mean ± std over the 3 seeds (n = 3 in every row):

| cell (stride 1) | FP32 | INT8 |
|---|---|---|
| `k2x2stacked` | 60.47 ± 0.16 | 60.63 ± 0.14 |
| `k2x2residual` | 60.35 ± 0.42 | 60.34 ± 0.22 |
| `k3x3residual` | 59.39 ± 0.63 | 59.50 ± 0.49 |
| `k3x3stacked` | 58.70 ± 0.16 | 58.61 ± 0.34 |
| VGG16 3x3 GAP + BN (Phase 11 reference) | 59.27 ± 0.16 | 59.45 ± 0.16 |

Per-seed contrasts (seeds 42 / 43 / 44, FP32 pp; INT8 in brackets):

- **3×3 → 2×2 body, stacked:** +1.82 / +1.80 / +1.68 (+2.52 / +1.61 / +1.92). Same sign and size in all three seeds.
- **3×3 → 2×2 body, residual:** +1.59 / +1.48 / **−0.18** (+0.93 / +1.48 / +0.11). The 3×3 residual net at seed 44 is 60.10 against 59.16 and 58.90 at seeds 42 / 43, the only 3×3 two-conv run
  above 59.2; it alone closes the gap. Mean +0.96 (INT8 +0.84).
- **skip, 3×3 body:** +0.63 / +0.05 / +1.37 (+1.26 / +0.05 / +1.36). Positive in all three seeds but 0.05 to 1.37, mean +0.68: a sign, not a size.
- **skip, 2×2 body:** +0.40 / −0.27 / −0.49 (−0.33 / −0.08 / −0.45), mean −0.12. The sign flips: no effect.

Medians over the 24-cell grid (the seed-42 grid; `contrasts.csv` stays seed 42, the other seeds are in `contrasts_by_seed.csv`) did not move: stride 2 → 1 **+3.03 / +3.40**, 5 convs → two per
stage +2.12 / +2.21, 3×3 → 2×2 +1.23 / +1.50, skip +0.41 / +0.16, conv1 → 11×11 −0.12 / −0.19. The ranking of the factors is the same.

What this settles and what it does not:

- 2×2 beats 3×3 with the stacked body in 3 of 3 seeds (+1.7 to +1.8 FP32). With the residual body it holds in 2 of 3, and the outlier is a single lucky 3×3 run (60.10), so quote it as
  "+1 to +2 pp, not robust per body" rather than as a floor.
- The two-conv 2×2 nets stay ahead of VGG16: their lowest run (59.92) is above VGG16's highest (59.37, range 59.09-59.37), at about 2/3 of its MACs and 1/6 of its parameters. The two-conv 3×3 nets
  do not clearly: `k3x3stacked` is below VGG16 (58.70 vs 59.27) and `k3x3residual` ties it (59.39).
- "skip = noise" stands for the 2×2 body. For the 3×3 body the skip is positive in 3 of 3 seeds, yet its spread (0.05-1.37) is as wide as its mean, so three seeds cannot size it.
- This unblocks the `\pendente` marks of the article that waited for the Phase 13 seeds (not edited yet).

## 2026-11-07: Stage B -- Bottleneck and Fire units on the best bodies (`phase_13_bottleneck_fire`)

User: "Faça o stage B da phase 13, mas coloque-os no final da fila"; on the design questions, all cells, every placement, and a local smoke
before submitting (once the laptop is free). Plan: `docs/plans/PHASE13_CLOSEOUT_PLAN.md` Bloco D.

Question: do the Bottleneck / Fire units cut the cost of the stride-1 bodies without losing accuracy? Body = the hybrids' stride-1 GAP + BN net,
no skip (noise on the 2×2 body, above), every conv replaced by a unit whose one spatial conv is k×k: unit (Bottleneck 1×1 → k×k → 1×1, mid =
out/4 ≥ 32 / Fire 1×1 squeeze to out/4 → 1×1 ‖ k×k expands of out/2) × units per stage (one, beside the 5-conv cell / two, beside `k<k>stacked`)
× kernel (3×3 / 2×2) = 8 cells, seed 42. `k3x3fire` (one per stage) **is** `AlexNetFire` layer for layer and, from the same seed, weight for weight
(`tests/test_registry.py`), so it is read from `phase_11_families/alexnet_fire` (seed 42, 54.57 FP32 / 54.44 INT8): 7 new runs.

| cell (stride 1, GAP + BN) | params | MACs (direct) | reference body (params, MACs) |
|---|---|---|---|
| `k2x2bottleneckstacked` / `k2x2firestacked` | 0.53 M / 0.60 M | 219 M / 211 M | `k2x2stacked` 2.33 M, 803 M (60.47 FP32, 3 seeds) |
| `k2x2bottleneck` / `k2x2fire` | 0.28 M / 0.32 M | 101 M / 99 M | `k2x2` 1.05 M, 298 M |
| `k3x3bottleneckstacked` / `k3x3firestacked` | 0.74 M / 1.00 M | 329 M / 368 M | `k3x3stacked` 5.18 M, 1803 M (58.70, 3 seeds) |
| `k3x3bottleneck` / `k3x3fire` (= AlexNetFire, read) | 0.39 M / 0.52 M | 156 M / 178 M | `k3x3` 2.30 M, 669 M |

Code: `_AlexBottleneck` / `_FireModule` take `kernel_size` (default 3; 2 = `_conv_bn`'s SymmetricPad2d rule; Fire keeps the attribute name
`expand3` so checkpoint keys stay put); `AlexNetStacked` takes `unit=` ("conv" / "bottleneck" / "fire") and `per_stage=` (2 / 1). Defaults
unchanged: a layout digest (state_dict keys + shapes + fuse map) of all 785 registered nets is identical before and after; 8 cells added.
Tests: `test_config` + `test_registry` 36 passed, the `test_quantization` slice of the 7 new cells 14 passed (QAT → INT8 = the literature
definition, layer by layer). Analysis: `scripts/phase13/bottleneck_fire.py` (2026-11-10; `tests/test_phase13_bottleneck_fire.py`, fake run tree) reads the 7 runs + `alexnet_fire` + the four plain stride-1 bodies -> `results/phase_13_bottleneck_fire/{all_runs,contrasts}.csv`, `01_bottleneck_fire.png`; not run on real data yet (waits for the first run, job 835668).
Smoke (laptop RTX 4060 Laptop, 2026-11-07 22:44, `scripts.train --smoke`, output discarded): `k2x2bottleneckstacked` and `k2x2firestacked`
(the new code: both units at 2×2) FP32 → QAT → INT8 exit 0, 10 and 9 min; one FP32 epoch 50 / 44 s, one QAT epoch 196 / 166 s, GPU peak 5.1 / 4.2 GB,
MemAvailable never under 3.3 GB; the INT8 kernel check (exact int32) passed on both. The one `must run observer before calling calculate_qparams`
warning also sits in the `.err` of finished runs (`k2x2stacked` s1 835242, `alexnet_fire` 832311): not new.

## 2026-11-11: Stage B, 3 of 7 runs in

Synced from PCAD (run dirs only, no `*.pth` / logs; summaries md5-identical to PCAD's): `k2x2bottleneckstacked` (835668, tupi3, 4 h 42),
`k2x2bottleneck` (835670, tupi3, 4 h 11), `k2x2firestacked` (835669, beagle, 17 h 18). All 500 + 50 epochs, FP32 best epoch 472-488
(none early), `agreement_qat_int8` 0.977-0.983. `k2x2fire` runs on beagle (epoch 170 of 500 at 08:40); the three 3x3 cells are pending.
The aggregate CSV waits for the last run. `bottleneck_fire` rerun; matched pairs, test top-1 FP32 / INT8, B - A:

| swap | kernel, units per stage | FP32 | INT8 | params | MACs |
|---|---|---|---|---|---|
| conv -> Bottleneck | 2x2, one | 56.38 -> 53.77 (-2.61) | 56.22 -> 53.89 (-2.33) | -73 % | -66 % |
| conv -> Bottleneck | 2x2, two | 60.35 -> 58.57 (-1.78) | 60.79 -> 58.85 (-1.94) | -77 % | -73 % |
| conv -> Fire | 3x3, one (`alexnet_fire`) | 57.66 -> 54.57 (-3.09) | 57.56 -> 54.44 (-3.12) | -78 % | -73 % |
| conv -> Fire | 2x2, two | 60.35 -> 60.11 (-0.24) | 60.79 -> 59.80 (-0.99) | -74 % | -74 % |
| Bottleneck -> Fire | 2x2, two | 58.57 -> 60.11 (+1.54) | 58.85 -> 59.80 (+0.95) | +13 % | -4 % |

So far (seed 42 only; seed noise +-0.7 pp FP32): every unit costs accuracy against its plain body except `k2x2firestacked`, which keeps
FP32 within the noise at a quarter of the parameters and MACs.

**Seeds 43 / 44 of `k2x2firestacked` queued (2026-11-11).** Its contrast with `k2x2stacked` (-0.24 FP32 / -0.99 INT8) is the one
inside the seed noise, and the one a "Fire at 2x2 keeps the accuracy at a quarter of the cost" claim would rest on; `k2x2stacked`
has seeds 42-44, so the pair compares at 3 seeds. `phase_13_bottleneck_fire_seed43/44.yaml` (32b7403; same protocol, test_config),
submitted with `--slurm tupi_beagle` at the end of the queue (user): 836100 (s43), 836101 (s44), priority 503 behind the three
pending 3x3 cells (2625). If `k3x3firestacked` also lands inside the noise against `k3x3stacked` (3 seeds), its seeds join these yamls.
When they land: sync, and `bottleneck_fire` gains a `contrasts_by_seed.csv` (as `hybrids.contrasts_by_seed`).

**Seeds 43 / 44 of the conv1 11x11 + 2x2 body with the skip queued; Stage B `contrasts_by_seed.csv` in place (2026-11-11, later).**
`alexnet_k11-2residual_stride1_2pool2x2_map16_gap_bn` (60.48 FP32 / 60.85 INT8 at seed 42, tied with the two best nets; the article's
"Poucas seeds" limitation named it) added to `phase_13_hybrids_seed43/44.yaml` (4636092, as the 3x3 counterparts were on 2026-11-05) and
submitted through `~/queue_followups.txt` on PCAD (`feed_queue.sh`, QOS normal, MAX 10; `--slurm tupi_4090`): 836175 (s43), 836179 (s44).
`hybrids.py` reads them with no change; when in, its `contrasts_by_seed.csv` gains the "conv1: kernel do corpo -> 11x11" pair at s43 / s44.
`bottleneck_fire` now writes `contrasts_by_seed.csv` through `hybrids.contrasts_by_seed` (which takes the pairs function; 96a3f84): seed 42
only for now, = `contrasts.csv`; the `k2x2firestacked` seeds need just a rerun. tupi3-6 were drained for a BIOS update from 2026-11-10;
tupi3 runs `k3x3bottleneckstacked` (835672) since 2026-11-11 ~14:50.

## 2026-11-12: Stage B, 6 of 7 runs in

Synced from PCAD (run dirs only, no `*.pth` / logs; summaries md5-identical to PCAD's): `k2x2fire` (835671, beagle, 15 h 34),
`k3x3bottleneckstacked` (835672, tupi3, 4 h 35), `k3x3firestacked` (835673, tupi3, 4 h 33). All 500 + 50 epochs, FP32 best epoch 480-494
(none early), `agreement_qat_int8` 0.974-0.985. `k3x3bottleneck` (835674, beagle) was at FP32 epoch 494 of 500 at 08:55, QAT next; the
aggregate CSV still waits for it. `bottleneck_fire` rerun; `runs_index.csv` gains the 3 rows (the 54 archive rows kept). New matched pairs,
test top-1 FP32 / INT8, B - A, seed 42:

| swap | kernel, units per stage | FP32 | INT8 | params | MACs |
|---|---|---|---|---|---|
| conv -> Bottleneck | 3x3, two | 58.53 -> 58.93 (+0.40) | 58.27 -> 58.34 (+0.07) | -86 % | -82 % |
| conv -> Fire | 3x3, two | 58.53 -> 60.20 (+1.67) | 58.27 -> 59.88 (+1.61) | -81 % | -80 % |
| conv -> Fire | 2x2, one | 56.38 -> 52.82 (-3.56) | 56.22 -> 52.77 (-3.45) | -70 % | -67 % |
| Bottleneck -> Fire | 3x3, two | 58.93 -> 60.20 (+1.27) | 58.34 -> 59.88 (+1.54) | +35 % | +12 % |
| Bottleneck -> Fire | 2x2, one | 53.77 -> 52.82 (-0.95) | 53.89 -> 52.77 (-1.12) | +13 % | -2 % |

Medians over the pairs so far, FP32 / INT8: conv -> Bottleneck -1.78 / -1.94 (3 pairs), conv -> Fire -1.66 / -2.06 (4),
Bottleneck -> Fire +1.27 / +0.95 (3).

So far (seed 42 only): one unit per stage loses 2.6-3.6 pp in every cell; two per stage keep or gain. `k3x3firestacked` is the one unit
that beats its plain body: 60.20 against `k3x3stacked`'s 58.53 / 58.85 / 58.73 (seeds 42-44, mean 58.70), +1.50 over the mean and
+1.35 over the best seed, at a fifth of the params and MACs. With Fire, two per stage at 3x3 and at 2x2 land together (60.20 vs 60.11),
where the plain bodies are ~1.8 pp apart (`k2x2stacked` mean 60.47). The 2026-11-11 rule queued seeds for `k3x3firestacked` only if it
landed inside the noise; it landed outside. The user asked for the 3-seed check anyway (2026-11-12, 2 runs): `k3x3firestacked` joins
`phase_13_bottleneck_fire_seed43/44.yaml` and goes at the end of `~/queue_followups.txt` on PCAD (`--slurm tupi_beagle`, as its
`k2x2firestacked` seeds); `bottleneck_fire`'s `contrasts_by_seed.csv` picks both up with no change.

Phase 13 jobs on PCAD at ~09:00: 835674 (`k3x3bottleneck`, beagle) and 836100 (`k2x2firestacked` s43, tupi1, FP32 epoch 31) running;
836101 (s44) waits for a node; 836175 / 836179 (`k11-2residual` s43 / s44) pending (priority). tupi1-6 are back from the BIOS drain.

## 2026-11-13: Stage B, 7 of 7 runs in; `k2x2firestacked` seed 43

Synced from PCAD (run dirs only, no `*.pth` / logs; 24 files md5-identical to PCAD's): `k3x3bottleneck` (835674, beagle, 15 h 29) and
`k2x2firestacked` seed 43 (836100, tupi1, 8 h 11), plus `outputs/pcad/aggregates/phase_13_bottleneck_fire_comparison.csv` (7 rows, one
per run). Both 500 + 50 epochs, FP32 best epoch 495 / 486 (none early), `agreement_qat_int8` 0.982 / 0.979. `bottleneck_fire` rerun
(figure 01 now has all 24 points); `runs_index.csv` gains the 2 rows (the 54 archive rows and the existing row order kept).

`k3x3bottleneck`: 55.98 FP32 / 55.91 INT8, 0.38 M params, 156 M MACs. Its two pairs, seed 42, B - A:

| swap | kernel, units per stage | FP32 | INT8 | params | MACs |
|---|---|---|---|---|---|
| conv -> Bottleneck | 3x3, one | 57.66 -> 55.98 (-1.68) | 57.56 -> 55.91 (-1.65) | -83 % | -77 % |
| Bottleneck -> Fire | 3x3, one (`alexnet_fire`) | 55.98 -> 54.57 (-1.41) | 55.91 -> 54.44 (-1.47) | +34 % | +14 % |

All 12 pairs at seed 42, FP32 B - A (INT8 in `results/phase_13_bottleneck_fire/contrasts.csv`):

| swap | 3x3, one | 2x2, one | 3x3, two | 2x2, two | median (FP32 / INT8) | params / MACs (median) |
|---|---|---|---|---|---|---|
| conv -> Bottleneck | -1.68 | -2.61 | +0.40 | -1.78 | -1.73 / -1.80 | -80 % / -75 % |
| conv -> Fire | -3.09 | -3.56 | +1.67 | -0.24 | -1.66 / -2.06 | -76 % / -74 % |
| Bottleneck -> Fire | -1.41 | -0.95 | +1.27 | +1.54 | +0.16 / -0.09 | +24 % / +5 % |

Stage B at seed 42, closed: one unit per stage costs 1.7-3.6 pp in all four cells, and there Bottleneck beats Fire (by 0.9-1.4 pp).
At two per stage the order flips: Fire beats Bottleneck by 1.3-1.5 pp and keeps (2x2, -0.24) or beats (3x3, +1.67) its plain body at a
quarter to a fifth of the params; Bottleneck keeps the 3x3 body (+0.40) and loses 1.8 pp on the 2x2 one. Units per stage decides more
than the unit: the Bottleneck -> Fire median (+0.16) averages two opposite signs.

Seed 43 of `k2x2firestacked`: 59.97 FP32 / 59.88 INT8 against `k2x2stacked` s43 60.65 / 60.56: -0.68 / -0.68 (seed 42: -0.24 / -0.99).
Means over the seeds so far, FP32 / INT8: `k2x2firestacked` 60.04 / 59.84 (s42-43), `k2x2stacked` 60.47 / 60.63 (s42-44): -0.43 / -0.79.
The FP32 gap is inside the ±0.7 seed noise, the INT8 one a little wider; s44 settles it.

Jobs at 08:25: 836101 (`k2x2firestacked` s44, beagle) at FP32 epoch 497 of 500, QAT next (ends in the afternoon). `k3x3firestacked`
s43 / s44 are not submitted yet: the last 2 lines of `~/queue_followups.txt`, waiting for a slot (QOS normal at 10 of 10 = 836101 +
9 pending on tupi by Priority, none of ours running there). `phase_13_bottleneck_fire_seed43_comparison.csv` waits for `k3x3firestacked` s43.

**Stage B in the article (2026-11-13, later; plan approved by the user, no submission date).** `report/artigo_kernel.tex`: new
subsection "Blocos Bottleneck e Fire barateiam o melhor corpo?" (`sec:unidades`, after `sec:combinacoes`) with `tab:unidades` (the 12
cells, seed 42: FP32, INT8, params, MACs, Winograd %), three paragraphs (one block per stage loses 1.68-3.56 pp; two per stage Fire
ties the 2x2 body and beats the 3x3 one, the 3x3 claim under `\pendente` until its seeds 43 / 44 land; Fire keeps 59.7 / 76.9 % of its
MACs on the accelerator, Bottleneck 40-64 %), the design in "Desenho dos experimentos", a `tab:blocos` row (7 (+1); total 279 -> 287 in
the abstract, "Dados" and the caption), a 10th question in the introduction, one sentence in `sec:hardware`, the Fire nuance in the
design bullets, the "Poucas seeds" limitation, and the Bottleneck/Fire item dropped from "Trabalhos Futuros". The Winograd % comes from
a new `winograd_pct` column of `results/phase_13_bottleneck_fire/all_runs.csv` (`bottleneck_fire` now applies
`design_figures.winograd_share`, the rule of the article's `tab:winograd`; test asserts the column). The 12 table rows were checked
against the CSV by script. PDF rebuilt: 21 pages, as before; one new 5 pt overfull vbox on a float page, nothing visible.
When the seeds land: the 2x2 Fire pair at 3 seeds, the 3x3 one replaces its `\pendente`, `tab:blocos` (+N).

**Table -> figure, and `report/` split (2026-11-13, later; user).** The article's `tab:unidades` became Figure `fig:unidades`:
`01_bottleneck_fire.png` redrawn at its printed size (8.9 x 4.2 in, constrained layout, legend under the panels, as `01_hybrids.png`;
`test_article_figures_print_at_a_readable_scale` would have failed the old 13 x 6 in), with each block's MACs and Winograd share
(3x3 above 2x2) under it, the other seeds as small FP32 dots beside their cell (`k2x2firestacked` s43 and the plain bodies' s43 / s44
show now; the Fire seeds will appear on a rerun), panels titled "um / dois blocos por estágio"; title cropped at 61.9 bp
(`test_artigo_trim` passes). PDF: still 21 pages, the figure on p. 15 under the hybrids figure, no overfull box left.
`report/` now holds one folder per document (8950cf5): `report/ic_report/` (IC report, its figures/ and generators, `pareto.py`) and
`report/artigo_kernel/` (the article); paths fixed in the scripts, `presentation/make_figures.py`, the two tests and the current docs;
both PDFs build from their folders with the same page counts (21 / 24).

## 2026-11-14: `k2x2firestacked` seed 44; the 2x2 Fire pair at 3 seeds

Synced from PCAD (run dir only, no `*.pth` / logs; summary md5-identical): `k2x2firestacked` s44 (836101, beagle), plus
`outputs/pcad/aggregates/phase_13_bottleneck_fire_seed43/44_comparison.csv` (one row each so far). FP32 best epoch 464 of 500,
`agreement_qat_int8` 0.977. The job hit beagle's 24 h wall in QAT (REQUEUED 2026-11-13 10:49, resumed from its epoch checkpoint, done
13:05): FP32 ran in one piece, QAT is a resumed stage. `bottleneck_fire` rerun (`contrasts_by_seed.csv` +1 row, figure 01 shows the s44
dot); `runs_index.csv` +1 row (the 54 archive rows and the row order kept); `results_cross_phase` unchanged.

s44: 59.35 FP32 / 59.03 INT8 against `k2x2stacked` s44 60.41 / 60.54. Fire - plain body, two units per stage, 2x2:

| seed | FP32 | INT8 |
|---|---|---|
| 42 | -0.24 | -0.99 |
| 43 | -0.68 | -0.68 |
| 44 | -1.06 | -1.51 |
| 3-seed means | 59.81 - 60.47 = -0.66 | 59.57 - 60.63 = -1.06 |

Same sign in 3 of 3 seeds, and outside the ±0.4 pp band of a 3-seed mean (±0.7 / √3, the article's rule): on the 2x2 body two Fire
units per stage cost a little (0.7 FP32 / 1.1 INT8) at a quarter of the params and MACs, they do not tie it as the 11-13 reading said.
The article's `sec:unidades` ("No 2x2, o Fire empata com o corpo comum") needs this.

Jobs at 08:45: `k3x3firestacked` s43 (837758, beagle) at QAT 10 of 50, ~250 s/epoch, ends ~11:30, inside its wall (13:08); s44 is the
last line of `~/queue_followups.txt` (feeder `QOS=normal MAX=10`). The f63c arm 836205 moved to QOS short (`scontrol update QOS=short
TimeLimit=04:00:00`, user OK; QOS short's cap is now 4 h, priority 10000), which also frees the normal slot the s44 line waits for.

**The article, same day (user: "um texto mais fluido", no bold run-in labels).** `report/artigo_kernel/artigo_kernel.tex` rewritten as
running prose: no paragraph or list item opens with a bold label any more (66 paragraphs + 20 items; the 4 study names and the 2
principles are italic), runs of short sentences joined, the Discussion's limitations are three paragraphs by theme; the design
recommendations and the 4-step pipeline stay lists, without labels. Structure, order, floats and captions untouched. The style pass was
checked by script against ab68098: preamble, the 29 floats and the bibliography byte-identical; the same 537 numbers, 54 cite keys and
65 refs in the prose. Then one content edit, kept apart: the 2x2 Fire pair at 3 seeds in `sec:unidades` ("fica um pouco abaixo",
-0,66 FP32 outside the ±0,4 band / -1,06 INT8), the Fire recommendation and the seeds limitation in the Discussion, `tab:blocos`
7 (+2), combinations 35, total 287 -> 288 (abstract, "Dados", caption). The 3x3 Fire pair keeps its `\pendente`. PDF rebuilt: 21 pages,
no overfull box; `test_artigo_trim` and `test_article_figures_print_at_a_readable_scale` pass. Plan:
`~/.claude/plans/crystalline-chasing-rocket.md`. Not committed.
