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
