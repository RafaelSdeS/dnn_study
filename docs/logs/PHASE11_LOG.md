# Phase 11 Implementation Log

Decision continuity across `/compact` boundaries. Append per finding.

---

## Run 1 — Kernel size sweep on PCAD (2026-09-13 → 2026-09-15) ✓

Submitted via `python -m scripts.cluster submit-sweep --experiment
phase_11_kernel_size_comparison --runtime pcad`, jobs 821242-821245 (AlexNets)
+ 821270-821271 (VGGs). All 5 models completed FP32 (500ep) + QAT (100ep),
no early stopping (`_protocols/no_patience.yaml`).

| Model | FP32 Top-1 | FP32 Top-5 | QAT Top-1 | QAT Top-5 | INT8 Top-1 | INT8 Top-5 |
|-------|-----------:|-----------:|----------:|----------:|-----------:|-----------:|
| alexnet_tv_scratch (orig. 11×11/5×5/3×3) | 27.51% | 52.28% | 23.55% | 46.46% | 22.91% | 46.72% |
| alexnet_tv_3x3 | 24.84% | 48.47% | 23.60% | 46.51% | 23.13% | 46.35% |
| alexnet_tv_2x2 | 23.93% | 46.26% | 22.50% | 44.90% | 22.03% | 44.64% |
| **vgg16_original** (native 3×3) | 47.71% | 70.22% | **0.50%** | **2.50%** | **0.50%** | **2.50%** |
| vgg16_2x2 | 54.26% | 75.96% | 49.03% | 69.09% | 48.29% | 67.73% |

Results aggregated to `results/phase_11_kernel_size_comparison/` and
`results/runs_index.csv` via `scripts/build_runs_index.py`. Renamed the
`vgg16` run dir/model slot to `vgg16_original` (both PCAD and local) once the
naming ambiguity with `vgg16_2x2` became confusing to read in tables.

## Finding — vgg16_original's QAT stage collapses to random-guess (2026-09-15)

**Symptom:** FP32 47.71% top-1 → QAT/INT8 0.50% (= 1/200, exactly random).
vgg16_2x2, same fuse map, same lr, same QAT schedule, degrades normally
(54.26% → 49.03%, a typical ~5pp QAT drop).

**Root-cause investigation** (PCAD job 821270 stderr,
`outputs/pcad/logs/phase_11_kernel_size_comparison/train.sbatch-821270.err`):

- QAT epoch 1 val_acc is *already* 0.50% — the model is broken before any
  fine-tuning happens, at the moment fake-quantization is enabled.
- Epochs 1-5 (observers still active) oscillate near-randomly; at epoch 6
  (`disable_observer_epoch: 5` in `configs/qat.yaml`) the loss locks to
  exactly `ln(200) = 5.2983` bit-for-bit for the remaining 94 epochs.
  `disable_observer` isn't the cause — it just freezes an already-broken
  state permanently (compare vgg16_2x2, which is *already climbing* by
  epoch 1 — 4.74% → 6.47% — and jumps to 31.70% the instant observers
  disable at epoch 6).
- No NaN/Inf in the logs — this is a silent quantization-range collapse,
  not a crash.
- Checked `BatchNorm.running_var` directly off the two FP32 `_best.pth`
  checkpoints (no forward pass needed): vgg16_original runs 2-12x hotter
  than vgg16_2x2 across nearly every conv stage, e.g. `features.18`
  216 vs 19.9, `features.21` 219 vs 18.9, `features.25` 275 vs 29.9; the
  final stage (`features.41`) hits running_var ~15,384 vs ~5,429.

**Hypothesis:** kernel_size=3 with `padding=1` on *every* one of the 13
conv layers (VGG16's native design, see `models/baselines.py:_vgg16_features`)
preserves spatial size at every stage with no padding-parity break, letting
activation variance compound layer-to-layer far more than the kernel_size=2
variant (which alternates 1/0 padding per stage). With INT8 fake-quant's
fixed ~256-level range, a BN-fused conv layer with this much activation
variance saturates almost every activation to the same quantization bin,
and with 13 such layers in series the information loss compounds into a
fully collapsed (uniform-output) network from the very first QAT forward
pass — well before any gradient step could plausibly cause it.

**Status:** not a code bug — this looks like a genuine numerical instability
in quantizing the native (unrestricted) VGG16 kernel size, arguably itself
a relevant Phase 11 finding (kernel restriction can be *required* for
quantizability, not just efficiency). Left as-is; not retrained.

**If revisited**, options in decreasing preference: (1) report the failure
as-is as a finding, (2) swap `MovingAverageMinMaxObserver` for
`HistogramObserver` in `ml/quantization.py`'s qconfig (more outlier-robust,
untested here), (3) retrain FP32 with stronger weight decay / lower BN
momentum to control activation variance, then redo QAT.

## Revisit — classifier fusion, a partial fix (2026-09-15)

Option (2) above was tried directly on PCAD (uncommitted `ml/quantization.py`
+ `ml/model_registrations.py` edit, `HistogramObserver` swap with
`quant_max=127`) and resubmitted as job 821600. It collapsed identically —
val_acc pinned at 0.49% from epoch 1, loss locked to `ln(200)` from epoch 6.
Cancelled. This ruled out the BN-running_var/observer-choice hypothesis:
no per-tensor affine observer, however chosen, can fix what turned out to be
a different layer entirely.

**Real root cause** (`outputs/pcad/.../vgg16_original/results/vgg16_layer_stats.json`):
`classifier.0` (`Linear(25088→4096)`, the first FC layer after the conv
stack — **not** a conv layer, so outside the BN-variance hypothesis) has a
genuinely heavy-tailed raw output: `p999 ≈ 27,153`, `p9999 ≈ 68,322`,
`max ≈ 115,465`. But `classifier.3`'s *input* — i.e. what's left after
`classifier.0 → ReLU → Dropout` — tops out at `0.49`. Nearly all of that
huge magnitude sits on units ReLU zeroes; only a small positive remainder
survives.

The bug: `classifier`'s `Linear→ReLU` was never fused for QAT (true for
*every* model in this codebase, `fuse_root_attr` only ever covers
`features`) — unlike the conv stack, which is always Conv-BN-ReLU fused.
So the fake-quant observer sat on the *raw pre-ReLU* Linear output, forced
to size an INT8 range around ±115k to cover values ReLU discards anyway,
crushing the real signal (~0–0.5 post-ReLU) into 0–1 quant levels. This is
a latent gap in every model's classifier, but only vgg16's classifier.0
output is extreme enough to actually collapse training; e.g. alexnet_tv's
classifier is equally unfused and trains fine.

**Fix, part 1 — necessary, not sufficient (see Revisit 2)** (`ml/quantization.py` `prepare_qat_model`/`build_qat_from_model`,
`ml/model_registrations.py`): added an optional `classifier_fuse_map`
registry field; when set, `classifier`'s Linear-ReLU pairs get fused
(`nniqat.LinearReLU`) same as the conv stack, moving the observer to the
post-ReLU value. Applied to both `vgg16` and `vgg16_2x2` (shared classifier
head, `CLASSIFIER_FUSE_MAP_VGG16 = [["0","1"], ["3","4"]]`; `classifier.6`
has no ReLU after it and stays a standalone quantized Linear) — re-running
`vgg16_2x2`'s QAT stage too so both kernel sizes share the same graph for a
fair comparison, even though `vgg16_2x2` trained fine under the old
(unfused) graph; its 49.03%/69.09% QAT/INT8 numbers above are superseded
once the rerun lands. Every other registered model is untouched
(`classifier_fuse_map` defaults to `None` → identical qconfig/fuse
behavior as before). Regression test:
`tests/test_quantization.py::test_vgg16_classifier_linear_relu_fusion_for_qat`.
No FP32 retrain needed — both models reuse their existing FP32 checkpoints.

## Revisit 2 — the observer-freeze trap (2026-09-15)

Resubmitted with the classifier fusion as job 821691. It learned while
observers were active — val_acc 0.66 → 0.82 → 0.86 → 1.32 → 1.53% over
printed epochs 1-5 (the unfused original was flat at 0.50%) — then collapsed
at printed epoch 6 (0.47%), with val_loss locked to 5.2983 = ln(200) from
epoch 9. Cancelled at epoch 11.

`Trainer.fit` calls `epoch_callback(epoch, model)` at the start of each epoch
with a 0-based index, so `disable_observer_epoch: 5` takes effect exactly at
printed epoch 6 — the collapse point in all three runs (821270, 821600,
821691). The trigger is the observer freeze, not the observer type
(MovingAverageMinMax and Histogram both collapsed) and not the BN freeze
(`freeze_bn_epoch: 3` = printed epoch 4, where 821691 kept improving).

What differs from vgg16_2x2 (FP32 `layer_stats`, `activation_in` p999 — the
post-BN-ReLU value a fused layer's observer actually sees; the Conv2d
`activation_out` figures, e.g. `features.40` max 55,422, are hooked on the raw
conv *before* BN and are not):

| layer | vgg16 | vgg16_2x2 |
|---|---:|---:|
| `features.37` | 3.37 | 2.47 |
| `features.40` | 65.1 | 0.94 |
| `classifier.0` | 250.3 | 0.015 |

vgg16's last conv block feeds the classifier activations 70× to 16,000×
hotter than vgg16_2x2's.

Mechanism (consistent with the data, not directly measured): once scales
freeze, a per-tensor scale sized for that range ends up coarser than the
logits' spread as training shrinks it; every logit rounds to the same value,
the output is uniform, loss is exactly ln(200), and no weight step smaller
than one quant step changes the forward pass, so training never escapes.
While observers are active the scale shrinks along with the signal.
vgg16_2x2, whose ranges are small, instead *jumps* (6.47% → 31.70%) when its
observers freeze.

**Fix, part 2 (vgg16 only):** `qat_disable_observer_epoch=None` in its
registry entry → `make_qat_callback` never disables observers (BN still
freezes at epoch 3). **Protocol deviation, recorded on purpose:** vgg16's QAT
keeps observers active for all 100 epochs, so activation ranges also update
during validation passes (already true for the first 5 QAT epochs of every
other model). vgg16_2x2 and the AlexNets keep the standard schedule.

Gate for the resubmission: val_loss must not lock at 5.2983 past epoch 6, and
val_acc must still be climbing (> ~3%) at epoch 15; otherwise cancel. Next
options if it fails: learnable ranges (LSQ via `_LearnableFakeQuantize`,
INT8-convert compatibility untested), report the collapse as a Phase 11
finding, or retrain FP32 with stronger weight decay.

## Revisit 3 — gate passed, final numbers (2026-09-15, job 821696)

Gate cleared: no lock, val_acc climbing throughout, best epoch 99/100 (not an
early-epoch fluke). Combined fix (classifier fusion + observers never frozen)
holds through the full 100-epoch QAT budget:

| Model | FP32 Top-1 | QAT Top-1 | INT8 Top-1 |
|-------|-----------:|----------:|-----------:|
| vgg16 (fixed) | 47.71% | 44.65% | 44.77% |
| vgg16_2x2 (rerun, same graph) | 54.26% | 49.07% | 48.36% |

`vgg16` and `vgg16_2x2` now both quantize like every other Phase 11 model —
a normal few-point drop, not a collapse. `vgg16_original`'s checkpoint dir is
kept (model_name field renamed to `vgg16_original` in its summary.json to
stop it colliding with the fixed `vgg16` row in `aggregate_results.py`'s glob)
purely as the pre-fix collapse's provenance; it's excluded from
`results/phase_11_kernel_size_comparison/` and the cross-phase rollup.

Separately found and fixed while syncing this run's data: `scripts/train.py`'s
`make_run_summary(fit_results=fp32_fit or qat_fit, ...)` silently mislabeled
QAT's epoch/best-val numbers as FP32's whenever the FP32 stage was skipped
because a checkpoint already existed (exactly this resubmission's case) —
`vgg16`'s and `vgg16_2x2`'s `results/*_summary.json` briefly reported
`epochs_used: 100`/`best_val_top1: <QAT's value>` instead of FP32's real
500/47.74 (54.29). Fixed to only fall back to `qat_fit` when FP32 was never
in `stage_list` at all; the two affected summaries were corrected from their
still-intact `checkpoints/*_meta.json` sidecars. Unrelated to the
classifier-fusion/observer-freeze bug above — a reporting bug, not a training
one.

## Finding — alexnet_tv_mixed_early2's FP32 stage collapses to random-guess (2026-09-17, job 822065)

`phase_11_mixed_kernel_comparison.yaml`'s sweep (8 models, jobs 822062-822069)
completed 7/8 cleanly. `alexnet_tv_mixed_early2` (2x2 kernels on the first 3
conv layers, 3x3 on the last 2 — see `_ALEXNET_KERNEL_SPECS` in
`models/baselines.py`) never learned: `val_loss` locked at `ln(200)=5.298` and
`val_acc` at chance (~0.3%) from epoch 1 through the full 500-epoch FP32
budget, LR decayed to 0, no recovery.

This is the same "no-BN, from-scratch net stuck at ln(200)" pattern as
`vgg16_original`'s QAT collapse above and the plain-VGG16 FP32 stall noted in
CLAUDE.md — `AlexNetTV` has no BatchNorm and `alexnet_tv_mixed_early2` trains
from scratch (`pretrained=False`), so a bad random init can saturate every
ReLU from step 1 with nothing to rescale activations and let it escape.
~~Seed-luck, not an architectural property.~~ **First diagnosis was wrong** —
see the next section. The retry at `seed: 43` (job 822325) collapsed
identically, which ruled seed luck out. The dead run's checkpoints were moved
aside to `alexnet_tv_mixed_early2_dead_seed42/` (kept for provenance, same
treatment as `vgg16_original` above) because `scripts/train.py` skips FP32
training whenever `{model}_best.pth` already exists — resubmitting into the
same directory without moving the dead checkpoint out would have skipped
straight to QAT on the collapsed FP32 weights.

## Root cause — torchvision's AlexNet ships no weight init (2026-09-17)

Reproduced locally against the real data pipeline (Tiny ImageNet, same
AMP/AdamW/label-smoothing/wd as the protocol). The result that broke the seed
theory: **`mixed_early3` died locally at seed 42 while `mixed_early2` trained
fine** — the exact inverse of PCAD. Which model dies is a coin flip, so it is
a property of the family, not of a kernel layout.

The mechanism, measured:

1. `torchvision.models.AlexNet` defines **no weight initialization at all**
   (verified by source inspection), so every conv falls back to PyTorch's
   `nn.Conv2d` default, `kaiming_uniform_(a=sqrt(5))` → `std=sqrt(1/(3*fan_in))`.
   That is `sqrt(6)` ≈ 2.45x smaller per layer than He, and it compounds: a
   from-scratch `AlexNetTV` produces logits with std ≈ 0.01.
2. Those near-zero logits put the net *on* the `loss = ln(200)` plateau from
   step 0, where the gradient reaching conv0 is ~1e-4 (measured).
3. Adam normalizes that tiny, noisy gradient into full-size `lr` steps — a
   random walk. The dead-ReLU fraction then creeps up monotonically
   (measured: 69% → 77% over one epoch).
4. With no BatchNorm to rescale, 100% dead ReLUs is an **absorbing state**:
   conv0's gradient hits *exactly* `0.00e+00` (measured), and from then on only
   weight decay acts, so the loss stays pinned at `ln(200)` forever.

Every run is therefore a race between escaping the plateau and ReLU death.
`alexnet_tv_scratch/3x3/2x2/mixed_alt/mixed_early3` won that race on PCAD; the
suspiciously flat 23.9–27.5% band they all landed in is consistent with having
spent a long time on the plateau first. torchvision's **VGG** never hits any of
this because its constructor *does* apply He init — which is exactly why
`vgg16` trains from scratch here (47.71%) while the AlexNetTV family is a coin
flip, and why CLAUDE.md's plain no-BN VGG16 note ("stuck at ln(200) for 22
epochs") is the same phenomenon one notch weaker.

**Fix:** `models/baselines.py:he_init`, applied when `pretrained=False` (never
over pretrained weights — `alexnet_tv`'s conv0 was verified bit-identical to
torchvision's after the change). Same for the hand-written no-BN CNNs in
`models/alexnet_variants.py` (`AlexNetStacked`/`AlexNetMixed`/
`AlexNetSmallKernel`), which had no init either. It is torchvision VGG's recipe
verbatim: `kaiming_normal_(fan_out, relu)` on convs, `normal_(0, 0.01)` on
Linears. (He on the Linears too was tried and reverted — it does not rescue the
`alexnet_stacked_fc_nobn` case it was meant for, see below, and it inflates the
initial logits from std 0.08 to 3.77.)

Validated on the real data pipeline at seed 42 — `alexnet_tv_mixed_early2`, the
model that died twice on PCAD, escapes the plateau by step ~600 and reaches
loss 4.92 after 2 epochs (vs. pinned at 5.2989 before), with conv0's gradient at
~1.3 instead of ~1e-4 and dead ReLUs flat at ~78% instead of climbing.
`mixed_early3` likewise recovers.

**Scope note:** BN variants are insensitive to this (BN normalizes the previous
layer's scale away), so the fix changes only the no-BN cells — the broken ones.
But every from-scratch `AlexNetTV`-family result already in
`results/phase_11_kernel_size_comparison/` and
`phase_11_mixed_kernel_comparison` predates it and carries the old init.

## Finding — no-BN + 3-layer-FC head is a second, harder case (2026-09-17)

`alexnet_stacked_fc_nobn` (a head/BN ablation cell added the same day: 10 convs,
no BN, FC head) hit the identical `ln(200)` lock on PCAD — 27 epochs at 0.31%
(job 822332, cancelled). He init on the convs alone does **not** rescue it: the
forward signal is healthy (activation std ~0.22–0.30 through all 10 convs,
measured) but the gradient dies on the way back through the FC head. Its
siblings isolate the two causes exactly — `alexnet_stacked` (same convs, *with*
BN) and `alexnet_stacked_gap_nobn` (same convs, no BN, but a *single* Linear)
both train fine. So depth alone is fine and no-BN alone is fine; it is no-BN
*and* a 3-layer FC head together that fails. Tried and rejected: He on the
Linears as well (conv0's gradient still reaches exactly 0, by step 700, and it
stays there for 4 full epochs — measured), and larger/Xavier classifier inits
(same plateau). This looks like the head/BN ablation's actual answer rather than
a bug to paper over: BN's role in this architecture family is what makes the
deep FC variant trainable at all. Treat that cell as "does not train" unless
the protocol itself changes (warmup, lower LR, or BN).

## Rerun after the init fix (2026-09-17, jobs 822344-822360)

Everything trained from scratch in the AlexNet family predates `he_init`, so it
was resubmitted rather than kept. Old outputs are preserved under
`outputs/pcad/archive_old_init/<experiment>/<model>/` — moved, not deleted,
because `scripts/train.py` skips the FP32 stage whenever `{model}_best.pth`
already exists and would otherwise have run QAT straight off the old weights.

| What | Jobs | Note |
|------|------|------|
| `alexnet_tv_mixed_early2` | 822344 | the originally broken model |
| `phase_11_head_bn_ablation`, all 11 cells | 822345-822355 | restarted from scratch; `alexnet_mixed`/`alexnet_stacked` added as reference cells so all 8 factorial cells share one init |
| `alexnet_tv_scratch`/`_3x3`/`_2x2` | 822356-822358 | `phase_11_kernel_size_comparison` |
| `alexnet_tv_mixed_alt`/`_early3` | 822359-822360 | `phase_11_mixed_kernel_comparison` |

`vgg16`/`vgg16_2x2` are untouched: torchvision's VGG already applied He init, so
they never had the bug — they are the control that made it visible.

**Not yet updated:** the curated `results/phase_11_*` trees, the cross-phase
rollup and `report/ic_report.tex` still carry the pre-fix numbers (the 23.9–27.5%
band). Re-aggregate once these land.

## Post-fix results land + mixed-kernel/head-BN comparison figures (2026-09-19 to 2026-09-21)

`results/phase_11_kernel_size_comparison_final_comparison.csv` and
`phase_11_mixed_kernel_comparison_final_comparison.csv` re-synced from the
he_init reruns above (jobs 822531-822533 range). `alexnet_tv_scratch_dead_heinit`
is kept as a row (0.5% top-1) — a second seed-42 retry that still died despite
the fix, alongside the surviving `alexnet_tv_scratch` (27.51% FP32); both are
preserved rather than overwritten so the plateau's non-determinism stays visible
in the data, not just in prose.

**Still not updated:** `results/results_aggregate/` (cross-phase rollup) and
`report/ic_report.tex` — neither has a commit since 45e8ba9, which predates this
resync. Re-run `scripts/build_cross_phase_results.py` and regenerate the report
figures before citing Phase 11 numbers from either.

`alexnet_tv_mixed_early2_gap` registered (`ml/model_registrations.py`) and added
to `phase_11_head_bn_ablation.yaml`, closing the last FC/GAP pairing gap in the
mixed-kernel sweep (the seed-43 retry for its non-GAP twin had been held back
pending exactly this confirmation — see the finding above). PCAD result landed
(26.51% FP32, 26.48% INT8) at
`outputs/pcad/phase_11_head_bn_ablation/alexnet_tv_mixed_early2_gap/` but has
**not** been folded into `phase_11_head_bn_ablation_final_comparison.csv` yet —
run `scripts/aggregate_results.py` for that experiment to pick it up.

New `scripts/phase11/plot_kernel_comparison.py` renders 10 single-question PNGs
(kernel pattern × head × architecture family) straight from the three curated
Phase 11 result trees into
`results/figures_generated/phase_11_kernel_size_comparison/`, since the FC/GAP
pairing for the mixed-kernel models spans two separate experiment configs and
no single `models:` list covers it.

## M7 — Winograd variant/packing accuracy cost (2026-09-19)

Added `QATWinoConfig` (`ml/config.py`) on top of `QATConfig`: `variant`
(`f23`/`f43`/`f63`) and `pack`/`u_w`/`v_w`/`k_dsp`, threaded through
`load_qat_wino_model(..., cfg=)` into `qat_wino.convert()`. Until this point the
bridge called `convert(model)` bare, which silently pinned F(4,3) with **no**
packing — every one of budget_unico's 14 accuracy runs is that one combination,
while the deploy bitstream packs (2 mult/DSP). `cfg=None` keeps that old
behavior so existing callers are unaffected.

Three new experiments, `configs/experiments/wino_f{23,43,63}_pack.yaml`, rerun
just the `qat_wino` stage (FP32 checkpoint reused from budget_unico, same
weights/seed) at the iso-DSP-budget packed point (`u_w=9, v_w=8, k_dsp=2` —
`u_w + 2*v_w <= 25` is the shared ceiling across all three, per the DSP48E2 port-A
budget) on 2 models (`alexnet_fire_bypass_fpga`, `vgg_style_fpga`). Results
(best val top-1 vs. the existing un-packed-F43 FP32 numbers):

| Variant | alexnet_fire_bypass_fpga | vgg_style_fpga |
|---------|--------------------------|-----------------|
| F23, packed | +0.09pp | -0.14pp (~unchanged) |
| F43, packed | -3.93pp | -8.28pp (moderate) |
| F63, packed | -15.65pp | -30.55pp (severe) |

This resolves budget_unico's `≠HW` caveat for these two models: packing itself
costs little, F(6,3) is the expensive choice. Full analysis in the sibling
Winograd-FPGA repo's `achados_varredura.md §6`.

## Finding — geometry confound: AlexNetTV vs. the adapted AlexNet family (2026-09-24)

Shape-traced every AlexNet-family model at 64×64 (forward pass on a zero tensor, no training).
"Kernel size is the only variable" is **false** both for Phase 11's `alexnet_tv_*` trio and for the
report's `AlexNet3x3-FC` vs. the pretrained baseline:

| Model | conv1 | pools | map into classifier | Dropout |
|-------|-------|-------|---------------------|---------|
| torchvision AlexNet @224 | 11×11 s4 | 3× MaxPool(3, s2) | 6×6 (55→27→13→6) | 2 |
| `AlexNetTV` @64 (orig / k=3 / k=2) | s4 | 3× MaxPool(3, s2) | **1×1** (15\|16→7→3→1), AAP(6,6) replicates it | 2 |
| `AlexNet3x3FC/GAP`, `Mixed`, `Stacked`, `Bottleneck`, `FinalFireResidual` | s2 | 2× MaxPool(2) | 8×8 (`Mixed` 6×6) | 0 |
| `AlexNetSmallKernel`, `Fire`, `FireBypass` | s1 | 2× MaxPool(2) | 16×16 (→ higher MACs) | 0 |
| `VGG16` k=3 / k=2 | s1 | 5× MaxPool(2) | 2×2 → AAP(7,7) upsample | 2 |

1. **Geometry is worth ~17pp, at matched conditions.** `alexnet_mixed` (adapted, GAP, no BN, 3-2-3-2-3) 45.28%
   vs. `alexnet_tv_mixed_alt_gap` (original geometry, GAP, no BN, 2-3-2-3-2) 27.90% FP32 — both in
   `phase_11_head_bn_ablation`, so same protocol and `he_init`; single seed, kernel order differs (the other
   two TV-GAP patterns land at 26.5–28.3%, so order barely matters). The original geometry is not broken
   (27.5% vs. 0.5% chance) but far below the adapted layout.
2. **`AlexNetTV(kernel_size=3/2)` is not overlap-preserving.** conv1 keeps stride 4, so it reads 3 of every 4
   input columns (k=3) or 2 of 4 (k=2): 56% / 25% of the pixels, vs. 100% for the original 11×11. Part of the
   27.51→26.50→25.03% trend is pixel loss, not kernel size.
3. **The trio mixes inits.** `alexnet_tv_scratch` is the pre-`he_init` run (job 821243, git 4e91e77; its
   `he_init` retry died — `alexnet_tv_scratch_dead_heinit`), while `_3x3`/`_2x2` are the `he_init` reruns (jobs
   822532/822533; the recorded git hash 120c5da predates `he_init` but `git_dirty: true`, and both accuracies
   moved vs. the pre-fix table at the top: 24.84→26.50, 23.93→25.03).
4. **The report's 35.79% vs. 26.50% (`AlexNet3x3-FC` vs. `alexnet_tv_3x3`, 9.3pp) is not "only stride/pool".**
   Dropout (2× vs. 0), init (`he_init` vs. PyTorch default) and protocol (Phase 2: early stopping, patience 5,
   37 ep; Phase 11: 500 ep) also differ. Protocol alone is worth ~+7pp on the same model: `alexnet_3x3_gap`
   40.32→46.82, `alexnet_2x2_gap` 33.15→40.27, `alexnet_mixed` 38.74→45.46 (Phase 2 vs. Phase 11 runs).
5. **The report's "3×3 gains 2.9pp over the baseline" (32.89→35.79) is confounded** by pretraining, geometry,
   Dropout and epochs (79 vs. 37); there is no adapted-geometry AlexNet with 11×11/5×5/3×3 kernels to isolate
   the kernel. The only same-geometry kernel evidence is item 2's trio (−1.0pp / −2.5pp), with items 2–3 caveats.
6. **BN confounds the compensation claim in the report.** `AlexNet3x3-GAP` has no BatchNorm; `AlexNetBottleneck`
   and `AlexNetFire` have 15 BN layers (shape trace). The report's "bottleneck adds +4.3pp over 3x3-GAP" therefore
   mixes the block with BN, and BN alone is worth +2.8 to +4.6pp in `phase_11_head_bn_ablation`
   (`alexnet_mixed` 45.28→`_bn` 48.37, `_fc` 37.20→`_fc_bn` 40.00, `alexnet_stacked_gap_nobn` 48.95→`_gap` 53.52;
   different protocol from the report's, same direction). Learning rate is a second confound: the report ran
   Bottleneck/Fire at 1e-3 (registry lr) and 3x3-GAP at 3e-4. The params/MACs advantage (0.39M / 39.5M vs.
   2.30M / 167.0M) is measured fact and unaffected.
7. `VGG16` here is torchvision cfg D **plus BatchNorm** (the original has none) and its 2×2 features are
   upsampled to 7×7 by AAP(7,7) at 64×64 — degenerate too, but milder than AlexNetTV's 1×1.

Corrected in this pass (nothing committed): `report/ic_report.tex` (architectures list, new "Geometria a 64×64"
paragraph, kernel-cost discussion, Limitações (i)), `docs/plans/MODELS.md`, `docs/plans/BEST_MODELS.md`,
`models/alexnet_variants.py`/`models/baselines.py` docstrings, `configs/experiments/phase_11_kernel_size_comparison.yaml`,
`TODO.md`, `CLAUDE.md`. `report/ic_report.pdf` and the Phase 11 figures were not rebuilt. Still open: the
missing controls listed in `TODO.md` (Phase 2 section) — none run.

## Controls submitted — geometry / kernel / BN (2026-09-24, PCAD jobs 824460-824467)

**Design history.** A first submission (jobs 824444-824447: 3 seeds × 9 models under the report's protocol — 100 ep,
early stopping patience 5, FP32 only) was cancelled while still pending, after checking "is the lr the same and how
many epochs?": the lr is only the *initial* lr — `ml/trainer.py` uses `CosineAnnealingLR(T_max=epochs)`, so
early-stopped runs stop at different points of the decay (stop at ep 37 of 100 → lr ≈ 2.1e-4; at 76 → ≈ 0.4e-4),
which makes epochs and lr-at-stop per-model variables. The ~+7pp "protocol" effect of item 4 holds for the GAP models (+6.5-7pp); the first finished control shows only +1.8pp for an FC model (`alexnet_3x3_fc` 35.79% with early stopping -> 37.60% at 500 ep), so "a schedule that finishes annealing" is at most part of the story and is not established. The controls therefore
use the Phase 11 protocol itself, so they pair with `alexnet_tv_*`, the mixed-kernel and head/BN runs.

**Models** (`models/alexnet_variants.py:AlexNetAdapted`, default init — no `he_init`): `alexnet_adapted_orig_{fc,gap}`
(kernels 11-5-3-3-3, the missing large-kernel control), `alexnet_adapted_2x2_{fc,gap}` (2×2 at the *same* 8×8 maps —
asymmetric `ZeroPad2d` before conv2-5, unlike the legacy `AlexNet2x2FC/GAP` whose maps shrink to 4×4), and
`alexnet_3x3_gap_bn` (3x3-GAP + BN). `tests/test_registry.py` checks that `kernels=(3,)*5` is `AlexNet3x3FC/GAP` layer
for layer, that every control ends on an 8×8 map, that each hand-written `fuse_map` points at Conv→(BN→)ReLU and
covers all 5 convs, and that FP32 → QAT → INT8 convert runs (the padded 2×2 net included). Side note: in this torch
build a no-BN Conv-ReLU `fuse_map` leaves `qat.Conv2d` + `ReLU` after `prepare_qat` — identical to the existing
`alexnet_3x3_fc`, so that is the pipeline's existing behavior, not something these models changed.

**Experiment** `phase_11_geometry_controls` (`extends: _protocols/no_patience`): 500 ep FP32 / 100 ep QAT (lr 1e-5,
`freeze_bn_epoch` 3, `disable_observer_epoch` 5 — every other Phase 11 run used all 100 QAT epochs with the same
settings) / INT8, no early stopping, seed 42, uniform lr 3e-4, weight decay 5e-4, batch 64. One job per model:

| Job | Model | Job | Model |
|-----|-------|-----|-------|
| 824460 | `alexnet_adapted_orig_fc` | 824464 | `alexnet_adapted_2x2_gap` |
| 824461 | `alexnet_adapted_orig_gap` | 824465 | `alexnet_3x3_gap_bn` |
| 824462 | `alexnet_3x3_fc` | 824466 | `alexnet_bottleneck` |
| 824463 | `alexnet_adapted_2x2_fc` | 824467 | `alexnet_fire` |

`alexnet_3x3_gap` is *not* re-run: it already exists at 500 ep / seed 42 / default init
(`phase_11_mixed_kernel_comparison`, 46.82%). Cost, from Phase 11's measured `avg_epoch_time_s`: ≈ 5–6 h per GAP
model and ≈ 9–10 h per FC model for the 500 FP32 epochs (Bottleneck/Fire not measured), plus QAT; ~65 GPU-h in all.

Read-outs once they land: kernel effect at fixed geometry = adapted_orig vs. 3x3 vs. adapted_2x2 (FC: `alexnet_3x3_fc`;
GAP: existing `alexnet_3x3_gap`); BN effect = `alexnet_3x3_gap` vs. `alexnet_3x3_gap_bn`; block effect =
`alexnet_3x3_gap_bn` vs. bottleneck/fire; geometry(+Dropout) effect at 11-5-3 kernels = `alexnet_adapted_orig_fc` vs.
`alexnet_tv_scratch` (both default init), at 3×3 = `alexnet_3x3_fc` vs. `alexnet_tv_3x3` (+ init: `he_init` there).
Variables that remain, by design or cost: `alexnet_fire` has a stride-1 stem (16×16 maps), so vs. `alexnet_3x3_gap_bn` it
is BN + block + geometry; seed 42 only, so no noise estimate — a second seed needs its own experiment name
(`outputs/<runtime>/<experiment>/<model>/` has no seed in its path).
Code reached PCAD by rsync (its tree was already dirty at 120c5da and had no line the local HEAD lacked), so the runs'
`git_hash` is 120c5da with `git_dirty: true`; commit + push and `git pull` there before relying on the hash.

## Geometry factorial + seed replicates submitted (2026-09-25, PCAD jobs 824667-824684)

First read of `phase_11_geometry_controls` (4 of 8 finished, seed 42, top-1 FP32): kernel effect at the adapted geometry is small
and points the other way from the report's story — FC 3x3 37.60 vs 11-5-3 36.16; GAP 3x3 46.82 vs 11-5-3 45.49 vs 2x2 44.00 — and
the protocol-matched geometry gap is 8.65pp at 11-5-3 (`alexnet_adapted_orig_fc` 36.16 vs `alexnet_tv_scratch` 27.51, geometry + Dropout
only) and 11.10pp at 3x3 (`alexnet_3x3_fc` 37.60 vs `alexnet_tv_3x3` 26.50, + init). Not yet decomposed, and the report's pretrained
baseline row was still on a different regime. This submission closes those:

`models/alexnet_variants.py:AlexNetAdapted` gained `stem_stride`, `stem_padding`, `pool_kernel`, `pool_count`, `dropout`, `pretrained`
(defaults = the adapted layout, so every earlier model is unchanged — `tests/test_registry.py` still asserts `kernels=(3,)*5` is
`AlexNet3x3FC/GAP` layer for layer; and that `alexnet_geo_s4_p3_fc` + Dropout is `AlexNetTV(pretrained=False)`'s exact layout: conv/pool
kernel-stride-padding, Linear shapes, 57.82M params). All default init, 11-5-3-3-3 kernels, FC, no BN unless the name says so.

| Experiment (seed 42) | Job | Model | Isolates (vs) |
|---|---|---|---|
| `phase_11_geometry_factorial` | 824667 | `alexnet_geo_s4_p3_fc` (torchvision geometry, no Dropout; 1x1 map) | Dropout (`alexnet_tv_scratch`) + corner of stride x pooling |
| | 824668 | `alexnet_geo_s2_p3_fc` (stride 2, torchvision pools; 3x3 map) | stride alone |
| | 824669 | `alexnet_geo_s4_p2_fc` (stride 4, adapted pools; 3x3 map) | pooling alone |
| | 824670 | `alexnet_geo_s2_pk3n2_fc` (pool kernel 3, 2 pools; 7x7) | pool kernel |
| | 824671 | `alexnet_geo_s2_pk2n3_fc` (pool kernel 2, 3 pools; 4x4) | pool count |
| | 824672 | `alexnet_geo_s4_p3_gap` | head x geometry (vs s4_p3_fc, adapted fc/gap) |
| | 824673 | `alexnet_geo_s2_p2_drop_fc` (adapted + Dropout 0.5) | Dropout x geometry interaction (`alexnet_adapted_orig_fc`) |
| | 824674 | `alexnet_geo_s4_p3_fc_k3` (3x3 kernels, torchvision geometry) | geometry at 3x3 without the init/Dropout confound (`alexnet_3x3_fc`) |
| | 824675 | `alexnet_tv` (ImageNet-pretrained) | pretraining at the original geometry (`alexnet_tv_scratch`) |
| | 824676 | `alexnet_adapted_orig_fc_pt` (pretrained convs + first 2 Linears, adapted geometry) | pretraining at the adapted geometry (`alexnet_adapted_orig_fc`) |
| `phase_11_geometry_seeds_s43` / `_s44` | 824677-80 / 824681-84 | `alexnet_3x3_fc`, `alexnet_adapted_orig_fc`, `alexnet_3x3_gap`, `alexnet_adapted_orig_gap` | noise bar on the 1-3pp kernel effect (seed 42 = `phase_11_geometry_controls` / `phase_11_mixed_kernel_comparison`) |

**Same protocol as every earlier Phase 11 run, checked, not assumed:** all three files `extends: _protocols/no_patience` and
`tests/test_config.py::test_geometry_experiments_share_the_phase_11_protocol_exactly` asserts that, once name/models/seed are removed, they
equal `phase_11_kernel_size_comparison`'s resolved config (500 ep FP32, `early_stopping_patience: null`, 100 ep QAT, `uniform_hparams`,
stages fp32/qat/int8; lr 3e-4, wd 5e-4, batch 64, label smoothing 0.1, AMP, QAT lr 1e-5 / `freeze_bn_epoch` 3 / `disable_observer_epoch` 5 come
from the unchanged `training.yaml`/`qat.yaml`/`data.yaml`, md5-identical on PCAD). The training/QAT code path for these families did not change
since the reference runs: `git log` for `ml/trainer.py`, `ml/quantization.py`, `ml/data.py`, `scripts/train.py`, the configs since 2026-09-13 shows only
vgg16-specific QAT fixes (a registry-only `qat_disable_observer_epoch`, default unchanged), reporting/resume fixes, and the `qat_wino`-only change.
Seeds 43/44 also re-draw the 90/10 split, so compare within a seed.

Remaining limits: seed 42 only for everything except the four kernel-pair models; "pooling" is split into kernel vs count but not finer;
pretraining at the adapted geometry reuses weights learned at stride 4, so a gain there is "transfer despite mismatch"; a run that stays on the
ln(200) plateau is a result. Cost ≈ 170-200 GPU-h for the 18 runs (FC ~8-15 h each incl. QAT, GAP ~6-8 h, from Phase 11's measured epoch times),
~2 days wall if the 6 shared 4090s are free. Read-outs are listed in each yaml header.
Code reached PCAD by rsync (md5-checked); runs record `git_dirty: true` on a stale HEAD until the tree is reconciled.

## M7 extension — packed `qat_wino` on the other 14 networks (2026-09-24; results synced 2026-09-25)

M7 (above) measured what matching the deploy bitstream costs on 2 models. The same experiments were then run for **14 further
networks** — the 12 other `budget_unico` models plus Phase 11's `alexnet_tv_3x3` and `vgg16` (with `alexnet_fire_bypass_fpga` and
`vgg_style_fpga` that is the 16 networks `scripts/winograd_fpga/dump_layer_configs.py` covers): `wino_f{23,43,63}_pack`, `qat_wino` only,
`pack: true`, `u_w=9, v_w=8, k_dsp=2`, 15 ep at lr 5e-5, seed 42, FP32 reused from `budget_unico` (12 models) / Phase 11 (the other two).
Data: `outputs/pcad/wino_f{23,43,63}_pack/<model>/results/`, 42 summaries (3 variants x 14 models). Top-1 change vs. FP32, in pp (the last
column is `budget_unico`'s original F(4,3) **without** packing, same models, for contrast):

| Model | FP32 | F23 pack | F43 pack | F63 pack | F43 no-pack |
|-------|-----:|---------:|---------:|---------:|------------:|
| alexnet_3x3_fc_fpga | 36.13 | +0.62 | -10.35 | -24.29 | +0.08 |
| alexnet_bottleneck_fpga | 42.78 | -0.03 | -4.69 | -17.88 | +0.21 |
| alexnet_final_bottleneck_residual_fpga | 43.14 | +0.12 | -4.31 | -16.37 | +0.15 |
| alexnet_final_fire_residual_fpga | 44.30 | -0.02 | -3.65 | -13.20 | +0.32 |
| alexnet_fire_fpga | 44.76 | -0.23 | -3.83 | -17.14 | +0.05 |
| alexnet_stacked_fpga | 44.88 | +0.11 | -8.15 | -25.21 | +0.08 |
| googlenet_fpga | 56.68 | +1.32 | -1.61 | -9.96 | +1.36 |
| repvgg_a0_fpga | 55.06 | +0.27 | -2.69 | -10.54 | +0.52 |
| resnet18_fpga | 54.91 | +0.64 | -2.73 | -17.38 | +0.61 |
| vgg13_fpga | 50.60 | -0.53 | -5.55 | -19.93 | -0.28 |
| wrn_16_4_fpga | 56.13 | +0.14 | -6.02 | -29.61 | -0.31 |
| wrn_28_2_fpga | 53.87 | -0.05 | -5.96 | -27.09 | +0.32 |
| alexnet_tv_3x3 (Phase 11) | 26.50 | -1.06 | -16.41 | -26.00 | — |
| vgg16 (Phase 11) | 47.71 | -3.03 | -16.00 | -47.08 | — |

- **F23 ~ free:** median +0.05pp, 11 of 14 within +-1pp, worst `vgg16` -3.03. **F43 packed is a real loss:** median -5.12pp, range -1.61
  (`googlenet_fpga`) to -16.41 (`alexnet_tv_3x3`); **F63 packed is severe:** median -18.90pp, range -9.96 to -47.08, and the two plain
  no-BN-style Phase 11 nets collapse (`alexnet_tv_3x3` 0.50%, `vgg16` 0.63% top-1). Same ordering as the 2-model M7 pair; the spread across
  architectures is large, BN-carrying / residual nets (googlenet, repvgg, resnet18) lose least.
- **The `≠HW` caveat is now quantified:** `budget_unico`'s F(4,3) numbers (no packing) sit within ~+-1.4pp of FP32 for all 12 of its models, but
  matching the deploy bitstream's packing costs 3.9-10.4pp at F43 on those same models — so those `budget_unico` accuracies overstate what the
  bitstream delivers, model by model (`alexnet_3x3_fc_fpga` -10.4, `alexnet_stacked_fpga` -8.2, `wrn_28_2_fpga` -6.3 ... `alexnet_fire_fpga` -3.9).
- Caveats: one seed, val-split accuracy, 15-epoch fine-tune; the ranking of close models is not established.
- `vgg13_fpga` is usable now: the 2026-09-14 memory note had its FP32 at chance (0.50%, plain VGG13 without BN); `budget_unico` now holds
  50.60% FP32 / 50.32% F43 no-pack, i.e. it was retrained.

**Provenance / reproducibility gap (open).** These runs did not come from the `~/dnn_study` checkout: PCAD has a second one,
`~/dnn_study_m7` (main @ `c9750c0`, bridge `~/winograd_bridge_m7`, Winograd-FPGA `fde71c7`, clean). That checkout holds **one commit that is not
on `origin`** — `c9750c0`, "qat_wino-only runs crashed in the run summary and would clobber fp32_top1" (`ml/reporting.py` None-safe gap guards +
`scripts/train.py` recovering `fp32_eval` from the prior summary) — plus **uncommitted** edits to `wino_f{23,43,63}_pack.yaml` (the 14-model
`models:` lists). This repo's `wino_f*_pack.yaml` still list only the original 2 models, and its `scripts/train.py` lacks the fix. The first
attempt (13 jobs, 824390-824402) died on exactly that crash; the 42 jobs 824403-824443 (one per model per variant, via `--model`) completed.
Until that commit and the yaml lists are brought into this repo (or at least pushed from PCAD), these results are not reproducible from `main`.
The two F23 `wrn_*` summaries were missing from the first local sync and were fetched from `~/dnn_study_m7` on 2026-09-25.

## QAT fusion bug + single protocol / full factorial (2026-09-30)

**Bug.** `ml/quantization.py:prepare_qat_model` deep-copied the model and then fused `fuse_root` -- a submodule of the
*original* (`build_qat_from_model` passes `getattr(model, "features")`). Every registry entry with `fuse_root_attr` (43
models: `alexnet_3x3_*`, `adapted_*`, `geo_*`, `stacked*`, `mixed`/`mixed_fc`, `smallkernel*`, `alexnet_tv*`, `vgg16*`,
`vgg_style`, `depthwisesep`, `factorized`, `groupconv`, `dilated_*`) therefore trained QAT with **no fusion at all**:
the activation observer sat on the pre-ReLU conv output and BN stayed unfolded. The `find_fuse_groups` models
(`bottleneck`, `fire`, `fire_bypass`, `residual`, `final_*`, `*_bn` controls) were fused correctly. Present since
ebe7c60 (2026-06-28); `prepare_sim` (`ml/quantization_advanced.py`) had the same pattern. Confirmed by building the QAT
model: `alexnet_3x3_gap` -> `Conv2d, ReLU` (unfused) vs `alexnet_3x3_gap_bn` -> `ConvBnReLU2d`.
Consequences: FP32 is untouched (every 500-ep FP32 checkpoint stays valid), but every INT8/ΔQAT number of those 43
models is suspect. The report's Eixo 5 split "compensation robust vs naive fragile" coincides exactly with the
fused/unfused split, the 8-25pp *conversion* loss of the no-BN GAP models (fig. 13) is unexplained until re-measured,
and vgg16's QAT collapse (BN unfolded while observers froze -- see "Revisit 2") is the likely same root cause.
**Fix:** both functions re-locate `fuse_root` inside the copy by name; `tests/test_quantization.py::
test_fuse_root_models_are_fused_in_the_qat_copy_not_the_input` fails on the old code.
`scripts/phase11/analyze_geometry.py:load` masks qat/int8 to NaN for runs made before the fix whose model needed it.

**Provenance.** `git_dirty` used `git status --porcelain` on the whole tree, so every PCAD run was "dirty" from its own
sibling runs' outputs -- the flag carried no information. Now `ml/runtime.py:code_changes()` limits it to
`ml/ models/ scripts/ configs/ requirements.txt` and the summary also records `git_dirty_files`; `scripts/cluster.py`
refuses to submit with uncommitted code/config (`--allow-dirty` to override; `--smoke`/`--dry-run` exempt). A run made
with this commit is recognisable by `git_dirty_files` in its provenance, and the analysis asserts none of those is dirty.

**Single protocol for the report.** Every classification CNN of the report on `_protocols/no_patience` (500 ep FP32 /
100 ep QAT, no early stopping, uniform lr 3e-4, seed 42). `tests/test_config.py` now globs every `phase_11_*.yaml`
against that protocol and forbids any per-model training override: vgg16's `qat_disable_observer_epoch=None` (its QAT
collapse workaround) is dropped -- the gate reruns vgg16's QAT with the default schedule, and it comes back only if that
still collapses. `configs/runtime/pcad.yaml` gets `persistent_workers: false`: every Phase 11 run used it (edited on PCAD,
never committed; origin said `true`), and the loader must stay identical.

**Full factorial** (`ml/model_registrations.py`, `alexnet_fx_<kernel>_s<stride>_pk<k>n<n>_<head>[_bn][_d][_pt]`):
AlexNetAdapted, kernel {11-5-3-3-3, 3x3, 2x2, 3-2-3-2-3} x stem stride {2, 4} x pool kernel {2, 3} x pool count {2, 3}
x head {FC, GAP} x BN, + Dropout(0.5) on FC, + ImageNet pretraining on the 11-5-3 FC no-BN net = 208 cells. 19 already
exist under other names (`FX_EXISTING`, checked layer for layer by `tests/test_registry.py`), incl. `alexnet_tv_scratch`
and the default-init `alexnet_tv_3x3` from `archive_old_init` (AlexNetTV(pretrained=False) == those cells). `AlexNetAdapted`
now allows a 2x2 stride-4 stem (64 -> 16, like `AlexNetTV(kernel_size=2)`). Fuse maps are derived from the module list
(`_conv_groups`), not hand-counted.

**Reuse search** (all of PCAD `$HOME` + laptop `outputs/`, 316 checkpoints): wandb holds only symlinks (263 live, 53 dead);
`archive_legacy_phases`, `fire_bypass_large_scale` (T_max 1000), `outputs/notebooks` (100 ep), `budget_unico*` (100 ep,
patience 10) are other protocols; only the two AlexNetTV runs above are new factorial reuse. Free side contrasts
(FP32 only): he_init vs default init on 6 identical architectures. `alexnet_se` is out: its SE block can't convert to INT8.

**Waves** (tupi only): 0 = gate (`scripts/pcad/rerun_qat_fused.sh` on `alexnet_3x3_gap`, `alexnet_adapted_orig_gap`,
vgg16 without its observer override: pass if QAT->INT8 conversion loses < 1pp and FP32 fields are unchanged);
1 = the 44 QAT reruns in that script + `phase_11_families.yaml` (17 models); 2 = `phase_11_factorial_core.yaml` (36);
3 = `phase_11_factorial_ext.yaml` (153). Analysis: `python -m scripts.phase11.factor_effects` -> every matched pair per
factor (`results/phase_11_geometry_analysis/factorial_pair_summary.csv`, fig. 17).
Status: committed/pushed (855a492, 6bb41a5); local smoke (`alexnet_fx_k2_s4_pk3n3_gap_bn`, CPU, FP32->fused QAT->INT8) exit 0;
29 tests pass on PCAD. PCAD synced to 6bb41a5 with clean code (backup `~/dnn_study_prepull_backup_20260930.tar.gz`, old
rsynced edits in `git stash`, 269 identical untracked files moved to `~/dnn_study_prepull_untracked_20260930/`).
Gate submitted 2026-09-30: jobs 826909 (`mixed_kernel_comparison/alexnet_3x3_gap`), 826910
(`geometry_controls/alexnet_adapted_orig_gap`), 826911 (`kernel_size_comparison/vgg16`, no observer override).

Waves submitted without waiting for the gate (2026-09-30): the gate only tests QAT/INT8, and FP32 -- most of each new
run's cost -- doesn't depend on it; if the gate fails, those runs only redo QAT (same archive + resubmit). Held for the
gate: the 41 other QAT-only reruns (pure QAT, wasted whole if the fix isn't enough). PCAD's QOS caps a user at **50 jobs
pending + running** (MaxSubmitJobs; the association shows none, the QOS does), so submit-sweep stopped after 50:
`phase_11_families` 17 (826912-826928), `phase_11_factorial_core` 30 (826929-826958). The other 6 core + 153 ext cells go
through `scripts/pcad/feed_queue.sh ~/queue_phase11.txt` (setsid/nohup on the PCAD login node, log `~/queue_phase11.log`):
one submit per free slot, max 48 queued, QOS rejections requeued at the head. After the gate passes, the 41 QAT reruns
are prepended (`scripts/pcad/rerun_qat_fused.sh <experiment>/<model>`, one line each).

Login-node load (measured 2026-09-30): the feeder idles at 3 MB + one 0.02 s `squeue` per 10 min, but every real submit
cost ~9 s CPU / ~640 MB because the dirty-code guard imported `ml` (torch). `scripts/cluster.py` now runs the git check
itself (its `CODE_PATHS` asserted equal to `ml.runtime.CODE_PATHS` in tests/test_config.py), and `feed_queue.sh` re-execs
under `nice -n 19 ionice -c3` and waits 60 s after each submit, so a burst of freed slots is one light submit a minute.

## Fused-QAT gate result + quantized GAP + val-calibrated observers (2026-09-30)

**Gate (jobs 826909-826911, code 9900c8a).** Fusion works: fused `alexnet_3x3_gap` INT8 35.52 -> 44.14, fused
`alexnet_adapted_orig_gap` 31.51 -> 42.39, INT8 ECE ~0.21 -> ~0.05, FP32 fields unchanged. But the conversion still lost
2.4 / 3.1pp (fake-quant 46.57 / 45.54), and `vgg16` without its override collapsed (QAT 0.50%). Neither is fusion.
All three runs were deleted and their QAT redone with the fixes below; the 48 jobs queued behind them were cancelled
before any started (all 3 used pre-fix code).

**Quantized GAP (root cause of the 2-3pp, measured by inference on the gate checkpoints).** Layer by layer, INT8 matches
fake-quant to < 0.04 LSB on average except at `AdaptiveAvgPool2d`: eager fbgemm pooling keeps its *input's* qparams, and
QAT has no observer after the pool, so the fake-quant model never sees that rounding. The GAP input's per-tensor scale is
set by rare peaks (1.13 / 1.74 from maxima ~144 / ~220), so 58% / 66% of channel means fall below one step and round to 0.
Top-1 over the 10k val split:

| | alexnet_3x3_gap | alexnet_adapted_orig_gap |
|---|---|---|
| fake-quant (QAT eval) | 46.55 | 45.24 |
| fake-quant, GAP rounded as in INT8 | 44.70 | 42.51 |
| INT8 as converted | 44.21 | 42.36 |
| INT8, float GAP + fitted requant | 46.45 | 45.13 |

Fix: `ml/quantization.py:requantize_avg_pools` (called by `prepare_qat_model`) wraps every AvgPool2d/AdaptiveAvgPool2d as
DeQuantStub -> pool -> QuantStub, i.e. int32 accumulate + requantize to the pool output's own scale -- what integer-only
inference does (TFLite's MEAN, TensorRT), and QAT now simulates it exactly (Jacob et al. 2018: training must simulate the
inference arithmetic). Check on the gate checkpoints (only the new observer calibrated, on train batches): INT8 46.47 vs
fake-quant 46.63, and 45.17 vs 45.08; fake-quant/INT8 top-1 agreement 0.75 -> 0.91. >= 96 of the 207 queued models have a
real-averaging pool on the INT8 path (GAP heads and several FC heads), so without it the head/BN factors' INT8 effects
would carry this artifact. A pool fed a float tensor after convert (Phase 8's attention-excluded heads) stays a float pool.
`tests/test_quantization.py::test_int8_avg_pool_requantizes_instead_of_inheriting_its_input_scale` fails on the old code.

**Val-calibrated observers.** `FakeQuantize` ignores `eval()`, so every per-epoch validation pass updated the activation
ranges from the val split -- for the first 5 QAT epochs of every model, and all 100 of vgg16's. `ml/trainer.py:
frozen_observers` now disables them in `_validate`/`evaluate` and restores each flag
(`test_validation_does_not_calibrate_qat_observers_on_val_data`).

**vgg16 (not fusion).** Its FP32 last stage is heavy-tailed: `features.37` max ~398, p99.9 ~114, typical values < 1. The
per-tensor 7-bit minmax scale (2.6) zeroes 96% of the nonzero activations, so the freshly calibrated QAT model (no step
taken) is already at 0.49%; SQNR through `classifier.0/3` is ~0 dB. `qat_disable_observer_epoch=None` is back as the one
documented protocol deviation (allowed by name in `tests/test_config.py`); with the val fix, its live observers now
adapt on train data only. Untested fused -- vgg16 is in the new gate. More principled alternatives (learned clipping:
PACT, LSQ) would change every model's QAT and were not taken.

**Rerun scope.** Every Phase 11 QAT now comes from one code version: `scripts/pcad/rerun_qat_fused.sh` gains the 5 runs
that were fused correctly all along but share the GAP/val defects (`alexnet_3x3_gap_bn`, `alexnet_bottleneck`,
`alexnet_fire`, `alexnet_mixed_bn`, `alexnet_mixed_fc_bn`; `alexnet_stacked_fc_nobn` is skipped -- its FP32 is dead at
0.5%). Queue (one `feed_queue.sh`): gate (the 3 runs above) -> `phase_11_families` -> `factorial_core` -> `factorial_ext`
-> the other 46 QAT reruns. Move those 46 up once the gate passes: INT8 within ~0.5pp of fake-quant on both AlexNets,
and vgg16's QAT climbing past ~3% by epoch 15.

Known limitation, unchanged and uniform across runs: the best epoch is selected on the same val split that is reported
(no held-out test split), a small optimistic bias.

## Saving audit before the rerun (2026-09-30)

A 1-epoch local run of `alexnet_fx_k3_s2_pk2n2_gap`, `resnet18tv` and `vgg16` (outputs kept, every artifact reopened)
plus a simulated `rerun_qat_fused.sh` on it found three pre-existing saving defects, all fixed before any job of the
rerun started (the 27 already-submitted jobs were held, the feeder paused, then both resumed on the fixed commit):

- **INT8 artifact unloadable.** `torch.save(int8_model)` pickled the module; quantized convs don't unpickle their
  nn.Module internals (`'ConvReLU2d' object has no attribute '_modules'`), so no `qat_<m>.pth(.gz)` ever written could be
  loaded back -- same machine, same torch. Metrics were unaffected (computed in-process). Now a state_dict (loads with
  `weights_only=True`), rebuilt by `ml.quantization.load_int8_model`; size identical to ~0.05%, so `int8_size_mb` stays
  comparable. Every `qat_*.pth.gz` committed before this is a dead pickle; the QAT best checkpoint (PCAD) regenerates it.
- **QAT rerun wiped the FP32 training record.** The summary is rewritten; its FP32 training fields come from the fit
  history. With no `<m>_resume.pth` (vgg16's PCAD dir) the skip path had none -> `final_train_loss`, `avg_epoch_time_s`,
  hardware/energy averages became None; with one, the no-op resume added its reload time to `total_training_time_s`.
  A run that trains no new FP32 epoch now keeps the prior summary's values (`_FP32_TRAINING_FIELDS`, scripts/train.py).
- **Summary top1/top5 were macro-averaged.** `Trainer.evaluate` used torchmetrics' default `average="macro"` (mean of
  per-class accuracy); the random 90/10 split has 31-78 val images per class, so it drifted up to ~0.2pp from the
  standard (micro) top-1 that `_validate` logs per epoch -- e.g. the gate's `alexnet_3x3_gap` best_val_top1 46.95 vs
  fp32_top1 46.82 on the same weights. Now micro. Every committed summary top1/top5 is macro; for runs with saved logits
  (Phase 11, 2026-09-13+) micro is recomputable from `*_val_logits.npz`, and every Phase 11 rerun re-evaluates FP32.

Tests: `test_saved_int8_artifact_reloads_and_reproduces_the_reported_int8_logits`,
`test_qat_rerun_keeps_the_fp32_training_record[resume|no resume]`, `test_evaluate_reports_the_standard_micro_top1`
(each fails on the old code).

## Gate on d2f4fdc + vgg16_2x2 joins vgg16's observer deviation (2026-10-01)

**Gate (jobs 827240-827242, code d2f4fdc).** Both AlexNets pass, conversion now within 0.4pp of fake-quant (FP32 fields
preserved, provenance clean):

| run | FP32 | QAT | INT8 | INT8 before (unfused, 2026-09) |
|---|---|---|---|---|
| `mixed_kernel_comparison/alexnet_3x3_gap` (827240) | 46.95 | 46.79 | 46.39 | 35.52 |
| `geometry_controls/alexnet_adapted_orig_gap` (827241) | 45.50 | 45.44 | 45.34 | 31.51 |

`vgg16` (827242) with live observers: QAT val 1.96% at epoch 10 -> 27.4% at 15 -> 48.0% at 45 (FP32 47.71).

**Queue audit.** The 255 feeder lines (`~/queue_phase11.txt` + `.done`) are exactly the expected set -- 17 families,
36 core, 153 ext, the 49 `rerun_qat_fused.sh` runs -- no duplicate, each through the right path (rerun vs fresh
submit), every one of the 208 factorial cells in core/ext/`FX_EXISTING`. Only `head_bn_ablation/alexnet_stacked_fc_nobn`
(dead FP32) is out, on purpose. Local structural check of the 246 queued models (fused QAT copy: no free BN; every
module avg pool wrapped; INT8 converts and runs): all pass. `mobilenetv2`/`resnet18tv` drift more between fake-quant
and INT8 on synthetic calibration (logit rel. error 0.21/0.11 vs <= 0.07 elsewhere) -- per module the INT8 output is
within ~0.2 LSB mean of its QAT twin, so it is depth accumulation, not a mis-simulated op. `mobilenetv2`'s functional
GAP (torchvision) is not wrapped by `requantize_avg_pools`; turning it into a module changed neither error. Check both
models' QAT -> INT8 gap when they land.

**vgg16_2x2.** It was in the rerun list on the default schedule (observers frozen at epoch 5), the schedule that
collapsed fused `vgg16`. Freshly calibrated fused QAT, no step taken, on the PCAD FP32 checkpoints (1024 val images,
8 train batches of calibration):

| | FP32 | eval-mode calibration | train-mode calibration (BN batch stats) |
|---|---|---|---|
| vgg16 | 47.6 | 47.8 | 3.4 |
| vgg16_2x2 | 55.0 | 2.0 | 1.3 |

So `vgg16_2x2` starts worse off than `vgg16`, and the pair is the Phase 11 VGG kernel contrast -- different QAT
schedules would add a second variable to its INT8 delta. `vgg16_2x2` now registers `qat_disable_observer_epoch=None`
too (`tests/test_config.py` allows exactly the pair). Correction to "Fused-QAT gate" above: the 0.49% "before any step"
reproduces only with train-mode calibration; the same `vgg16` checkpoint calibrated in eval mode keeps 47.8%, so the
start-of-QAT collapse is a BN batch-stat vs running-stat range mismatch, not the per-tensor scale alone.

**Queue reordered (the 46 QAT reruns first, as planned once the gate passed).** Prepending them to the feeder file
alone would not have moved them: the 50-job QOS cap was full with 47 never-started families/core jobs (~7 h each), so
the reruns (~1 h each) would have entered Slurm only after those, ~5-7 days later. Feeder stopped, the 47 *pending*
jobs cancelled (`scancel --state=PENDING`; none had started, nothing lost), `~/queue_phase11.txt` rewritten (backup
`.bak_20261001`) as 46 reruns -> 17 families -> 36 core -> 153 ext, `vgg16_2x2` last among the reruns so it starts
after vgg16's INT8 lands; feeder restarted (first reruns 827742-827744). Stale-checkpoint audit before that: the
family/factorial run dirs hold no checkpoint yet; each of the 46 reruns has a complete FP32 (`epochs_used` = 500, so
its FP32 "resume" is a no-op) and all three old QAT files, which `rerun_qat_fused.sh` moves to the archive before
submitting (checked on the first two: 0 `qat_*.pth` left in the run dir). Every checkpoint load is strict, so an
architecture drift fails the job loudly; a forward-only drift would show as the rerun's re-evaluated `fp32_top1`
differing from the old summary's `best_val_top1` (same weights, same split -- the gate: 46.95/46.95, 45.50/45.49).

**Protocol audit** (every Phase 11 summary's recorded config; the 3 gate runs from their archived copy, which holds
the FP32 run's config). Identical across all 55 FP32 runs: `training` (500 ep, lr 3e-4, wd 5e-4, label smoothing 0.1,
AMP, no early stopping, no warmup, cosine T_max = epochs), `data` (bs 64, 4 workers, 90/10 split), `qat` (100 ep,
lr 1e-5, BN freeze 3, observer freeze 5), uniform_hparams, stages; all on tupi RTX 4090, same torch/CUDA/cuDNN/Python;
every one at `epochs_used` = 500. Differences, all known: seed 43/44 (the replicates); `persistent_workers` -- 50 runs
recorded **true**, 5 false, so `configs/runtime/pcad.yaml`'s "every Phase 11 run used false" was wrong (comment fixed,
value kept: every run since 2026-09-30, all QAT and all new FP32, uses false). It does not change training: the four
train augmentations draw only from torch's RNG (checked in torchvision 0.20's source), which the loader reseeds every
epoch either way; `worker_init`'s `random.seed` reaches nothing. The FP32 loop (optimizer, scheduler, loss, AMP,
augmentation, sampler, best-epoch selection) is unchanged from 120c5da to HEAD -- only `frozen_observers` (no-op
without FakeQuantize), the micro `evaluate` (reporting) and qat_wino code moved -- so the 206 new FP32 runs train
like the reused ones. Every QAT/INT8 of the report now comes from the current code: QAT starts from `<m>_best.pth`
(the best FP32 epoch, `build_qat`), INT8 converts the best QAT epoch (`fit()` reloads `qat_<m>_best.pth`, observer
scales included). Starting points by design: init per architecture family (`AlexNetAdapted` and so every factorial
cell = PyTorch default; the two AlexNetTV runs mapped into the factorial predate `he_init`, so default too; the
`kernel_size_comparison` trio still mixes inits, see "Geometry confound" item 3), ImageNet weights for the `_pt`
cells, `alexnet_tv`, `mobilenetv2` and `resnet18tv`. Not bitwise reproducible (AMP, no
`torch.use_deterministic_algorithms`), and a QAT rerun starts from a different RNG state than a same-job QAT.

Left as is, not worth changing code under 206 pending jobs: `scripts/train.py`'s `no_new_fp32_epoch` (an FP32
retrain with the same `epochs_used` over an old summary would keep the old timing fields -- no queued run does that),
and `analyze_geometry.load`'s `post_fix = "git_dirty_files" in prov` (true for 855a492..9900c8a too, but no run made
on those commits survives).

## Float logits layer (2026-10-02)

**Audit of the 15 runs done on d2f4fdc/eb2557d** (the 3 gates + 12 reruns). Data and artifacts check out:
- the 90/10 split reproduces exactly on the laptop (same labels in all 15 `*_val_logits.npz`, no train/val
  overlap, per-class sd 6.97 ≈ binomial);
- top-1, ECE and agreement recomputed from the saved logits match the summaries;
- `load_int8_model` on the saved `qat_<m>.pth` reproduces the saved INT8 logits **bit for bit** (5 AlexNets + vgg16).

One measurement artifact: the logits Linear's output got the same 8-bit fake-quant as every activation. So QAT/INT8
logits had only 8–31 distinct values per image (FP32: 95–160, fp16-limited). Consequences:
- 14–31% of val images tied for top-1 (vgg16 7%), and 47–82% tied at the 5th/6th place, so the reported top-5 depended
  on the tie-break (torch.topk vs argpartition: up to 0.5 pp);
- best-epoch selection picks the luckiest tie-break, putting the reported QAT top-1 on average +0.20 pp (14/15
  positive) above the random-tie-break expectation, and INT8 +0.10 pp;
- re-evaluating the saved QAT checkpoints with only the last fake-quant disabled cost nothing on the GAP heads (top-1
  −0.6..+0.0 pp vs the reported first-index number). `alexnet_smallkernel_fc` gained +0.40 top-1 and +0.97 top-5
  (44.55→44.95, 65.19→66.16);
- corr(FP32→INT8 drop, % ties) = −0.61. Part of "FC heads lose more under INT8" was this grid, not weight/activation
  quantization.

**Change.** `ml/quantization.py:keep_logits_float`, applied by `build_qat_from_model`, the path of every real run:
- the logits layer is the last Conv/Linear to run on a 64×64 probe, found in execution order (quantizable ResNet18
  registers its input QuantStub after `fc`);
- it becomes DeQuantStub → float Linear (`qconfig=None`);
- any head other than "Linear followed only by DeQuantStub" fails loudly.

Checked on every registry model: the 91 non-factorial ones plus one per factorial cell type. Every Phase 11 head is a
Linear + DeQuantStub. The only failures predate the change: `*_fpga`/`*_orig` never take the fbgemm INT8 path, and
`alexnet_se`.

INT8 now means every conv and hidden Linear in INT8 and the logits layer in FP32. `int8_size_mb` grows by that layer's
FP32 weights: +0.14 MB (+6%) on the GAP AlexNets, +2.3 MB (+4%) on the FC ones.

Summaries carry `qat_float_logits: true`, and `analyze_geometry.load` treats any qat/int8 without it as superseded.
That replaces the `git_dirty_files` proxy, which every run since 09-30 satisfies.

Not changed: `quantization_advanced.prepare_sim` (Phase 9 mixed-precision PTQ), `qat_wino`, Phase 7's own QAT
builders (backbone only, heads already FP32), and the INT8 results already recorded for phases 6/9.

**Rerun scope.** The 15 runs above, archived with
`ARCHIVE=outputs/pcad/archive_quantized_logits_qat scripts/pcad/rerun_qat_fused.sh RUN...` and resubmitted. Every
queued job (34 QAT reruns, 206 new runs) imports the new code at start; the 48 pending were held before the pull.
Gate first: `mixed_kernel_comparison/alexnet_3x3_gap` (GAP) and `head_bn_ablation/alexnet_smallkernel_fc` (FC). The
other 47 stay held until both pass:
- exit 0, QAT 100/100;
- `fp32_top1` unchanged (46.95 / 45.55);
- the flag present, with clean provenance;
- ~200 distinct logits per image;
- QAT−INT8 < ~0.5 pp;
- `int8_size_mb` ≈ 2.37 / 56.8.

**Update (2026-10-02 ~11:30).** Gate skipped at the user's request: all 49 jobs released at once. The criteria
above became first-landing checks, not a hold.

## VGG factorial + wave-1 priority (2026-10-02)

**Why.** The AlexNet factorial (`alexnet_fx_*`, 208/208 cells queued or reused) had no counterpart in a deeper,
natively-3×3 family, and only one mixed pattern (3-2-3-2-3). `VGG16(kernel_size=2)` can't be extended to it: its 1/0
padding alternation keeps the pooled sizes only under MaxPool 2×2, so with a 3×3 pool the map size would depend
on the kernel pattern.

**Model.** `models/baselines.py:VGGAdapted` is VGG16 (cfgs["D"] + BN) with one knob per factor:
- kernels: 3 or 2 per conv. 2×2 uses a right/bottom ZeroPad2d, as `AlexNetAdapted` does, and pads 0 on a strided stem.
- stem_stride: 1 / 2.
- pool_kernel: 2 / 3. The 3 is MaxPool(3, 2, padding=1), which changes only the overlap, never the map size.
- pool_count: 5 / 4.
- head: GAP / FC / FC + Dropout 0.5.

Final map at 64×64, identical across kernel patterns and pool kernels: s1/n5 2×2, s1/n4 4×4, s2/n5 1×1, s2/n4 2×2.
s1/n5 and s2/n4 both end at 2×2, which separates final map size from where the resolution drops.

The defaults are `VGG16(kernel_size=3)` layer for layer, with the same init. So vgg16's run is the
`vgg_fx_k3_s1_pk2n5_fc_d` cell (`VGG_FX_EXISTING`), and `vgg16_2x2` is not a cell.

BN is fixed, not a factor: a plain VGG16 stays at ln(200) from scratch (jobs 821246/7).

`pretrained=True` loads torchvision's `vgg16_bn`: its convs and BNs, plus the first two Linears when the head is FC.
vgg16_bn's conv bias is folded into BN's running_mean, since our convs are bias-free; the test checks the eval
output matches. The weights were copied into PCAD's `~/.cache/torch/hub/checkpoints`, md5 checked.

**Cells.** `ml/model_registrations.py` registers all 168:
- 6 patterns × 2 strides × 2 pool kernels × 2 pool counts × 3 heads;
- + 24 `_pt` cells (3×3 only).

The 6 patterns are k3, k2, alt32, alt23, early3 (stages 1–3 at 3×3) and early2. The four mixed ones each put 6–7 of
the 13 convs at 3×3, so they compare order and position at roughly equal proportion.

Every cell keeps vgg16's live QAT observers, so the family has one protocol. `tests/test_config.py` now allows the
deviation for `vgg_fx_*` as well.

**Wave 1** (`configs/experiments/phase_11_vgg_factorial.yaml`, 32 runs):
- kernel {k3, k2, alt32} × stride {1, 2} × pool count {5, 4} × head {FC, GAP}, with pool 2×2 and no Dropout: 24 runs;
- {alt23, early3, early2} × VGG's own geometry × {FC, GAP}: 6 runs;
- 2 pretrained cells, each paired with a from-scratch twin: `k3_s1_pk2n5_fc_d_pt` with vgg16, and
  `k3_s1_pk2n5_gap_pt` with `k3_s1_pk2n5_gap`.

Later waves (pool kernel 3, Dropout, the mixed patterns at other geometries) need only a yaml.

**AlexNet wave 1, same four factors.** kernel {orig, k3, k2, mix} × stride {2, 4} × pool {adapted pk2n2, torchvision
pk3n3} × head {FC, GAP}, with no BN, Dropout or pretraining. That is 32 cells: 11 reused, 21 queued. On PCAD those 21
lines were moved to the head of `~/queue_phase11.txt` (backup `.bak_20261002_wave1`).

**Queue on PCAD** (code 40937c0, pulled; `~/queue_phase11.txt` 238 lines, backup `.bak_20261002_vgg`):

| Order | What | Where |
|---|---|---|
| 1 | 49 QAT reruns | already submitted, pending in Slurm |
| 2 | 21 AlexNet wave-1 cells | queue lines 1–21 |
| 3 | 32 VGG wave-1 runs | queue lines 22–53 |
| 4 | 17 families, then the other 168 core/ext cells (BN, crossed pools, Dropout, pretraining) | lines 54 onward |

The reruns stay ahead because the feeder only submits below 48 queued jobs, and the reruns are older. Holding them
wouldn't free slots: held jobs still count toward the QOS limit.

The `vgg16_bn` ImageNet weights were copied into PCAD's shared `~/.cache/torch/hub/checkpoints` (md5 checked), so
the `_pt` jobs don't depend on network access from a compute node.

`num_workers` stays at 4 (`configs/data.yaml`), like every other run (user decision, 2026-10-02). The FP32 stage is
loader-bound (4–40% GPU utilization on the 4090s), but more workers would change each worker's augmentation random
stream relative to the runs already trained.

### QAT rerun coverage audit (2026-10-02)

The question: does every Phase 11 run with a usable FP32 get a QAT/INT8 from the current code? The inputs were
pulled from PCAD: every `phase_11_*` run dir's summary and checkpoint list, `squeue` (job → experiment via its
StdOut path) and the queue file. Result:
- **49 run dirs with a 500-epoch FP32 = 49 pending reruns**, matched one to one. Every pending run has its
  `<m>_best.pth` and no `qat_<m>_best/_resume.pth` left over. A leftover would make `scripts/train.py` skip QAT and
  leave the old INT8 in place.
- None is valid yet. The 15 that had finished on the 09-30 code were archived in wave 2 (float logits) and resubmitted.
- The 20 cells the factorials reuse from earlier runs (19 in `FX_EXISTING`, `vgg16` in `VGG_FX_EXISTING`) are all
  among the 49.
- Every model of every finished Phase 11 yaml is covered, with three explained exceptions:
  - `alexnet_stacked_fc_nobn` (FP32 0.50%, dead) is left out on purpose.
  - `mixed_kernel_comparison.yaml` still lists `alexnet_mixed`/`alexnet_stacked`. Their default-init runs were moved
    aside as `*_preheinit`, and the canonical `he_init` runs are `head_bn_ablation`'s, which are pending.
  - `mixed_kernel_comparison_early2_retry.yaml` files into `mixed_kernel_comparison/alexnet_tv_mixed_early2` via its
    `name:`, which is pending.
- The other run dirs are kept only for provenance and are not cells of anything: `vgg16_original`,
  `alexnet_tv_scratch_dead_heinit`, `alexnet_tv_mixed_early2_dead_seed42`, `alexnet_{mixed,stacked}_preheinit`.
- `families`, `factorial_core/_ext` and `vgg_factorial` have no FP32 yet: they are new runs in the feeder queue.

Check as they land: `qat_float_logits` true, QAT 100/100, `fp32_top1` equal to the archived summary's, QAT−INT8 < ~0.5 pp.

## Literature-standard INT8 + held-out test set (2026-10-03)

**Why.** Every choice in the QAT/INT8 path and in the evaluation should be one a reference already justifies, so
the paper needs a citation, not an explanation. An audit against the references found five deviations:
- activations were **7-bit**: `tq.get_default_qat_qconfig("fbgemm")` sets `reduce_range=True` (0..127), fbgemm's
  workaround for int16 saturation on CPUs without AVX-512 VNNI (checked in torch 2.5.1's source);
- the logits Linear had **FP32 weights**;
- every FC head except vgg16's ran its **Linear-ReLU unfused** (observer before the ReLU) -- an INT8 handicap only
  FC heads had, i.e. a confound in the head factor's INT8 effect; residual blocks ran add then a standalone ReLU;
- the best epoch was **picked and reported on the same 90/10 split**, which also moves with the seed;
- `Trainer.benchmark` timed the **DataLoader** along with the model.

**Change** (QAT/INT8/eval side; the FP32 recipe changed the same day -- next section -- so every Phase 11 run is
retrained, not only the 49 QATs):

| Choice | Reference |
|---|---|
| 8-bit per-tensor affine activations, EMA min/max, BN folding (`INT8_QAT_QCONFIG`, onednn's QAT qconfig) | Jacob et al., CVPR 2018; Krishnamoorthi 2018; affine costs activations nothing over scale quantization (Wu et al. 2020 Sec. 3.3) |
| Per-channel symmetric weights in [-127, 127] | Wu et al. 2020 Sec. 6; LiteRT int8 spec |
| Inputs and weights of every Conv/Linear quantized; logits output FP32 (`_FloatLogits`: INT8 input + weights, stored as int8); pools requantized as the next layer's input | Wu et al. 2020 Sec. 4 ("An operation is quantized by quantizing all of its inputs (e.g. weights and activations). The output of a quantized operation is not quantized to int8 because the operation that follows it may require higher precision") and Sec. 3.3 (Eq. 10: integer GEMM, then a floating-point rescale). Not Sec. 5.1 (Partial Quantization: inputs and computation left in float), cited here until 2026-10-04 |
| Activation fused into its producer (`fuse_sequential_relus`, `FloatFunctional.add_relu`) | Jacob 2018; PyTorch `fuse_modules`; LiteRT fused activations |
| QAT 50 ep (1/10 of FP32's 500 ep), same optimizer, lr 1e-4 (1/100 of FP32's 0.01) cosine to 1e-6 (1/100 of that) (`_protocols/no_patience.yaml`) | Wu et al. 2020 App. A.2 |
| Observers update during the first 4 QAT epochs, BN stats during the first 3 | torchvision `references/classification/train_quantization.py` defaults (`--num-observer-update-epochs 4`, `--num-batch-norm-update-epochs 3`, "number of total epochs to update"); its loop freezes after epoch N, i.e. one epoch later (5 and 4) |
| Best epoch on the 90/10 split, reported on Tiny ImageNet's official val (`create_test_loader`, 10k) | Cawley & Talbot, JMLR 2010 |
| 95% Wilson CI; exact McNemar between two models on the same test images (`ml/reporting.py`) | Wilson 1927; Dietterich, Neural Computation 1998 |
| Seed-to-seed spread as the noise floor (`noise_band`); bootstrap CI of the factorial's matched-pair medians | Bouthillier et al., MLSys 2021; Efron & Tibshirani 1993 |
| Latency: `torch.utils.benchmark` median + IQR on one device-resident batch, fixed threads | PyTorch benchmark utilities |
| 15-bin ECE; mixed-precision training, FP32 eval (unchanged) | Guo et al. 2017; Micikevicius et al. 2018 |

Kernels: `QUANT_ENGINE = "onednn"`, set by `convert_to_int8` (the runtime yamls' `quantized_engine` is gone). INT8 vs
its own fake-quant on this laptop (i7-13650HX, AVX-VNNI but no AVX-512), random weights, SQNR in dB:

| | alexnet_3x3_gap | alexnet_mixed_fc_bn | alexnet_residual | resnet18tv | mobilenetv2 |
|---|---|---|---|---|---|
| onednn, 8-bit (now) | 50.8 | 18.0 | 18.7 | 24.4 | 13.2 |
| fbgemm, 7-bit (until today) | 48.4 | 12.7 | 13.2 | 19.3 | 7.2 |
| fbgemm, 8-bit | 42.7 | 14.2 | 17.3 | 21.6 | 10.0 |

onednn is the most faithful everywhere; fbgemm at 8 bits is worse than at 7, the saturation `reduce_range` exists for.
The low numbers of the BN/residual nets are backend-independent: per-block hooks on resnet18 show the error growing
~3 dB per block from 42 dB at conv1, i.e. 1-LSB rounding differences amplified by random weights, not one broken op.
The real check is on trained weights: summaries now carry `agreement_qat_int8` (test set).

Summaries carry `quant_protocol: "2026-10-03"`, `test_{fp32,qat,int8}_{top1,top5,ece}` and `*_test_logits.npz`;
`analyze_geometry.load` reads the test numbers and treats any other protocol as superseded (shown faded, from its
validation-split numbers). `build_cross_phase_results` labels Phases 1-9's INT8 `legacy` (7-bit activations,
validation split) -- kept, not rerun.

**Rollout** (superseded by the next section). The 49 pending QAT reruns were held (`scontrol hold`) before any could
start on the old code; with the FP32 recipe changing too they were cancelled, never run.

## One recipe, every number from a reference (2026-10-03)

**Why.** Every QAT was being redone anyway, so the FP32 recipe was the only thing still without a citation -- and
the cheapest moment to change it (49 of the program's runs existed). The old recipe had three unreferenced choices:
AdamW lr 3e-4 with wd 5e-4 (decoupled, so it shrank the weights only ~5% over 500 epochs; the usual AdamW value is
0.01-0.05), RRC(0.7-1) + 15° rotation + AutoAugment, and PyTorch's default init in the factorial cells but He init
elsewhere. Decision (user, 2026-10-03): adopt a fully referenced recipe and retrain the whole Phase 11 program on it,
gated by a pilot of the cells most likely to break.

**FP32 recipe** (`configs/experiments/_protocols/no_patience.yaml`; every `phase_11_*.yaml` extends it and
`tests/test_config.py` fails any that drifts):

| Choice | Value | Reference |
|---|---|---|
| Optimizer | SGD, momentum 0.9, L2 weight decay 5e-4 | Krizhevsky et al., NeurIPS 2012 (AlexNet); same values in Simonyan & Zisserman, ICLR 2015 (VGG) |
| LR, batch | 0.01, 128, for every net | Krizhevsky et al. 2012 (VGG's paper: the same 0.01 at batch 256) |
| Schedule | cosine annealing to 0, no warmup | Loshchilov & Hutter, ICLR 2017 (replaces AlexNet's /10 on plateau); warmup is for large-minibatch LR scaling (Goyal et al. 2017), AlexNet used none at this lr/batch |
| Budget | 500 epochs, fixed, no early stopping; best epoch picked on the 90/10 split | Li, Yumer & Ramanan, ICLR 2020 (fixed budget, LR decayed to zero by its end); Cawley & Talbot, JMLR 2010; the 90/10 hold-out of the train set: He et al., CVPR 2016 Sec. 4.2 ("determined on a 45k/5k train/val split") |
| Augmentation | 4-px pad + random crop + horizontal flip, then AutoAugment's ImageNet policy | He et al., CVPR 2016 Sec. 4.2; Cubuk et al., CVPR 2019 |
| Init | He normal (fan_out) on convs, N(0, 0.01) on Linears, in every from-scratch model | He et al., ICCV 2015; Krizhevsky et al. 2012 (the Linear std); torchvision's VGG init |
| Loss | cross-entropy, label smoothing 0.1 | Szegedy et al., CVPR 2016 |
| Precision | mixed-precision training, FP32 evaluation | Micikevicius et al., ICLR 2018 |
| Pretrained cells (`_pt`) | the same recipe -- pretraining is the only variable of their contrast | design choice of the factorial; at lr 0.01 x 500 epochs it is retraining from a pretrained init, where a long from-scratch schedule closes the gap (He, Girshick & Dollar, ICCV 2019, "Rethinking ImageNet Pre-training") |

QAT, INT8 and evaluation: the table of the previous section (QAT = Wu et al. 2020 App. A.2 applied to this recipe:
50 ep, SGD, lr 1e-4 cosine to 1e-6).

**What no single reference fixes.** The 500-epoch budget: no Tiny ImageNet paper prescribes one. 500 epochs = 352k
iterations at batch 128, between CIFAR's long schedules (WRN 200 ep = 78k, DenseNet 300 ep = 234k) and ImageNet's
(ResNet 600k at batch 256, AlexNet ~844k at batch 128); returns diminish with budget (Wightman et al. 2021: 100/300/600
ep -> 78.1/79.8/80.4%, though their A3/A2/A1 procedures differ in more than epochs, e.g. A3 trains at 160 px). Under the old recipe all 51 finished FP32 runs had converged (median best epoch 423, median
gain of the last 100 epochs +0.09 pp, max +0.53). Under the new one this is re-checked per run, not assumed:
`analyze_geometry.convergence()` flags any run whose last-20%-of-epochs gain exceeds the seed noise band
(Bouthillier et al., MLSys 2021). Batch size, `num_workers` (4) and the 90/10 split are protocol constants, identical
for every run.

**Code.** `TrainerConfig.optimizer/momentum/eta_min`, `DataConfig.train_aug` (old values kept as the defaults, so
Phases 1-10 configs are unchanged); `he_init` in every from-scratch class -- an init audit over all 279 Phase 11
models (last Linear std == 0.01) caught `AlexNetTV`'s GAP head still on PyTorch's default, fixed (resnet18tv /
mobilenetv2 are pretrained, their new head on torchvision's default); `load_profile` now resolves `extends:` when
given a file path (it silently trained a path-loaded config on the bare defaults); `dataset_fingerprint` (SHA-256 of
train/ and val/ file list + sizes) in the provenance (Pineau et al., JMLR 2021); `holm()` adjusts the contrast
table's ~35 McNemar p-values (Holm 1979); worker seeding follows PyTorch's reproducibility notes (`seed_worker`).
Not adopted: `torch.use_deterministic_algorithms` -- `AdaptiveAvgPool2d`'s CUDA backward has no deterministic kernel,
and PyTorch promises no bit-exactness across versions/platforms anyway; seeds, `cudnn.deterministic`, pinned
`requirements.txt` and the recorded git/CUDA/GPU/dataset provenance are what the checklists ask for.
`phase_11_reuse_old_init.yaml` and `phase_11_mixed_kernel_comparison_early2_retry.yaml` are deleted: the first
re-QAT'd a superseded checkpoint, the second trained the same slot as its parent yaml (He init is now universal).

**Smoke** (laptop, 2026-10-03, real data, `alexnet_3x3_gap_bn` + `alexnet_3x3_fc`, 2 FP32 + 2 QAT epochs): resolved
config = the table above; lr 0.01 -> 5e-3 -> 0, QAT 1e-4 -> 5.05e-5 -> 1e-6; FP32 -> QAT -> INT8 -> test set end to
end; INT8 artifact 1/4 of FP32 (the logits weights are int8 too); INT8 vs its fake-quant: logit SQNR 40.2 / 38.4 dB,
top-1 agreement 0.980 / 0.979. The disagreements are near-ties of a 2-epoch model (median QAT top-2 margin 0.005 on
them vs 0.14 over all images), so the `agreement_qat_int8 >= 0.99` gate applies to trained runs.

**Superseded.** Every `outputs/pcad/phase_11_*` run (old recipe) moves to `outputs/pcad/archive_adamw_recipe/` on
PCAD before anything is submitted (`scripts/train.py` would otherwise resume from the old checkpoints). The tracked
summaries stay in git until each new run overwrites its own; `analyze_geometry.load` shows none of their numbers
(no `quant_protocol`).

**Program = the full factorial.** `phase_11_vgg_factorial_ext.yaml` (135 cells) completes the VGG factorial, so every
registered cell of both factorials is queued: AlexNet 208 (kernel {11-5-3-3-3, 3x3, 2x2, 3-2-3-2-3} x stem stride
{2, 4} x pool kernel {2, 3} x pool count {2, 3} x head {GAP, FC, FC + Dropout} x BN, + pretraining on the 11-5-3 FC
no-BN cells), VGG 168 (6 kernel patterns x stem stride {1, 2} x pool kernel x pool count {5, 4} x head, +
pretraining on the 3x3 cells), plus the families, mixed-kernel, head/BN and seed-43/44 experiments: 424 runs. Cost,
from the old logs' 35-78 s/epoch on a 4090: ~5-11 h per AlexNet FP32 run, ~3,000-3,500 GPU-h in total, i.e. weeks on
tupi's six 4090s.

**Pilot first** -- the cells most at risk under a fixed, untuned lr 0.01 (every earlier collapse was a no-BN net stuck
at ln(200)): `alexnet_adapted_orig_fc` (11-5-3 no-BN FC) and `alexnet_adapted_2x2_fc` (2x2 no-BN FC) in
`phase_11_geometry_controls`, `vgg16` (deepest; its QAT collapsed with frozen observers under the old QAT) and
`alexnet_tv_scratch` (stride 4, original layout) in `phase_11_kernel_size_comparison`, `alexnet_3x3_gap_bn` (BN
folding on trained weights) in `phase_11_geometry_controls`. Gate: each FP32 leaves ln(200) early (val top-1 far above
the 0.5% chance level by epoch ~20); on completion exit 0, QAT 50/50, `quant_protocol` + test fields present,
`agreement_qat_int8 >= 0.99`. A cell that does not train is fixed globally (lr for all), never per cell. Then the
rest, in the existing priority: the earlier experiments' runs, AlexNet wave 1 (kernel x stride x pool x head), VGG
wave 1, families, the rest of the AlexNet factorial, VGG wave 2. (Superseded the same day by the next section: the
pilot jobs 828503-828507 were cancelled before starting and resubmitted under the new names.)

## Reduced design, descriptive names (2026-10-03)

**Why.** The old runs (AdamW recipe, validation split, seed 42) already sort the factors into two groups:

| Factor (matched pairs) | Effect on FP32 top-1 |
|---|---|
| Head GAP vs FC | +8 to +9 pp |
| Geometry (conv1 stride, pooling) | up to ~12 pp |
| ImageNet pretraining | +7 to +8 pp |
| BN | +3 pp |
| Dropout 0.5 on the FC head | +2 pp |
| **Kernel 11-5-3 / 3x3 / 2x2 at fixed geometry** | **1 to 3 pp** |
| Seed-to-seed noise band (2*sqrt(2)*pooled SD, 4 models x 3 seeds) | 1.0 to 1.8 pp |

The 424-run full factorial spent most of its runs crossing geometry with everything at one seed: precise where the
answer is already plain, unresolved on the study's own question -- the kernel effect sits inside the noise band, and
single-seed differences that small are not evidence (Bouthillier et al., MLSys 2021). Decision (user): keep only the
most informative runs, replicate the main contrast, and name every run by what it is.

**Design** -- 130 runs instead of 424 (~1,000 GPU-h instead of ~3,400), in queue priority:

| # | Experiment | Question | Runs |
|---|---|---|---|
| 1 | `phase_11_kernel_head_bn` (+ `_seed43`, `_seed44`) | Main question: what the kernel restriction costs, and whether head/BN change it. Kernel {11-5-3, 3x3, 2x2, alternating 3-2} x head {GAP, FC} x BN, 64px geometry (8x8 map), 3 seeds | 48 |
| 2 | `phase_11_kernel_geometry` | Does the kernel effect depend on how the net downsamples? 4 kernels x head x the other 3 of stride {2, 4} x pooling {two 2x2, three 3x3}, no BN (incl. torchvision's layout, where a 2x2 kernel at stride 4 reads 25% of the pixels) | 24 |
| 3 | `phase_11_vgg_kernel_head` (+ `_seed43`, `_seed44`) | Does it replicate in a deeper BN net? VGG16 + BN at its own geometry: 6 kernel patterns x head {GAP, FC}, + VGG16's own FC + Dropout head for 3x3 and 2x2, + the 3x3 pretrained twin; 3x3 vs 2x2 x head at seeds 43/44 | 15 + 8 |
| 4 | `phase_11_dropout_pretraining` | The original AlexNet's other levers: FC + Dropout for the 4 kernels at both layouts (the 11-5-3 original-layout cell is AlexNet from scratch, every figure's baseline), + 2 pretrained/scratch pairs | 10 |
| 5 | `phase_11_stacked_narrow` | Compensating a 3x3-only net: depth (two 3x3 per stage x head x BN) and a narrow cheap net (x head) | 6 |
| 6 | `phase_11_families` | The compensation/hybrid tables of the report (+ `alexnet_bottleneck`, `alexnet_fire`) | 19 |

A fractional factorial (Box, Hunter & Hunter 2005; Montgomery 2017) was the alternative, but its resolution-V
estimates assume negligible 3-factor interactions, and the old geometry cells show a strong one (stride x pool kernel x
pool count set the final map together, and the FC head's size with it).

**Names** (`ml/model_registrations.py`, `CELL_FACTORS` holds each cell's factors):
`alexnet_<kernels>_stride<s>_<n>pool<k>x<k>_map<m>_<gap|fc|fcdrop>_<bn|nobn>[_pretrained]` and the same with `vgg16_`.
E.g. `alexnet_k2x2_stride4_3pool3x3_map1_fc_nobn` = 2x2 kernels, conv1 stride 4, three 3x3 max-pools, 1x1 last map, FC
head, no BN. `map` is computed from the built net (it depends on the kernel at stride 4: 3x3 for 11-5-3, 4x4 for the
small kernels with two 2x2 pools). `tests/test_registry.py::test_every_cell_name_says_what_the_net_is` reads every
factor back off all 382 built cells; `test_named_reference_cells_are_the_reference_nets` checks that the cells named
after AlexNet/VGG16 are those nets layer for layer.

**Removed.** The 11 superseded `phase_11_*.yaml` (factorial_core/_ext, geometry_controls/_factorial/_seeds_s43/_s44,
head_bn_ablation, kernel_size_comparison, mixed_kernel_comparison, vgg_factorial/_ext), `FX_EXISTING` /
`VGG_FX_EXISTING` (nothing is reused now), and the `alexnet_tv_mixed_*` / `AlexNetMixed` runs (the `kalt3-2` cells
cover mixed kernels at both layouts). The legacy registry names stay, for the archived runs.

**Pilot** (same 5 risky cells, new names): `alexnet_k11-5-3_stride2_2pool2x2_map8_fc_nobn`,
`alexnet_k2x2_stride2_2pool2x2_map8_fc_nobn`, `alexnet_k3x3_stride2_2pool2x2_map8_gap_bn` (kernel_head_bn),
`vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn` (= VGG16, vgg_kernel_head), `alexnet_k11-5-3_stride4_3pool3x3_map1_fcdrop_nobn`
(= AlexNet from scratch, dropout_pretraining). Gate as in the previous section.

**Analysis.** `factor_effects.factorial_cells` / matched pairs now read `CELL_FACTORS` (both families + stacked, with a
3x3 -> stacked contrast) and `analyze_geometry.BASE_KEY` is the new AlexNet-from-scratch name. The other figure code
(`plot_kernel_comparison.py`, `analyze_geometry`'s geometry/kernel-seed tables, `factor_effects`' hand-picked contrast
table) still keys on the legacy names: it gets rebuilt on these blocks when the first results land (core: mean and
95% CI over the 3 seeds, kernel x head x BN).

## Old runs archived, runs kept apart on every machine (2026-10-03)

The superseded Phase 11 runs sat at the live paths in git (`outputs/pcad/phase_11_*`, 319 tracked files) while PCAD had
already moved them to `outputs/pcad/archive_adamw_recipe/`, and their curated tables/figures sat where the new results
will land (`results/phase_11_*`, `results/figures_generated/phase_11_kernel_size_comparison/`). Now:
- git and PCAD hold them at the same path, `outputs/pcad/archive_adamw_recipe/` (+ `results/archive_adamw_recipe/`), each
  with a `SUPERSEDED.md`; PCAD's older superseded states (`archive_old_init`, `archive_unfused_qat`,
  `archive_quantized_logits_qat`) got one too. `.gitignore`'s one-off vgg16 QAT exception follows the move, plus a
  pattern for every VGG16 FC-head cell of the new design (~134M params, INT8 artifact over GitHub's 100 MB limit).
- `build_runs_index` indexes runs under a `SUPERSEDED.md` with `superseded=<archive>` and no phase (54 rows);
  `build_cross_phase_results` reads one label per live `results/phase_11_*` dir (seed replicates share model names) and
  never the archive; both CSVs regenerated -- the only rows that left are the 5 old Phase 11 ones.
- `scripts/train.py:refuse_foreign_run_dir`: a run dir whose `resolved_config.json` differs in data/training/qat/seed
  (the dataset path excepted) stops the run instead of being resumed or skipped -- a restored archive, an rsync to the
  wrong path or a reused experiment name can no longer mix two protocols. A Slurm requeue passes.

## Minimal design, symmetric 2x2 padding, scratch twins (2026-10-04)

**Why.** A review of the 2026-10-03 program found: two pretrained nets with no from-scratch twin, a 2x2 padding artifact
the size of the kernel effect, text that cited the wrong sections, and a 130-run program that would take weeks. User:
the fewest runs that still draw every figure, no seed replicates beyond a noise floor, every pretrained net with a
from-scratch twin, the 21 families kept, and the queue held until the pilot's full gate passes.

**2x2 padding: fixed, not only cited.** Every stride-1 2x2 conv of `AlexNetAdapted`/`VGGAdapted` was padded right/bottom
only (`ZeroPad2d((0, 1, 0, 1))`). Wu et al., NeurIPS 2019 ("Convolution with even-sized kernels and symmetric padding")
show that this shifts the map 0.5 px per layer toward one corner ("the shift problem", Sec. 3.1-3.2: the post-ReLU
values drift to the top-left), that such C2 nets fall behind C3 and saturate as depth grows (Sec. 4.1), and that their
symmetric padding (C2sp: four channel groups padded at four different corners, Sec. 3.3) gains 2.5% on ImageNet over C2
(Sec. 6). The kernel effects measured here are 1-3 pp, so the old padding would bias the 2x2 cost upward by about the
effect itself. `models/baselines.py:SymmetricPad2d` is C2sp; it replaces the old pad 1:1 (same Sequential indices, so the
fuse maps hold), stems with stride > 1 stay unpadded, and the conv after it is still a plain 2x2 conv (Winograd
F(m, 2)-eligible). VGG's stride-1 RGB stem has 3 channels, so it fills 3 of the 4 corners (Wu's exact symmetry needs
channels % 4 == 0). INT8: the cat of slices of one quantized tensor keeps its qparams, so no observer is needed;
converted and run in tests. The archived 2x2 runs used the old pad (their git_hash points at that code).

**Scratch twins.** `mobilenetv2` / `resnet18tv` load ImageNet weights by default (their Phase 1 meaning), so
`phase_11_families` trained them pretrained next to from-scratch compensation nets. `mobilenetv2_scratch` /
`resnet18tv_scratch` join the families. `tests/test_config.py::test_every_pretrained_phase_11_net_has_a_scratch_twin` reads
`pretrained` off each Phase 11 ctor (explicit or class default) and requires the twin.

**Same maps for every kernel.** At conv1 stride 4 with two 2x2 pools, torchvision's 11x11 pad-2 conv1 ended the 11-5-3
cells on 3x3 against 4x4 for the small kernels. That pad is now kept only in torchvision's own layout (three 3x3 pools,
1x1 map for every kernel); elsewhere k // 2, so `alexnet_k11-5-3_stride4_2pool2x2_map3_*` became `..._map4_*`.

**Design: 76 runs (seed 42 + a 4-run noise floor), ~560 GPU-h** (from the old logs: ~7 h per AlexNet / family run,
~10 h per VGG16 run on a 4090), down from 130 / ~1,000:

| Block | Runs | Figures |
|---|---|---|
| `phase_11_kernel_head_bn`: kernel {11-5-3, 3x3, 2x2, alt 3-2} x head {GAP, FC} x BN, 64px layout | 16 | 03, 05, 10, 13, 14 |
| `phase_11_kernel_geometry`: torchvision's layout x 4 kernels x {GAP, FC} (8) + stride 4 / two 2x2 pools and stride 2 / three 3x3 pools, FC, x 4 kernels (8) | 16 | 02, 06, 07, 13 |
| `phase_11_vgg_kernel_head`: {3x3, 2x2, alt 3-2} x {GAP, FC} (6) + FC + Dropout 3x3 (= VGG16) and 2x2 (2) + pretrained 3x3 (1) | 9 | 04, 05, 09 |
| `phase_11_dropout_pretraining`: FC + Dropout x 4 kernels at torchvision's layout (incl. AlexNet from scratch) + 2 pretrained | 6 | 02, 08, 09 |
| `phase_11_stacked`: {3x3, 2x2} two convs per stage x {GAP, FC}, BN | 4 | 11, 14 |
| `phase_11_families`: the 19 + 2 scratch twins | 21 | 01, 11 |
| `phase_11_kernel_head_bn_seed43/44`: k3x3 and k2x2, GAP + BN (`noise_band`) | 4 | noise floor |

Out: the VGG kalt2-3 / k3x3then2x2 / k2x2then3x3 cells (no factor figure reads them), FC + Dropout at 64px (Dropout's
effect is measured where the real AlexNet has it), the no-BN stacked nets, the "narrow" AlexNetSmallKernel cells (width,
stride and map change together -- not a one-factor contrast), and the seed replicates of the rest (40 runs); the
`_vgg_kernel_head_seed43/44` files are deleted, `phase_11_stacked_narrow` is now `phase_11_stacked`. New:
`alexnet_k2x2stacked_*` (`AlexNetStacked(kernel_size=2)`), the 2x2 counterpart of the depth compensation.

**Rollout.** The k2x2 pilot cell (828513) was held until PCAD pulled this commit; the other four pilot cells contain no
2x2 conv, so they are unchanged. Queue file regenerated (71 lines, without the 5 pilot cells); the feeder stays stopped
until the pilot's full gate passes.

## Figures by factor; the pilot overflows to beagle (2026-10-04)

**Pilot.** tupi's six 4090s had ~23 day-long jobs of other users queued ahead (~3-4 days of wait), so the four AlexNet
pilot cells became eligible for `tupi,beagle` (beagle: 2x GTX 1080 Ti, node-exclusive, 32 GB -- memory lowered to 16/15
GB); 828512 (`alexnet_k11-5-3_stride2_2pool2x2_map8_fc_nobn`) started there at once, at ~116 s/epoch, off ln(200) from
epoch 1. 828513 (the 2x2 cell) was held until PCAD pulled the C2sp commit. Caveat: beagle's GPU/CPU latencies and its INT8
kernel path (Sandy Bridge, no AVX2) differ from tupi's; if INT8 crawls or crashes there, only the INT8 stage is redone on
tupi (scripts/train.py skips stages whose best checkpoint exists). The vgg16 pilot stays on tupi.

**Figures.** Every Phase 11 figure now picks its runs by their factors (`CELL_FACTORS`, via `design_figures.frame`),
never by name, so the 76-run design needs no per-name tables:
- new (`scripts/phase11/design_figures.py`): 16 accuracy x cost Pareto front, 17 kernel x head x BN grid (delta vs
  11-5-3-3-3, inside the seed band greyed out), 18 kernel x geometry, 19 INT8 robustness (+ the worst layer-input
  max/p99.9 from `*_layer_stats.json`), 20 measured batch-1 latency vs MACs, 21 training curves. Previewed first on the
  superseded runs (`--archive`, `results/archive_adamw_recipe/figures_generated/phase_11_preview/`; validation split,
  pre-fix INT8, approximate twins marked †) and approved;
- adapted: 02-04 (kernel per layout/head/BN), 05-11 (one figure per factor over every matched pair of the design, plus
  `FAMILY_CONTRASTS`: Bottleneck/Fire vs 3x3 + BN + GAP, MobileNetV2/ResNet-18 scratch vs pretrained), 14 (every
  matched pair per contrast, now with the seed noise band), 15 and the tables (`main_grid.csv`, `kernel_geometry.csv`);
- retired: 01 (16 plots every run against MACs and size), 12 (~100 rows with every matched pair; 14 summarizes them), 13
  (17/18 show the interactions with more data).
`tests/test_phase11_figures.py` renders all 18 on a fake run tree -- the whole design and a half-finished queue (the 5
pilot cells) -- since no real run had finished yet.

## History moved from CLAUDE.md (2026-10-05)

Verbatim from CLAUDE.md's Model Inventory row for Phase 11 (everything before the 2026-10-04 minimal design); the dated sections above have the details.

History: `AlexNetTV(kernel_size=None\|3\|2)` (original 11×11/5×5/3×3, 3×3, 2×2, no BN) and `VGG16(kernel_size=3\|2)` (torchvision cfgs["D"] + BatchNorm -- plain (no-BN) VGG16 from scratch measured stuck at ln(200) loss for 22 epochs on PCAD, 2026-09-13; kernel_size=3 is VGG's own native design) — all 5 trained from scratch, no early stopping, via `configs/experiments/phase_11_kernel_size_comparison.yaml` (`_protocols/no_patience.yaml`: 500ep FP32 / 50ep QAT since 2026-10-03, Wu et al. 2020). **Mixed-kernel + head/BN ablation extension:** `alexnet_variants.py`'s `AlexNetMixed`/`AlexNetStacked`/`AlexNetSmallKernel` and `AlexNetTV(kernel_size="mixed_alt"\|"mixed_early2"\|"mixed_early3")` cross kernel pattern × GAP/FC head × BN on/off, via `phase_11_mixed_kernel_comparison.yaml` and `phase_11_head_bn_ablation.yaml`. `models/baselines.py:he_init` (2026-09-17, `docs/logs/PHASE11_LOG.md`) fixes a from-scratch dead-ReLU plateau that killed several no-BN cells (`alexnet_stacked_fc_nobn` still doesn't train — BN turns out load-bearing for that depth+FC-head combination, treated as a finding not a bug). `alexnet_tv_mixed_early2_gap` (added 2026-09-19) closes the last FC/GAP pairing gap, 26.51% FP32 on PCAD — not yet folded into the curated `results/phase_11_head_bn_ablation_final_comparison.csv`. `scripts/phase11/{plot_kernel_comparison,factor_effects,analyze_geometry,design_figures}.py` render the 18 figures (see the scripts/phase11/ entry above). **Geometry caveat (verified 2026-09-24, `docs/logs/PHASE11_LOG.md` "Geometry confound"):** `AlexNetTV` keeps the 224×224 stride/pool layout, which collapses to a 1×1 map before the classifier at 64×64 (and with `kernel_size=3/2` its stride-4 conv1 reads only 56%/25% of the pixels), while `AlexNet3x3FC/GAP`/`Mixed`/`Stacked` and the compensation family use an adapted layout (stem s2, 2× MaxPool(2), no Dropout) — ~17pp better at matched protocol (`alexnet_mixed` 45.28% vs. `alexnet_tv_mixed_alt_gap` 27.90%). So "AlexNet compacto" vs. `alexnet_tv_*`, and `AlexNet3x3-FC` vs. the pretrained AlexNetTV baseline, are NOT pure kernel-size comparisons (Dropout, init, protocol, pretraining also differ); never describe either as a single-variable kernel ablation, and don't extrapolate the 64×64 absolutes to 224×224 AlexNets. Controls for that confound: `models/alexnet_variants.py:AlexNetAdapted` (`alexnet_adapted_orig_{fc,gap}` = 11-5-3-3-3 kernels, `alexnet_adapted_2x2_{fc,gap}` = 2×2 at the same 8×8 maps, `alexnet_3x3_gap_bn` = BN control) run via `phase_11_geometry_controls.yaml` (`_protocols/no_patience.yaml`: 500 ep / QAT 50 ep, seed 42, one job per model); submitted 2026-09-24, results pending (`docs/logs/PHASE11_LOG.md`). **Geometry factorial + seed replicates** (2026-09-25): `AlexNetAdapted` also takes `stem_stride`/`stem_padding`/`pool_kernel`/`pool_count`/`dropout`/`pretrained` (defaults = adapted layout), registered as `alexnet_geo_<stem>_<pool>_<head>[_drop|_k3]` + `alexnet_adapted_orig_fc_pt`; run via `phase_11_geometry_factorial.yaml` (10 models incl. pretrained `alexnet_tv`) and `phase_11_geometry_seeds_s43/_s44.yaml` (4 kernel-pair models each) — same `_protocols/no_patience.yaml` protocol, enforced by `tests/test_config.py`; results in. **QAT fusion bug + full factorial (2026-09-30, `docs/logs/PHASE11_LOG.md`):** `prepare_qat_model` fused the caller's original instead of the QAT copy, so every `fuse_root_attr` model (43, incl. all hand-mapped AlexNets/VGGs) trained QAT unfused -- their INT8/ΔQAT are invalid until rerun (`scripts/pcad/rerun_qat_fused.sh`, old artifacts -> `outputs/pcad/archive_unfused_qat/`); FP32 is unaffected. **Float logits layer (2026-10-02):** the 15 QATs finished on the 09-30 code are rerun too (old artifacts -> `outputs/pcad/archive_quantized_logits_qat/`, gate `alexnet_3x3_gap` + `alexnet_smallkernel_fc` first), so every Phase 11 QAT/INT8 comes from one commit. The report's classification CNNs all move to this protocol: `phase_11_families.yaml` + the 208-cell `alexnet_fx_*` AlexNetAdapted factorial (`phase_11_factorial_core/_ext.yaml`, 19 cells reused via `FX_EXISTING`); analysis `scripts/phase11/factor_effects.py`. **VGG factorial (2026-10-02):** `models/baselines.py:VGGAdapted` (VGG16+BN, per-conv 3/2 kernels with ZeroPad'd 2×2, stem stride 1/2, MaxPool 2 or 3/s2/p1, 5/4 pools, GAP/FC/FC+Dropout, `vgg16_bn` pretraining) -> 168 `vgg_fx_*` cells registered, `vgg16` reused as `vgg_fx_k3_s1_pk2n5_fc_d`; wave 1 = `phase_11_vgg_factorial.yaml` (32 runs). PCAD queue order: QAT reruns -> the 21 AlexNet wave-1 cells (kernel×stride×pool×head, no BN/Dropout/pt) -> VGG wave 1 -> families -> rest of core/ext (`docs/logs/PHASE11_LOG.md` "VGG factorial"). QAT coverage audited 2026-10-02: the 49 Phase 11 runs with a usable FP32 = the 49 pending reruns, incl. all 20 reused factorial cells; only `alexnet_stacked_fc_nobn` (dead FP32) is left out on purpose

## INT8 accuracy engine (2026-10-06)

**Why.** The first three runs on the 2026-10-03 protocol (all three INT8-evaluated on beagle) lost 1-5pp from QAT to
INT8 (test: 38.23 -> 33.04, 35.80 -> 33.48, 34.57 -> 33.69), with only 64-74% of top-1 predictions surviving the
conversion. On 1000 test images, re-running each INT8 layer in float64 from the kernel's own int8 input and weights
(exact int32 arithmetic) showed the cause: only the first conv was wrong (7-13% of its outputs off by >=2 steps, up to
137), and the exactly-computed network matched fake-quant (97-98% agreement; top-1 37.3/33.8/35.2 vs fake-quant
37.0/33.7/35.2 vs onednn 32.5/31.4/32.2). Emulating int16 saturation of the (c0, c1) input-channel pair sums reproduces
onednn's first-conv output to within one rounding step: without VNNI it computes u8*s8 with VPMADDUBSW, which sums products in pairs into
int16 and saturates (255*127*2 = 64770 > 32767). Only the first conv is hit because its input carries the image's zero
point (~114), so most u8 values are large; post-ReLU inputs have zero point 0. The old 7-bit `reduce_range` (until
2026-10-03) could not saturate (127*127*2 = 32258).

| Reference | What it says |
|---|---|
| oneDNN Developer Guide, "Nuances of int8 Computations" | AVX2/AVX-512 without DL Boost: VPMADDUBSW "accumulates the result into s16 with potential saturation"; "it is the user's responsibility to choose the quantization parameters so that no overflow/saturation occurs" (u7 activations or s7 weights); with VNNI, VPDPBUSD is exact |
| pytorch/pytorch#103646 (2023) + PR #103653 | onednn's 8-bit default "silently causes numeric saturation on CPUs without avx512_vnni"; PyTorch's docs now recommend onednn with 8-bit activations only on VNNI CPUs, `reduce_range` otherwise |
| Jacob et al., CVPR 2018 | integer-only inference accumulates u8*s8 products in int32 -- the arithmetic the reported INT8 number must come from |

PCAD CPUs (the INT8 stage runs on the training node's CPU): beagle Xeon E5-2650 (AVX only), tupi1/tupi2 Xeon E5-2620 v4
(AVX2, no VNNI), tupi3-6 i9-14900KF (AVX-VNNI, not verified with oneDNN 3.5.3); the laptop's i7-7700K (AVX2, no VNNI)
saturates identically. So INT8 accuracy depended on the node a job landed on. *Correction (same day, "Machine
independence" below): the laptop is an i7-13650HX with AVX-VNNI, where onednn is exact -- only fbgemm saturates there.*

**Change.** INT8 accuracy comes from QNNPACK (`ml/quantization.py:ACCURACY_ENGINE`, `convert_to_int8`'s default): on the
laptop it has no output off by >=2 steps and agrees 99% with the exact emulation (top-1 40.8 vs 40.6 on 500 test
images; onednn 35.0). INT8 latency stays on onednn (`QUANT_ENGINE`): saturation changes its results, not its speed.
Kept 8-bit activations rather than PyTorch's `reduce_range` fix, which would retrain every QAT and leave the
literature's 8-bit definition. `scripts/train.py` saves the INT8 state_dict (engine-independent), evaluates it rebuilt on
QNNPACK, benchmarks it rebuilt on onednn, and deletes the run's INT8 logits before re-evaluating, so a summary never reads
an older evaluation's. `tests/test_quantization.py` checks the default engine against an exact int32 reference on the
saturating case (it fails with onednn on this laptop). `QUANT_PROTOCOL` = "2026-10-06": analysis drops every run still
on "2026-10-03" until it is re-evaluated.

**Reruns.** Training is unaffected (FP32 and QAT train on GPU with fake-quant), so nothing is retrained: a 2026-10-03
run is resubmitted, FP32 and QAT "resume" at 500/500 and 50/50 (no-ops), and only the evaluations run again. Their old
summaries and INT8 logits go to `outputs/pcad/archive_saturated_int8/<experiment>/<model>/` first. A probe of that path
found two records a rerun overwrote, both fixed with a test that resubmits a finished run and allows only the
re-measured latencies to change (`test_resubmitting_a_finished_run_reevaluates_it_without_touching_its_training_record`):
`qat_total_training_time_s` (the no-op resume added its reload time; now kept like the FP32 training fields) and the
provenance of the job that trained the checkpoints (`resolved_config.json` was rewritten; its `provenance_history` now
keeps every earlier job). The four reruns go to `--slurm beagle`, where they trained (GTX 1080 Ti, whole node), so
their FP32/QAT evaluations must reproduce the archived numbers exactly -- the check that reuse changed nothing.

## C2sp receptive field, latency machine, best-epoch tie-break (2026-10-06)

**C2sp receptive field.** Measured from input gradients (all-ones weights, max-pool read as average pool): each C2sp
channel group sees a different 2x2 of the 3x3 window, so a stack of stride-1 C2sp 2x2 convs has exactly a 3x3 stack's
receptive field (5x5 after 2 layers, 9x9 after 4, 17x17 after 8; one-sided 2x2: 3x3, 5x5, 9x9). Its gradient-weighted
width is 0.87x the 3x3 stack's at every depth (one-sided: 0.61x). The centre unit of the 8x8 map sees 60-64 px of the
64 px image in every AlexNet kernel cell (effective width k11-5-3 12.6 px, k3x3 11.8, kalt3-2 11.3, k2x2 10.4). So in
this design the kernel factor 2x2 vs 3x3 measures weights/MACs per filter at a near-equal receptive field, not a
smaller one: a 2x2 cost here must not be explained by receptive field, and the archived one-sided 2x2 runs had a
different receptive field, so they are not a reference for the current cells. VGG's 3-channel stride-1 stem (three
corners filled) shifts the map -1/6 px at that layer only (the next layers have >= 64 channels, symmetric) -- negligible
next to the old 0.5 px per layer.

**Latency machine.** Figure 20 grouped machines by `gpu_name`, but tupi1/2 (Xeon E5-2620 v4, 16 threads) and tupi3-6
(i9-14900KF) share "RTX 4090": on the archived runs the same model's INT8 batch-1 latency was 1.6-3.7x higher on tupi1/2
(old benchmark, DataLoader included, so the size is indicative), and ~27% of those runs landed there. Provenance now
records `cpu_model`, `analyze_geometry.load` builds `machine` = GPU / CPU model (hostname before this), and figure 20
fills only the most common machine. Summaries now carry `benchmark_num_threads` and each benchmark's
`*_latency_iqr_ms_per_image` (Trainer.benchmark measured them; train.py dropped them), and `configs/slurm/beagle.yaml`
asks 8 CPUs like tupi_4090 (it asked 4, so a `--slurm beagle` re-evaluation timed CPU latency at half the threads of
the overflowed tupi job that trained the run). 829186/829187/829211 were submitted with 4 and keep it, recorded.

**Best-epoch tie-break.** A summary's `epochs` is the 0-based BEST epoch (`ml/reporting.py`), but
`scripts/build_cross_phase_results.py:_epochs` and `report/generate_figures.py` used it as "completed epochs" to pick
between duplicate summaries; both now prefer `epochs_used`. No duplicate existed in either tree, so no table changed.

## Machine independence and reproducibility (2026-10-06)

Runs will land on tupi1-6 (RTX 4090; Xeon E5-2620 v4 on tupi1/2, i9-14900KF on tupi3-6), beagle (GTX 1080 Ti, Xeon
E5-2650 without AVX2) and the laptop (RTX 4060 Laptop, i7-13650HX), maybe others. A cell's numbers must be a function of
its protocol, not of the node. What can depend on the machine, and how each is handled:

| Factor | Handling | Reference |
|---|---|---|
| Data | Same files (dataset sha256 `932bfab8...` on the laptop and PCAD), split from `torch.Generator(seed)`, workers seeded (`ml/data.py:_seed_worker`), `persistent_workers: false` on both runtimes | Pineau et al., JMLR 2021 (record the dataset version) |
| Software | torch 2.5.1+cu121 / torchvision / numpy / pillow identical on both; torch loads cuDNN 9.1.0 on both. Provenance records torch/torchvision/CUDA/cuDNN/Python, git hash + dirty files, host, GPU, CPU model, Slurm job | PyTorch "Reproducibility" notes (no guarantee across releases or platforms) |
| FP32 arithmetic | `scripts/train.py` turns TF32 off (`cudnn.allow_tf32`, `cuda.matmul.allow_tf32`): cuDNN defaults to TF32 convs (10-bit mantissa) on Ampere and later (4090, 4060), never on Pascal (1080 Ti), so evaluation and QAT (no AMP) would depend on the GPU. AMP's FP16 training is unaffected. Phase 6 profiling keeps TF32 (it measures it). | PyTorch "CUDA semantics: TF32 on Ampere"; Micikevicius et al., ICLR 2018 (FP16 compute, FP32 evaluation) |
| INT8 arithmetic | Accuracy on qnnpack (exact int32 on any CPU). `ml/quantization.py:int8_kernel_error_steps` measures the kernels against exact int32 on the saturating first-conv case, a depthwise conv and an FC-head Linear; `scripts/train.py` runs it before every INT8 evaluation, records `int8_kernel_max_err_steps`, and stops the run if > 1 step (resubmit elsewhere; FP32/QAT are saved) | Jacob et al., CVPR 2018; oneDNN dev guide, "Nuances of int8 computations" |
| Latency | Machine-bound by nature: compared only within a machine (figure 20, GPU + CPU model), with `benchmark_num_threads` and IQRs recorded | Hoefler & Belli, SC 2015 (report the system and threads; medians with nonparametric spread) |
| GPU nondeterminism | GAP cells: none -- same seed, same machine retrains bit for bit (measured below). FC cells: `AdaptiveAvgPool2d`'s CUDA backward uses atomics whenever the map does not divide the 6x6/7x7 output (8->6, 1->6, 4->6, VGG 2->7: all FC heads here), and PyTorch has no deterministic kernel for it, so a same-seed retrain differs. Not chased bit by bit; the seed noise floor (`_seed43/_seed44`, on whatever nodes they land) measures seed + machine + nondeterminism together | PyTorch "Reproducibility"; Zhuang et al., MLSys 2022 (tooling variance comparable to seed variance); Bouthillier et al., MLSys 2021 |
| Where runs live | `analyze_geometry.RUNS` reads `outputs/pcad` and `outputs/local`; the same cell in both stops the analysis (one canonical result per cell) | -- |

**Measured: one beagle run re-evaluated on the laptop** (`alexnet_k11-5-3_stride2_2pool2x2_map8_fc_nobn`, same
checkpoints, official test set, batch 128; scratchpad script, not part of the pipeline):

| Evaluation | Laptop top-1 | Beagle top-1 | Prediction agreement with beagle | Max logit difference |
|---|---|---|---|---|
| FP32, RTX 4060, TF32 off | 38.27 | 38.27 (FP32) | 99.97% | 0.004 (the fp16 the logits are stored in) |
| FP32, RTX 4060, TF32 on | 38.26 | 38.27 (FP32) | 99.98% | 0.012 |
| INT8, qnnpack | 38.15 | 38.23 (QAT fake-quant) | 97.5% | 0.36 |
| INT8, onednn (exact here: AVX-VNNI) | 38.17 | 38.23 (QAT fake-quant) | 97.5% | 0.37 |
| INT8, qnnpack | 38.15 | 33.06 (onednn, saturated) | 64.0% | 8.9 |

The FP32 number reproduces across GPU generations; TF32 triples the logit deviation but moved top-1 by 0.01 pp on this
run, so turning it off costs little and closes the one GPU-dependent arithmetic path. Exact INT8 reproduces QAT to 97.5%
of predictions, as the float64 emulation did; qnnpack and onednn agree on 99.1% of predictions, which is the rounding gap
between two exact engines and why the engine is part of the protocol (`QUANT_PROTOCOL`). The beagle re-evaluation
(829186) should match the laptop's qnnpack logits up to the float logits layer's CPU rounding; check when it finishes.

**The laptop's CPU, corrected.** The section above and the 2026-10-06 test docstrings called the laptop an i7-7700K
without VNNI. `/proc/cpuinfo` says i7-13650HX with AVX-VNNI (as the 2026-10-03 entry has it). There
`int8_kernel_error_steps` gives qnnpack 1, onednn 0, fbgemm 103 (98% of outputs >= 2 steps off): only fbgemm saturates.
The "onednn 35.0 on 500 images" above does not reproduce on this machine (the full test set gives onednn 38.17).
Saturation needs a CPU without VNNI (beagle, tupi1/2); tupi3-6 (AVX-VNNI, same generation as the laptop) are probably
exact under onednn too, so the archived runs' INT8 depended on tupi1/2 vs tupi3-6 as well as beagle.

**Measured: same seed, same machine, twice** (the identical-seed replica design of Pham et al., ASE 2020 and Zhuang et
al., MLSys 2022). `alexnet_k3x3_stride2_2pool2x2_map8_gap_bn` on the laptop, Phase 11's protocol cut to 10 FP32 epochs,
two runs of `scripts/train.py` from scratch (TF32 off, `cudnn.deterministic`): every epoch's train loss and val accuracy
equal, all 32 weight tensors of the last epoch bitwise equal, val and test logits bitwise equal (test top-1 23.32 both).
The op alone, 30 backward passes on the GPU: `AdaptiveAvgPool2d(1)` on 8x8 is bitwise stable; `AdaptiveAvgPool2d(6)` on
8x8, 4x4 or 1x1 and `AdaptiveAvgPool2d(7)` on 2x2 are not, and `torch.use_deterministic_algorithms(True)` rejects them
("adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation"). So a GAP cell retrained on the same
GPU and code reproduces exactly; an FC cell reproduces within its nondeterminism, which a same-seed FC pair would size.

**Measured: how large the FC nondeterminism is.** Its GAP twin `alexnet_k3x3_stride2_2pool2x2_map8_fc_bn`, same setup
(laptop, 10 FP32 epochs), three runs: seed 42, seed 42 again, seed 43. The same-seed pair already differs in epoch 1
(train loss 4.9005 vs 4.8995) and its val accuracy differs by up to 0.46 pp at equal epochs. After 10 epochs, on the test
set:

| Pair | Test predictions that differ | Test top-1 difference | Median relative weight distance |
|---|---|---|---|
| same seed (nondeterminism only) | 17.9% | 0.23 pp (36.02 vs 36.25) | 0.19 |
| different seed (42 vs 43) | 32.8% | 0.21 pp (36.02 vs 35.81) | 1.41 |

One pair each, early in training, so indicative only -- but it agrees with Zhuang et al. (MLSys 2022): nondeterminism
alone moves accuracy about as much as a new seed, while the weights stay much closer. Consequence for the design: the
noise floor (`_seed43/_seed44`) reruns two GAP cells, which are deterministic, so it measures seed variance only; an FC
cell's run-to-run spread adds nondeterminism of comparable size, and the GAP-based band can understate it. Not acted
on (decided 2026-10-06): differences of this size don't matter here; what has to reproduce is the macro -- every
quantization stage the literature's, correctly applied, on any machine (next section).

## INT8 correctness: every stage, every model, every machine (2026-10-06)

What must hold for a reported INT8 number, and what now enforces it:

| Stage | Definition | Enforced by |
|---|---|---|
| QAT graph | weights 8-bit symmetric per-channel in [-127, 127], activations 8-bit affine 0..255 (Jacob et al., CVPR 2018; Wu et al. 2020 Sec. 6); Conv-BN(-ReLU) folded (Jacob Sec. 3.2; Krishnamoorthi 2018); every ReLU fused into its producer | `tests/test_quantization.py::test_every_phase_11_model_is_the_literature_int8_through_qat_and_convert`, over every model the design trains (72, read from the phase_11_*.yaml, so a new cell is covered when added) |
| Conversion | every Conv/Linear but the logits layer an int8 kernel with codes in [-127, 127]; no float Conv/Linear, no BatchNorm left; the logits layer int8 weights + FP32 output | the same test (72/72 pass; 79 s) |
| Kernels on this CPU | int32-exact accumulation (Jacob et al. 2018) for the three int8 GEMM kinds the design runs: first conv (image zero point, the saturating case), depthwise conv, FC-head Linear | `int8_kernel_error_steps()` before every INT8 evaluation in `scripts/train.py`; the run stops if > 1 step |
| The trained model | its INT8 reproduces its QAT (the premise of QAT, Jacob et al. 2018) | `scripts/train.py` fails the job if `agreement_qat_int8` < `MIN_QAT_INT8_AGREEMENT` = 0.90, summary kept (`tests/test_train_cli.py`) |

Kernel exactness measured on the laptop (i7-13650HX, AVX-VNNI), max steps off exact int32: qnnpack 1 / 1 / 0 (first
conv / depthwise / Linear 9216), onednn 0 / 1 / 0, fbgemm 69-103 / 97 / 12; residual `add_relu` 0 on all three. The
0.90 agreement cut is empirical, not a literature number: exact kernels reproduced QAT on 97.5% of test predictions,
saturating onednn on 64% (the beagle run above). The structure is code, the same on every machine; the kernels and
the trained model's agreement are what a machine could change, and both are now checked on the machine the job runs on.
Replaces two single-model tests (the QAT-definition and fake-quant-present checks) that the 72-model test covers.

## Protocol audit (2026-10-07)

**Why.** The user asked for an audit of the whole infrastructure after the qnnpack fix: no choice without a reference
(or, if unavoidable, justified and written down), and no hidden trap that could invalidate a result. Focus on what can
change a conclusion, not on run-to-run noise.

**The INT8 fix holds.** The k3x3 gap_bn pilot's QAT checkpoint re-evaluated on the laptop: qnnpack INT8 = beagle's
qnnpack INT8 on 99.98% of test predictions (max logit difference 0.008), the local GPU fake-quant = the pilot's QAT
logits on 99.87%; INT8 0.06 pp under QAT. The 4 pilots: INT8 within 0.22 pp of QAT and 0.3 pp of FP32. 229 tests pass.
Why `agreement_qat_int8` was 0.96-0.975 and not the pilot gate's 0.99: PyTorch's fused fake-quant (qconfig version 1,
the default -- also torch's own `get_default_qat_qconfig`) computes per-channel symmetric weight scales in C++ (255/254
of the observer's, ±0.39%), while `convert()` quantizes with the observer's `calculate_qparams()`: the INT8 weights left
the grid QAT trained on. Converting with the fused scales lifted the agreement 0.9614 -> 0.9850 (top-1 +0.01 pp); the rest
is fp32 rounding (fake-quant on GPU vs CPU agree on 98.6%). Fixed: `INT8_QAT_QCONFIG` from `version=0` (plain
FakeQuantize, scale == calculate_qparams, mismatch 0 measured), and `scripts/train.py` switches the trained QAT model's
observers off before converting (convert runs each weight through its fake-quant; an observer still on moved the grid).

**Decisions** (references in `_protocols/no_patience.yaml`'s header):

| Choice | Value | Reference / reason |
|---|---|---|
| Optimizer | SGD m0.9, lr 0.01, wd 5e-4, bs 128, every net | Krizhevsky et al. 2012 Sec. 5; Simonyan & Zisserman 2015 Sec. 3.1 used the same values |
| Schedule, loss | cosine to 0, label smoothing 0.1 | He, T. et al. CVPR 2019 ("Bag of Tricks") Sec. 5, the two together across ResNet/Inception/MobileNet. The original AlexNet recipe was considered (2012 + its author's fixed schedule, x 250^(-1/3) at 25/50/75%, Krizhevsky 2014) and rejected: its regularization was set for 1.2M images x 90 epochs, here ~100k x 500, so FC heads would be compared by how they overfit |
| One recipe for every net | controlled comparison | Radosavovic et al. ICCV 2019. Limitation stated: not MobileNetV2's (wd 4e-5) or ResNet's (lr 0.1, wd 1e-4) own recipe; recipes can reorder architectures (Bello et al. 2021; Wightman et al. 2021) |
| Data | Tiny ImageNet 64x64 (speed); shifts up to 4 px + flip + AutoAugment | Chrabaszcz et al. 2017 Sec. 3 (ImageNet 64x64: "random image shifts (up to 4 pixels)"; downsampled-ImageNet hyperparameter conclusions carry over); = AlexNet's crop range relative to the image (32/256 = 8/64); Cubuk et al. 2019 |
| Budget | 500 epochs | the author's choice, so every model converges (352k iterations, < AlexNet's ~844k); checked per run (`convergence()`) |
| Init | He | deviation from AlexNet's fixed N(0, 0.01), whose starting signal scale grows with sqrt(fan-in) -- with the kernel size this study varies; He et al. 2015 Eq. 10 normalizes by the fan |
| FC head | with Dropout 0.5 | AlexNet's and VGG's own head (Krizhevsky 2012 Sec. 4.2; Simonyan 2015). The FC-without-Dropout level (every main-grid FC cell until today) is a head no reference uses: the FC pilots memorized the training set (train 99.87% / val 38.5%). It stays only as the Dropout contrast: 64px no-BN x 4 kernels (2 already trained) and VGG 3x3/2x2. Total runs unchanged |
| Noise floor | 4 kernels x GAP + BN x seeds 42/43/44 (+4 runs) | Bouthillier et al. MLSys 2021. The kernel effect (1-3 pp in the old runs) was the size of the seed band; now every kernel has 3 replicates in one cell |
| Sanity guard | `MIN_QAT_INT8_AGREEMENT` 0.90 | a check that the code is right, not a reported metric |

**Effective cost of FC heads.** `AdaptiveAvgPool2d(6)` (VGG: 7) upsamples small maps: a 1x1 map becomes 36 copies, and
fc1 multiplies each with its own weights. The pool is linear, so fc1 folds exactly into a Linear on min(k*k, 36) positions
(`tests/test_registry.py::test_replicated_fc_weights_fold_away_exactly`). In 18 of the 34 FC cells 36-74% of the
parameters (and the same number of MACs and INT8 bytes) are such copies -- torchvision's AlexNet at 64 px: 57.8M
parameters, 21M of which compute the function; VGG16 at a 2x2 map: 94M of 135M. `ml.reporting.replicated_fc_weights`
counts them; `analyze_geometry.load` adds `params_eff_m`/`macs_eff_m`/`int8_eff_mb`, which figures 03, 16, 20 and the
factor contrasts' cost changes use; the as-built numbers stay in the tables.

**Operations.** `configs/slurm/tupi_beagle.yaml` (partition `tupi,beagle`, 15G): AlexNet/family jobs start on whichever
frees first, replacing the manual `scontrol update`; `beagle.yaml` gets 24 h + requeue/signal (8 h without requeue
before, shorter than a 16 h beagle run). `scripts.cluster` passes `exclude`. `refuse_foreign_run_dir` lets a run dir take
a new QAT protocol on its FP32 once its `qat_*` checkpoints are archived. Latency: once every run is done, all are
re-evaluated on one machine type (`exclude: tupi1,tupi2` -> tupi3-6), so figure 20 compares every cell; Phase 11 cells no
longer run on the laptop (everything in `outputs/pcad`). PCAD is the only copy of the checkpoints: tracked outputs are
rsynced to the laptop and committed, and the final checkpoints get a second copy.

**Pre-registered analysis** (written before the results):
- Primary endpoint: top-1 on the test set (Tiny ImageNet's official val), FP32 and INT8.
- Kernel effect: GAP + BN cell, 3 seeds per kernel; each kernel vs 11-5-3-3-3 as the mean difference with a 95% CI from
  the pooled replicate SD (pure error, 8 df; Montgomery, Design and Analysis of Experiments).
- Every other single-seed contrast (matched pairs, `factor_effects`): judged against the noise band
  2*sqrt(2)*pooled SD; inside it = "no evidence of a difference", never a ranking.
- Per pair, exact McNemar on the test images (Dietterich 1998), Holm across the contrast table (Holm 1979).

**QAT and BN, measured** (laptop, the k3x3 gap_bn pilot's FP32 checkpoint from PCAD, the real `scripts/train.py` with a
scratch runtime root; validation top-1 per QAT epoch, FP32 = 54.62):

| QAT BN handling | Epochs 1-3 | Rest of the run |
|---|---|---|
| updated, frozen after 3 epochs (protocol until today) | 54.07 / 54.27 / 54.25 | 52.8-53.6 after the freeze |
| statically folded from the start (Nagel 2021, the first plan) | 53.03 / 53.08 / 53.30 | 52.5-53.3 (stopped at epoch 10) |
| **intact, never frozen (Wu 2020; Nagel 2021 per-channel)** | 53.93 / 54.08 / 53.99 | 53.9-54.5 for all 50 epochs |

Training with BN frozen at lr 1e-4 costs ~1.5 pp whenever it starts -- not the freeze event. So the protocol keeps BN
intact (`freeze_bn_epoch: null`), which also takes the whole QAT stage from one reference: Wu et al. leave BN an
unquantized layer during QAT, and Nagel et al. (Sec. 4.2, eq. 41, Table 7) show that with per-channel weights it folds
into each channel's scale afterwards, on par or better than static folding. In PyTorch that is the fused Conv-BN never
frozen: the per-channel symmetric grid of W*gamma/sigma is gamma/sigma times the grid of W. Lr 1e-5 with static folding
(Krishnamoorthi 2018's fine-tuning step) was the alternative, rejected as a second source for one stage and a value
picked on one cell. The BN-intact run, criteria fixed before it ran: test FP32 53.50 (= PCAD's, the FP32 reuse is exact),
QAT 53.68, INT8 53.64; `agreement_qat_int8` 0.9892 (0.9617 with the fused fake-quant and the old QAT; >= 0.98 required);
best QAT epoch 16; `int8_kernel_max_err_steps` 1.

**Tests that look for the errors that would invalidate a result** (2026-10-07, after a review of the code the runs
execute). The review found one more bug: the head contrast had vanished from the analysis. `factor_effects.pairs_of`
pairs cells equal in every other factor, and after the FC -> FC + Dropout swap GAP (no Dropout) and the FC head (Dropout)
differ in two -- the head contrast fell from 17 matched pairs to the 6 no-Dropout ones. Fixed: a PAIRS level may move two
columns, and the head contrast is GAP vs FC + Dropout (`FC_HEAD`/`GAP_HEAD`). New tests, each shown to fail on the bug
it is for:

| Test | Checks | Shown to catch |
|---|---|---|
| `test_quantization.py::test_every_int8_layer_computes_what_its_qat_layer_simulated` (72 models) | per layer, same quantized input: INT8 kernel within 1 step of exact int32, the QAT layer within 1 step of it, logits layer equal | the fused fake-quant (version=1): "QAT != INT8 definition"; fbgemm's int16 saturation: "INT8 kernel off exact int32" |
| `test_phase11_figures.py::test_the_design_has_every_contrast_and_no_orphan_cell` | matched pairs per contrast pinned to the design; every cell in some contrast; the noise floor's 4 cells in both seed yamls | the head contrast above (6 pairs vs 17) |
| `test_phase11_runs.py` (new) | every synced run on the current QUANT_PROTOCOL: protocol == its yaml's, every job clean and in this branch's history, 500/50 epochs, alive (top-1 >> chance, finite loss), INT8 kernel and agreement guards, all logits; one dataset; older-protocol runs listed as a warning | a run on uncommitted code (the laptop diagnostic run) |
| `test_phase11_runs.py::test_the_test_set_and_the_split_are_what_the_protocol_says` | test set 50 per class through the train class indices; 90/10 split disjoint and fixed by the seed | -- (holds on the real data) |
| `test_train_pipeline.py::test_two_fresh_runs_with_the_same_seed_train_the_same_weights` | two runs from scratch, bit-identical FP32 and QAT weights | `set_global_seed` removed: QAT weights differ |
| `test_train_pipeline.py::test_a_qat_redone_on_archived_artifacts_records_its_own_training_time` | a re-QAT records its own time, FP32's kept | the bug fixed in 01349d2 |

`test_phase11_runs.py` reads the tracked outputs, so it checks the real runs once they are synced from PCAD into git.
Full suite: 309 passed, 1 skipped (no current-protocol run synced yet).

## Three jobs dropped by a Slurm outage; the feeder now waits one out (2026-10-08)

**What happened.** On 2026-10-07 at 15:55 PCAD's Slurm controller was unreachable for about a minute. `squeue` failed,
`scripts/pcad/feed_queue.sh` read that as 0 jobs queued and kept popping lines, and each `sbatch` ("Unable to contact
slurm controller") sent its line to `~/queue_phase11_full.txt.failed`, which is never retried. Dropped: the 2026-10-07
QAT redo of two pilots, `alexnet_k3x3_stride2_2pool2x2_map8_gap_bn` (seed 42, the noise floor's reference cell) and
`alexnet_k2x2_stride2_2pool2x2_map8_fc_nobn`, and the new cell `alexnet_k11-5-3_stride2_2pool2x2_map8_fcdrop_nobn`. The
other two pilots' redo (829369, 829370) had been submitted at 11:22. Found 2026-10-08 while auditing the synced runs: the
4 pilots' live summaries are still on QUANT_PROTOCOL 2026-10-06 (their QAT in `archive_qat_bnfreeze/`) and none of the 3
was queued anywhere.

**Fix.** A failing `squeue` now waits `SLEEP` instead of counting 0 jobs, and a submit that fails with a transient Slurm
error (controller unreachable, socket timeout, "temporarily unable") goes back to the head of the queue like the QOS
limit, logged as `RETRY`. `tests/test_feed_queue.py::test_feed_queue_waits_out_a_slurm_controller_outage` replays the
outage and fails on the old script (both lines in `.failed`). The 3 lines went back to the head of the queue file and the
feeder was restarted on this commit.
