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
   different protocol from the report's, same direction). The params/MACs advantage (0.39M / 39.5M vs.
   2.30M / 167.0M) is measured fact and unaffected.
7. `VGG16` here is torchvision cfg D **plus BatchNorm** (the original has none) and its 2×2 features are
   upsampled to 7×7 by AAP(7,7) at 64×64 — degenerate too, but milder than AlexNetTV's 1×1.

Corrected in this pass (nothing committed): `report/ic_report.tex` (architectures list, new "Geometria a 64×64"
paragraph, kernel-cost discussion, Limitações (i)), `docs/plans/MODELS.md`, `docs/plans/BEST_MODELS.md`,
`models/alexnet_variants.py`/`models/baselines.py` docstrings, `configs/experiments/phase_11_kernel_size_comparison.yaml`,
`TODO.md`, `CLAUDE.md`. `report/ic_report.pdf` and the Phase 11 figures were not rebuilt. Still open: the
missing controls listed in `TODO.md` (Phase 2 section) — none run.
