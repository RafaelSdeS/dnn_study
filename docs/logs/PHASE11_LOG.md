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
| 8-bit per-tensor affine activations, EMA min/max, BN folding (`INT8_QAT_QCONFIG`, onednn's QAT qconfig) | Jacob et al., CVPR 2018; Krishnamoorthi 2018 |
| Per-channel symmetric weights in [-127, 127] | Wu et al. 2020 Sec. 6; LiteRT int8 spec |
| Inputs and weights of every Conv/Linear quantized; logits output FP32 (`_FloatLogits`: INT8 input + weights, stored as int8); pools requantized as the next layer's input | Wu et al. 2020 Sec. 3, 5.1 |
| Activation fused into its producer (`fuse_sequential_relus`, `FloatFunctional.add_relu`) | Jacob 2018; PyTorch `fuse_modules`; LiteRT fused activations |
| QAT 50 ep (1/10 of FP32's 500 ep), same optimizer, lr 1e-4 (1/100 of FP32's 0.01) cosine to 1e-6 (1/100 of that) (`_protocols/no_patience.yaml`) | Wu et al. 2020 App. A.2 |
| Observers frozen after epoch 4, BN stats after epoch 3 | torchvision `references/classification/train_quantization.py` |
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
| LR, batch | 0.01, 128 | Krizhevsky et al. 2012 (VGG: 0.01 at batch 256) |
| Schedule | cosine annealing to 0, no warmup | Loshchilov & Hutter, ICLR 2017 (replaces AlexNet's /10 on plateau); warmup is for large-minibatch LR scaling (Goyal et al. 2017), AlexNet used none at this lr/batch |
| Budget | 500 epochs, fixed, no early stopping; best epoch picked on the 90/10 split | Li, Yumer & Ramanan, ICLR 2020 (fixed budget, LR decayed to zero by its end); Cawley & Talbot, JMLR 2010 |
| Augmentation | 4-px pad + random crop + horizontal flip, then AutoAugment's ImageNet policy | He et al., CVPR 2016 Sec. 4.2; Cubuk et al., CVPR 2019 |
| Init | He normal (fan_out) on convs, N(0, 0.01) on Linears, in every from-scratch model | He et al., ICCV 2015; Krizhevsky et al. 2012 (the Linear std); torchvision's VGG init |
| Loss | cross-entropy, label smoothing 0.1 | Szegedy et al., CVPR 2016 |
| Precision | mixed-precision training, FP32 evaluation | Micikevicius et al., ICLR 2018 |
| Pretrained cells (`_pt`) | the same recipe -- pretraining is the only variable of their contrast | design choice of the factorial |

QAT, INT8 and evaluation: the table of the previous section (QAT = Wu et al. 2020 App. A.2 applied to this recipe:
50 ep, SGD, lr 1e-4 cosine to 1e-6).

**What no single reference fixes.** The 500-epoch budget: no Tiny ImageNet paper prescribes one. 500 epochs = 352k
iterations at batch 128, between CIFAR's long schedules (WRN 200 ep = 78k, DenseNet 300 ep = 234k) and ImageNet's
(ResNet 600k at batch 256, AlexNet ~844k at batch 128); returns diminish with budget (Wightman et al. 2021: 100/300/600
ep -> 78.1/79.8/80.4%). Under the old recipe all 51 finished FP32 runs had converged (median best epoch 423, median
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
