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
