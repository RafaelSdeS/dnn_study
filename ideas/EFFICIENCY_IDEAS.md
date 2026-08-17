# Efficiency Ideas — Candidate Techniques (Unscoped)

Brainstormed techniques for reducing size/FLOPs or improving accuracy, beyond what Phases 1–9
already cover (kernel restriction, compensation blocks, QAT/INT8, mixed-precision PTQ, structured
pruning, k-means/entropy weight compression). None of these are scoped or scheduled — this is a
menu to pick from, not a plan. See **Priority** at the bottom for where I'd start.

---

## Architecture-level

### Ghost modules (GhostNet)
Generate half your output feature maps with a normal conv, then generate the other half from
those via a cheap depthwise op instead of another full conv.

- **Fits here because:** the "ghost" op is depthwise/small-kernel by construction — directly
  Winograd-friendly, same angle as `compensation.py`'s `AlexNetDepthwiseSep`.
- **Would touch:** a new block in `models/compensation.py`, same shape as the existing
  depthwise-separable variant.
- **Tradeoff:** cuts conv FLOPs roughly in half at matched channel count; ghost features are
  linear transforms of real ones, so accuracy ceiling is a bit below a full conv of the same width.
- **Key paper:** Han et al., "GhostNet: More Features from Cheap Operations" (CVPR 2020).

### Channel shuffle (ShuffleNet-style)
After a grouped convolution, permute channels across groups before the next layer so information
mixes between groups instead of staying siloed.

- **Fits here because:** `AlexNetGroupConv` already exists and underperformed (29.18% FP32,
  Phase 3) — group convs without shuffling can't mix cross-group info, which is a likely cause.
  Shuffle is near-zero extra FLOPs (just an index permutation).
- **Would touch:** `models/compensation.py`, add a shuffle op after `AlexNetGroupConv`'s grouped
  convs, or a new `AlexNetGroupConvShuffle` variant for a clean before/after comparison.
- **Tradeoff:** free efficiency-wise; only real cost is code complexity of the shuffle op under
  QAT fusion (shuffle isn't a fusable Conv-BN-ReLU pattern, needs to sit between fused blocks).
- **Key paper:** Zhang et al., "ShuffleNet: An Extremely Efficient CNN for Mobile Devices" (CVPR 2018).

### Tucker/CP decomposition of trained conv kernels
Factor an existing trained conv weight tensor into two (or three) smaller convs post-hoc, instead
of hand-designing the factorization like `AlexNetFactorized` does.

- **Fits here because:** it's a more aggressive, *learned* version of what `AlexNetFactorized`
  already does by hand (spatial separable 3×3 → 3×1 + 1×3). Could be applied to any existing
  checkpoint without retraining from scratch (fine-tune after decomposition).
- **Would touch:** a new `ml/decomposition.py` utility (rank selection, Tucker/CP factorization via
  `tensorly` or manual SVD-based approach) applied to an existing `_best.pth`.
- **Tradeoff:** needs a new dependency (`tensorly`) unless implemented via numpy SVD by hand for
  the 2D case; decomposition rank is a tunable knob (more rank = more accuracy, less compression).
- **Key paper:** Kim et al., "Compression of Deep Convolutional Neural Networks for Fast and Low
  Power Mobile Applications" (ICLR 2016).

---

## Training-level (accuracy without changing inference cost)

### Knowledge distillation from existing pretrained models
Train a small-kernel/compensation student (e.g. `AlexNetBottleneck`, `AlexNetFire`) against a
soft-label loss from an already-trained teacher — `ResNet18TV` or `MobileNetV2TV` from Phase 1 are
already in the repo and pretrained.

- **Fits here because:** it directly answers the project's own open question ("do small-kernel
  CNNs match pretrained models?") and costs nothing at inference — the student architecture,
  size, and FLOPs are unchanged.
- **Would touch:** a distillation loss (KL on softened logits + task loss) in a `Trainer` subclass
  or an optional loss-mixing hook in `ml/trainer.py`, reusing `ResNet18TV`/`MobileNetV2TV` as
  frozen teachers.
- **Tradeoff:** roughly doubles training-time compute (teacher forward pass every step) but zero
  extra cost at deployment; needs a teacher checkpoint already trained to reasonable accuracy.
- **Key paper:** Hinton et al., "Distilling the Knowledge in a Neural Network" (NIPS-W 2015).

### Mixup / CutMix + label smoothing
Standard data-level regularization: blend pairs of images/labels (Mixup) or paste patches between
images (CutMix), plus soften one-hot targets.

- **Fits here because:** Tiny ImageNet-200 is small enough (relative to 200 classes) that
  regularization like this reliably helps, and it's a pure `DataConfig`/loss change — no
  architecture or QAT implications at all.
- **Would touch:** `ml/data.py` (batch-level mixing) or a collate function, plus a label-smoothing
  flag in `TrainerConfig`.
- **Tradeoff:** cheapest idea on this list to try; usually needs a few more epochs to converge
  since the training signal is noisier per-batch.
- **Key papers:** Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018); Yun et
  al., "CutMix" (ICCV 2019).

### Self-distillation across QAT stages (born-again style)
Use the FP32 model as a soft-label teacher for its own QAT fine-tuning pass, instead of QAT
training against hard labels only.

- **Fits here because:** could directly shrink the FP32→INT8 accuracy drop this project already
  tracks per architecture (`quantization_drop_top1` in `make_run_summary()`), especially for the
  currently QAT-unstable ones (`AlexNetDepthwiseSep` at –2.92pp, `AlexNetSmallKernel` at –9.89pp).
- **Would touch:** `ml/quantization.py`'s `make_qat_callback` or the QAT fit loop, adding a soft-
  label loss term sourced from the already-trained FP32 checkpoint.
- **Tradeoff:** small extra compute (one extra frozen forward pass during QAT); doesn't help if the
  QAT instability is from the quantization ops themselves rather than optimization difficulty.
- **Key paper:** Furlanello et al., "Born Again Neural Networks" (ICML 2018).

---

## Compression-level (post-hoc size reduction)

### N:M structured sparsity or ternary weights
Push compression past INT8 — either enforce N-nonzero-per-M-weight blocks (e.g. 2:4, which modern
sparse tensor cores accelerate directly) or go to ternary {-1, 0, 1} weights.

- **Fits here because:** natural extension of `ml/quantization_advanced.py`'s existing
  mixed-precision/sub-INT8 machinery (`assign_mixed_precision`, `apply_weight_ptq`); Phase 9's
  `measure_compression.py` already measures headroom below nominal INT8 via k-means clustering —
  this is a further step down that same axis.
- **Would touch:** `ml/quantization_advanced.py`, add a sparsity-mask or ternary-weight PTQ path
  alongside the existing mixed-precision one.
- **Tradeoff:** accuracy drop grows fast below ~2 bits/weight without fine-tuning; N:M sparsity
  needs hardware/kernel support to realize a real speedup, not just a size number on disk.
- **Key papers:** Zhou et al., "Learning N:M Fine-grained Structured Sparse Neural Networks From
  Scratch" (ICLR 2021); Li et al., "Ternary Weight Networks" (2016).

### Soft weight-sharing (joint prune + quantize)
Instead of the current post-hoc k-means clustering in `measure_compression.py`, learn a Gaussian-
mixture prior over the weights *during* training/fine-tuning, so pruning and clustering emerge
from the loss rather than being applied after the fact.

- **Fits here because:** it's the "trained" version of what `measure_compression.py --evaluate`
  already does post-hoc — likely to close the accuracy gap the post-hoc k-means approach pays.
- **Would touch:** a new fine-tuning loop with a mixture-prior regularization term, probably its
  own script alongside `scripts/measure_compression.py` rather than modifying it in place.
- **Tradeoff:** meaningfully more implementation work than the other items here (custom
  regularized training loop, not just a post-hoc pass); best framed as a follow-up to the existing
  Phase 9 Task 3 result if that headroom turns out to matter.
- **Key paper:** Ullrich et al., "Soft Weight-Sharing for Neural Network Compression" (ICLR 2017).

---

## Directly tied to the Winograd thesis

### Winograd-aware training
Add a fine-tuning step that regularizes weights against the numerical error Winograd's F(m,r)
transform introduces (the transform amplifies certain weight distributions more than others),
recovering accuracy specifically for the Winograd-friendly small kernels this project argues for.

- **Fits here because:** this is the most direct way to strengthen the project's central claim —
  right now the study measures Winograd *speed* (Phase 6) and *accuracy* (Phases 2–3) as separate
  axes; this would let a model be trained to be good at both simultaneously, rather than accepting
  whatever error the transform happens to introduce.
- **Would touch:** a new regularization term in the FP32 or QAT fine-tuning loop, informed by
  `scripts/winograd_quant_error.py` (Phase 6 extension) which already measures INT8 quantization
  error from Winograd F(2×2,3×3) transforms — the loss term would penalize weight patterns that
  script flags as high-error.
- **Tradeoff:** the most novel and most implementation-heavy item on this list; requires
  understanding the Winograd transform's error characteristics well enough to write a correct
  regularizer, not just import one.
- **Key paper:** Li et al., "Winograd Convolution for Deep Neural Networks: Efficient Point
  Selection" / Liu et al., "Efficient Sparse-Winograd Convolutional Neural Networks" (ICLR 2018,
  closest prior art for "Winograd-aware" training).

### Per-layer kernel-size search (tiny differentiable NAS)
Instead of hand-designing kernel-size choices per layer (`AlexNetMixed`, `AlexNetStacked`), let a
differentiable search (DARTS-style, over a small discrete set like `{1×1, 2×2, 3×3}` per layer)
pick them.

- **Fits here because:** automates exactly the manual tradeoff Phase 2 explored by hand — would
  either validate the hand-picked choices in `alexnet_variants.py` or find a better combination
  the manual sweep missed.
- **Would touch:** a new search script, likely reusing `ml/trainer.py`'s loop with a
  supernet-style model that has per-layer kernel-choice weights, then a discretization step at the
  end to emit a normal `models/` architecture.
- **Tradeoff:** search cost (training a supernet) is larger than any single model in the current
  inventory; also the search space here (kernel size only, not depth/width) is narrow enough that
  a full NAS framework is probably overkill — a small custom DARTS-lite loop is more in scope.
- **Key paper:** Liu et al., "DARTS: Differentiable Architecture Search" (ICLR 2019).

---

## Priority

If picking one to actually try next:

1. **Knowledge distillation** — cheapest to implement (reuses `ResNet18TV`/`MobileNetV2TV` already
   in the repo, zero inference-cost change), and answers a question the project already asks.
2. **Winograd-aware training** — most novel, most directly strengthens the project's central
   argument, but the most implementation-heavy; worth scoping properly (its own PLAN doc) before
   starting, not a quick add.

Everything else here is lower-priority filler for later — see `TODO.md` Phase 10 for where this
list is tracked.
