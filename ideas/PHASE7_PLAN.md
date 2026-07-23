# Phase 7 — Detection & Segmentation Kernel Study (Implementation Plan)

Phases 1–6 answer the kernel-restriction question for **classification** only (Tiny ImageNet-200,
64×64, single label per image). Phase 7 asks whether the same finding — that Bottleneck/Fire
(small-kernel + channel-compression) backbones match or beat large-kernel baselines at a fraction
of the size, and survive INT8 conversion with <1pp drop — **transfers to dense prediction**, where
the loss is spatial (per-box or per-pixel) rather than a single softmax. This is the one item in
`TODO.md` that is currently unaddressed by any completed phase and is explicitly named in the
project's stated scope ("Detection/segmentation are future work").

This plan is a blueprint only — no code is written here. It follows the same structure as
`ideas/PHASE6_PLAN.md`: hypotheses first, then a decision record for every design choice that
isn't forced by the codebase, then task-by-task implementation detail, then a self-critical
blocking-issues pass.

**Scale warning, stated up front:** Phase 7 is categorically bigger than Phase 6. Phase 6 added
one profiling module on top of infrastructure that already existed (Trainer, data loaders, QAT
pipeline all reused as-is). Phase 7 needs a new dataset family, new box/mask-aware data
augmentation, two new model-assembly layers, two new loss functions, two new metrics, and a
Trainer variant — while still landing inside the existing QAT → INT8 → compare pipeline so results
are comparable to Phases 1–6. Budget accordingly (see **SCOPE & EFFORT**).

---

## Research Hypotheses

### H1: Small-Kernel Compensation Transfers to Dense Prediction
**Claim:** `alexnet_bottleneck` and `alexnet_fire` (Phase 3's Pareto-optimal, quantization-stable
backbones) reach competitive mAP/mIoU against the `alexnet_tv` large-kernel baseline at far
smaller backbone size, mirroring the classification result (44.6%/44.0% vs. 32.9% top-1 at
1/100–1/150th the size).

**Expected Outcome:**
- Detection: `alexnet_bottleneck`/`alexnet_fire` backbones reach ≥80% of `alexnet_tv`'s mAP@[.5:.95]
  at <10% of its backbone parameter count.
- Segmentation: same relative pattern on mIoU.
- Backbone size ranking should hold in this new task exactly as it did in Phase 3 — this is a
  necessary but not sufficient condition, since size doesn't change between tasks, only the head does.

**Evidence to Collect:** `mAP@[.5:.95]`, `mAP@.5`, `mIoU`, `pixel_accuracy` per backbone, plus
backbone-only parameter/size counts (already known from Phase 3/`results/results_aggregate/model_details_cross_phase.csv`).

**Acceptance Criterion:** Backbone accuracy-per-parameter ranking (Bottleneck/Fire > AlexNetTV)
from Phase 3 reproduces in both detection and segmentation, allowing for a wider absolute accuracy
gap than in classification (see H3).

---

### H2: Quantization Robustness Ranking Transfers to Heads
**Claim:** The backbone property that made Bottleneck/Fire QAT-stable in classification
(−0.08pp / +0.33pp INT8 drop, Phase 3) is a property of the backbone's *activation distributions*,
not of the classification task — so it should carry over even though the head sitting on top of
the backbone is entirely new (SSDLite for detection, DeepLabHead for segmentation).

**Expected Outcome:** INT8 mAP/mIoU drop for Bottleneck/Fire backbones stays within ±2pp;
AlexNetTV, which already showed a larger classification drop context (−0.98pp, but off a much
lower FP32 base), does not necessarily get *worse* — this hypothesis is specifically about
backbone ranking, not absolute magnitude.

**Evidence to Collect:** FP32 vs. INT8 mAP/mIoU per backbone; compare drop ranking to Phase 3's
`quantization_drop_top1` ranking (Spearman correlation).

**Acceptance Criterion:** Spearman ρ > 0.6 between Phase 3's per-backbone INT8 accuracy-drop
ranking and Phase 7's per-backbone INT8 mAP/mIoU-drop ranking. (ρ threshold is lower than Phase
6's H4 because only 3 backbones are compared here — with n=3, ρ has few achievable values, so
this is a directional check, not a strong statistical claim; report the raw numbers alongside ρ.)

---

### H3: Dense Prediction Is More Receptive-Field-Sensitive Than Classification
**Claim:** Localization (detection) and per-pixel labeling (segmentation) depend more heavily on
spatial context than whole-image classification does, because a bounding box or a pixel boundary
needs information from beyond the object itself (occlusion reasoning, context for scale/aspect
disambiguation). Phase 3's entire compensation study exists because small kernels shrink the
receptive field — so this effect should bite harder here than it did for classification, where
GAP + 5 stages already gave "enough" effective receptive field for a whole-image label.

**Expected Outcome:** The *relative* gap between AlexNetTV (11×11 first-layer kernel, largest
native receptive field of the three) and Bottleneck/Fire should be smaller in classification-space
terms than it is on mAP (i.e., AlexNetTV closes some or all of the 12–15pp classification gap it
had against Bottleneck/Fire once localization is required) — or, if it doesn't, that is itself the
interesting negative result: Phase 3's channel-compression + effective-depth tricks are sufficient
even for dense prediction, and receptive field from raw kernel size is not the bottleneck people
assume it is for Winograd-friendly design. Report both outcomes as valid findings; this hypothesis
is deliberately falsifiable in either direction.

**Evidence to Collect:** Per-backbone classification-vs-detection-vs-segmentation accuracy deltas,
normalized (e.g., mAP / classification-top1 ratio per backbone) to separate "this backbone is
just worse overall" from "this backbone is disproportionately worse at localization."

**Acceptance Criterion:** None fixed in advance — this is exploratory/diagnostic, not a
pass/fail gate. Report the normalized deltas and let the direction of the effect answer the
question; pre-committing to a threshold here would just be p-hacking a 3-backbone comparison.

---

### H4: Head Architecture Reintroduces Winograd-Relevant Structure Independent of Backbone
**Claim:** SSDLite's extra blocks and prediction heads are depthwise-separable (see
`torchvision.models.detection.ssdlite._extra_block` — 1×1 → depthwise 3×3 stride-2 → 1×1, ReLU6),
i.e. **not** Winograd-eligible by the same `groups=1` criterion established in Phase 6 (H2), while
DeepLabHead's ASPP branches (dilated 3×3 convs at rates 12/24/36) are dense but **dilated**, a
third category Phase 6 never tested. So the backbone's Winograd-eligibility (established in Phase
6 for `alexnet_bottleneck`/`alexnet_fire`/`alexnet_tv`) does not determine the *whole model's*
Winograd profile once a head is attached — the head can dominate end-to-end latency structure.

**Expected Outcome:** Full detection-model latency is dominated by SSDLite's depthwise extra
blocks (memory-bound, not Winograd-accelerated) regardless of backbone choice; full
segmentation-model latency is dominated by ASPP's dilated convs (cuDNN typically falls back to
implicit-GEMM for dilated convs, not Winograd) regardless of backbone choice. If confirmed, the
practical implication is that backbone kernel-size choice matters less for end-to-end
detection/segmentation latency than Phase 6 suggested it does for pure classification — the head
architecture choice dominates.

**Evidence to Collect:** Reuse `ml/profiling.py`'s `profile_kernel_trace()` and
`profile_model_latency()` (Phase 6, already built) on the assembled detection/segmentation models;
per-module latency breakdown (backbone vs. head) via `torch.profiler` with `record_shapes=True`.

**Acceptance Criterion:** Head latency (SSDLite extra blocks + prediction convs, or ASPP + decoder)
exceeds 40% of total forward latency for at least 2 of 3 backbones, on at least one of the two
tasks. This is a call back to Phase 6's profiling infrastructure — no new profiling code is
required, only new models to point it at (see Task 9).

---

## Decision Records

### D1 — Dataset: Pascal VOC (2007+2012), not COCO-subset or Cityscapes

| Axis | Pascal VOC 2007+2012 | COCO subset | Cityscapes |
|---|---|---|---|
| Auto-download, no auth | ✅ `torchvision.datasets.VOCDetection(download=True)` / `VOCSegmentation(download=True)` | ⚠️ Full COCO is ~25 GB; a "subset" means hand-rolling a filtered download + a custom `pycocotools`-backed `Dataset`, more moving parts | ❌ Requires manual registration + login; cannot be scripted into a reproducible one-command pipeline |
| Disk footprint | ~2.9 GB (both years) | 25 GB+ full, subset still needs the full annotation JSON (~1 GB) to build from | ~11 GB (fine, but blocked by the auth issue above) |
| One dataset for both detection AND segmentation | ✅ VOC has both box (`VOCDetection`) and pixel-mask (`VOCSegmentation`) annotations over overlapping images/classes (20 object classes + background) | ❌ COCO has no built-in segmentation-class task in the classic sense (has instance masks, panoptic — different problem framing, would need its own head design) | ❌ Segmentation only, no detection boxes |
| Community precedent for small/lightweight-model benchmarking | ✅ Standard benchmark for SSD (Liu et al. 2016), Faster R-CNN (Ren et al. 2015), DeepLab (Chen et al. 2017) at exactly this model scale | ⚠️ COCO is the modern standard but sized for large-scale training runs, not a laptop 4060 | ⚠️ Designed for full-scene urban segmentation, larger images (2048×1024 native), heavier per-image cost |
| Fits `RTX 4060` (8.2 GB) + `RTX 4090` two-tier hardware setup already used in Phase 6 | ✅ ~16.5K detection images, ~2.9K segmentation images — full epochs in minutes, not hours | ⚠️ Possible but requires more careful subsetting to stay small | ❌ Blocked before hardware is even a question |

**Chosen: Pascal VOC 2007 trainval + 2012 trainval for detection (`image_set="trainval"`, both
years concatenated via `torch.utils.data.ConcatDataset`), VOC 2007 test as the held-out detection
eval set (the standard "07+12 train, VOC07 test" protocol from the original SSD/Faster-RCNN
papers — reusing a known protocol means Phase 7's numbers are sanity-checkable against published
mAP figures for other lightweight detectors, even though this study isn't chasing SOTA).**
For segmentation, VOC 2012 `train`/`val` splits (`VOCSegmentation`, ~1.5K/~1.5K images, 21 classes
including background, 255 = ignore/boundary).

Reference: Everingham, M. et al. "The PASCAL Visual Object Classes (VOC) Challenge." IJCV 2010.

### D2 — Input Resolution: 256×256, not 64×64

Tiny ImageNet's 64×64 convention (used identically across Phases 1–6) is unsuitable here: VOC
objects at 64×64 shrink to a handful of pixels, well below what any of these lightweight
backbones' downsampling stages (stride 8–32) can represent as a coherent feature. This is not
a contradiction of the project's Tiny-ImageNet scope — Phase 7 is explicitly the extension into a
*different* dataset family (TODO.md: "Extend the kernel-restriction findings... to object
detection and semantic segmentation"), and kernel size (2×2 vs. 3×3 vs. 11×11 *convolution
kernels*) is orthogonal to *input resolution* — Winograd eligibility is about the conv kernel's
spatial extent, not the feature map's.

**Chosen: 256×256**, resized+letterboxed (aspect-preserving resize + pad, not a naive stretch,
so bounding boxes and object aspect ratios stay undistorted) via `torchvision.transforms.v2`. This
matches SSDLite's own convention of small-ish square inputs (SSDLite320 uses 320×320; 256 is
chosen here specifically to keep the smallest backbones' compute/memory profile close to what
already runs comfortably on the RTX 4060 in Phases 1–6, while still giving objects enough pixels
to be detectable). Document this explicitly in the notebook and any cross-phase comparison as a
deliberate resolution change, not an oversight.

### D3 — Backbones: `alexnet_bottleneck`, `alexnet_fire`, `alexnet_tv`

Directly what `TODO.md` asks for ("reuse Phase 3's Pareto-optimal backbones... paired with... a
large-kernel baseline backbone (AlexNetTV or VGGStyle)"). `alexnet_tv` is chosen over `vgg_style`
for the large-kernel role because it is the literal large-kernel control already used as the
Phase 6 baseline (`ideas/PHASE6_PLAN.md`'s selected-models table) — reusing the same control
across phases lets Phase 7's findings be read directly against Phase 6's latency/Winograd numbers
for the *same* model (H4 above depends on this). `vgg_style` (all-3×3, Winograd-eligible
throughout, Pareto-optimal in Phase 3) is a natural 4th backbone if time allows — flagged as a
stretch addition in **SCOPE & EFFORT**, not blocking.

### D4 — Detection Head: torchvision's `SSDLiteHead` + `DefaultBoxGenerator` + `SSD`, not a
hand-rolled detector

Reimplementing anchor matching, hard-negative mining, the multi-part SSD loss (smooth-L1 box
regression + cross-entropy classification), and NMS from scratch is a large, error-prone
undertaking with no benefit here — torchvision ships all of it, tested and widely used
(`torchvision.models.detection.ssdlite320_mobilenet_v3_large` is the reference implementation this
plan's assembly mirrors). Confirmed against the installed `torchvision==0.20.1` API
(see Task 3): `SSD.__init__` accepts any `backbone` module whose `forward(x)` returns an
`OrderedDict[str, Tensor]` of multi-scale feature maps — it does **not** require the backbone
to subclass anything from `torchvision.models.detection`, so a from-scratch `MODEL_REGISTRY`
backbone can be wrapped directly. Loss computation (`SmoothL1Loss` for boxes + a hard-negative-mined
CE for classification) and inference-time NMS live inside `SSD.forward()` already — reuse as-is,
do not reimplement.

Reference: Liu, W. et al. "SSD: Single Shot MultiBox Detector." ECCV 2016. Sandler, M. et al.
"MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018 (SSDLite's depthwise-separable
extra-block design).

### D5 — Segmentation Head: torchvision's `DeepLabHead` (ASPP), not a hand-rolled U-Net decoder

`torchvision.models.segmentation.deeplabv3.DeepLabHead(in_channels, num_classes, atrous_rates=(12,
24, 36))` takes a single backbone feature map and produces per-pixel logits, upsampled to input
resolution via bilinear interpolation — this is exactly the mechanism behind
`deeplabv3_mobilenet_v3_large`. Reusing it avoids hand-designing skip-connection wiring (U-Net-style)
for three architecturally different backbones. Trade-off, stated explicitly: ASPP's dilated 3×3
convs are a third conv category (dense, but `dilation>1`) that Phase 6 never profiled — this is
*useful* new data for H4, not a confound, but it does mean the segmentation head is not a "pure"
kernel-size-only comparison. A plain-stacked-3×3 decoder (no dilation) is noted as an optional
ablation in **MEDIUM-PRIORITY IMPROVEMENTS** if ASPP's dilation turns out to dominate the result.

Reference: Chen, L-C. et al. "Rethinking Atrous Convolution for Semantic Image Segmentation."
arXiv:1706.05587, 2017 (DeepLabv3).

### D6 — New dependencies: `pycocotools`, `torchmetrics` (already listed in `CLAUDE.md`'s Stack
but confirmed **not present** in `.venv` at plan-writing time — see Task 7), `torchvision.transforms.v2`
(already ships inside the pinned `torchvision==0.20.1`, no new install)

`torchmetrics.detection.MeanAveragePrecision` implements the COCO mAP protocol
(`mAP@[.5:.95]` averaged over 10 IoU thresholds, plus `mAP@.5`, `mAP@.75`, per-size breakdowns) and
needs `pycocotools` installed as a backend dependency. `torchmetrics.segmentation.MeanIoU` needs
`torchmetrics>=1.0` (soft requirement; verify exact version at implementation time — this
functionality was added in a 1.x minor release). Add both to `environment.yml`'s `pip:` block
alongside the existing `torchmetrics`/`fvcore`/`wandb` entries.

---

## Task 1 — Data Pipeline (`ml/det_seg_data.py`)

**What:** `create_voc_detection_loaders(cfg)` and `create_voc_segmentation_loaders(cfg)`, mirroring
`create_imagenet_loaders`'s signature/return shape (`(train_ds, val_ds, train_loader, val_loader)`)
so the rest of the pipeline (Trainer subclasses, config plumbing) stays structurally consistent
with Phases 1–6.

**Why:** Detection and segmentation need target-aware transforms (a horizontal flip must also flip
box x-coordinates or the mask) and detection needs a custom `collate_fn` (each image has a
variable number of boxes — the default `DataLoader` collate can't stack a ragged list of boxes
into one tensor). Neither of these needs exist anywhere in the current `ml/` package.

**How:**
- Wrap `torchvision.datasets.VOCDetection`/`VOCSegmentation` outputs into `torchvision.tv_tensors`
  (`tv_tensors.BoundingBoxes` for detection targets parsed out of VOC's XML annotation format,
  `tv_tensors.Mask` for segmentation's PNG masks) and drive all augmentation through
  `torchvision.transforms.v2` (`RandomHorizontalFlip`, `Resize`, custom letterbox pad — v2's
  transforms apply consistently across image + boxes/masks automatically once wrapped in
  `tv_tensors`, which is the entire reason to use v2 over hand-rolled box/mask flipping logic).
- VOC XML → `tv_tensors.BoundingBoxes` + integer label list: write a small `_parse_voc_target(xml)`
  helper (the 20 VOC class names → indices 1–20, 0 reserved for background per SSD convention).
- Detection `collate_fn`: `lambda batch: (torch.stack([b[0] for b in batch]), [b[1] for b in batch])`
  — images stack normally (fixed 256×256 after letterbox), targets stay a list of per-image dicts
  (`{"boxes": ..., "labels": ...}`), which is exactly what `torchvision.models.detection.SSD`
  expects for both training targets and loss computation.
- Segmentation needs no custom collate — masks are fixed-size after resize, stack normally.
- Determinism: same `torch.Generator().manual_seed(cfg.seed)` pattern as `create_imagenet_loaders`,
  though VOC's `train`/`val`/`trainval`/`test` splits are dataset-defined (not a random split), so
  no seeded permutation is needed for the split itself — seed only matters for shuffling and any
  augmentation randomness (already handled by seeding `torch`/`random` globally at notebook start,
  per existing `CLAUDE.md` reproducibility convention).

**Inputs:** A new `DetSegDataConfig` dataclass (`ml/config.py` addition) — `img_size: int = 256`,
`voc_root: str`, `batch_size`, `num_workers`, `seed` (mirrors `DataConfig` but detection/segmentation
don't need `train_val_split` since VOC provides fixed splits).

**Outputs:** Four `DataLoader`s (detection train/val, segmentation train/val), each yielding
`(image_batch, target)` where `target` is either a list-of-dicts (detection) or a mask tensor
batch (segmentation).

**Dependencies:** None beyond what's installed (`torchvision>=0.15` for `transforms.v2`/`tv_tensors`
— already satisfied by the pinned `0.20.1`).

**Deliverables:** `ml/det_seg_data.py`, `DetSegDataConfig` in `ml/config.py`, exported from
`ml/__init__.py`.

**Pitfalls / Alternatives:**
- VOC's official download mirror (`host.robots.ox.ac.uk`) is occasionally slow/unreachable; if
  `download=True` fails, fall back to a mirrored copy (e.g. via `kagglehub`, which already hosts a
  VOC dataset in this project's existing dependency) — check `kagglehub` search results for a VOC
  mirror before hardcoding a URL, keep the primary path as `download=True` since it's the
  canonical, most reproducible source.
- VOC's boundary/ignore label (255 in segmentation masks) must be excluded from the loss
  (`ignore_index=255`) and from `torchmetrics.segmentation.MeanIoU`'s `input_format`/ignore
  handling — silently including it as a 21st "class" would corrupt both loss and metric.
- `AutoAugment`/heavy classification-style augmentation (used in `create_imagenet_loaders`) is
  **not** appropriate here — geometric augmentations that don't have a defined effect on boxes/masks
  (color-only augmentations are fine and safe; geometric ones need the v2 box/mask-aware versions
  only). Keep augmentation intentionally lighter than the classification pipeline: horizontal flip
  + resize/letterbox + mild color jitter.

**Validation:** A `demo()`/assert-based smoke check — load one batch from each of the 4 loaders,
assert image shape `(B, 3, 256, 256)`, assert detection targets have `boxes` in `[0, 256]` and
`labels` in `[1, 20]`, assert segmentation masks have values in `{0..20, 255}`. Visualize 4 samples
with `torchvision.utils.draw_bounding_boxes`/`draw_segmentation_masks` and eyeball them once before
trusting the pipeline for a multi-hour training run — this is the single most valuable 10-minute
check in the whole phase, since a silently-misaligned box/mask transform would corrupt every
downstream mAP/mIoU number without raising any error.

---

## Task 2 — Backbone Feature-Map Adapter (`ml/det_seg_models.py`)

**What:** `DetSegBackbone(arch_name: str, feature_indices: list[int], extra_blocks: int = 2)` — a
wrapper `nn.Module` that takes any `MODEL_REGISTRY` classification backbone, strips its
GAP+classifier head, taps intermediate feature maps at specified `.features[i]` indices, and
appends `extra_blocks` SSDLite-style downsampling stages (reusing
`torchvision.models.detection.ssdlite._extra_block`, already confirmed depthwise-separable +
ReLU6, i.e. QAT-compatible per this project's existing rules) to build a 3–4 level feature pyramid.

**Why:** Every classification backbone in this codebase (`AlexNetBottleneck`, `AlexNetFire`,
`AlexNetTV`, and everything in `compensation.py`/`final_architecture.py`) ends in
`nn.AdaptiveAvgPool2d(1)` → `nn.Linear`, collapsing all spatial information — the opposite of what
detection/segmentation need. None of these backbones natively produce more than one useful spatial
resolution either (they're flat `nn.Sequential`s with 2 `MaxPool2d` stages, i.e. one native stride
level, ~stride 8 for the Bottleneck/Fire family at their existing channel-width design). SSD-style
multi-scale detection needs several strides (small anchors on high-res, early feature maps; large
anchors on coarse, late feature maps) — hence appending extra downsampling stages, exactly how
`torchvision.models.detection.ssdlite.SSDLiteFeatureExtractorMobileNet` builds its own pyramid on
top of MobileNetV3's backbone (confirmed via source read, Task-prep research).

**How:**
- Per-architecture `feature_indices`, hand-specified (matching this codebase's existing
  hand-indexed style for `fuse_map`, e.g. `configs`/`register_model(fuse_map=[["0","1"],...])`)
  rather than automated `torch.fx`-based extraction — simpler and more debuggable for 3 flat
  `nn.Sequential` backbones:
  - `alexnet_bottleneck`/`alexnet_fire`: tap after the 2nd `_AlexBottleneck`/`_FireModule` stage
    (post 2nd `MaxPool2d`, ~stride 8) and after the final stage before `AdaptiveAvgPool2d` (~stride
    8 still, same resolution — these backbones don't downsample further after stage 2; confirm
    exact index by counting `nn.Sequential` children once the module is instantiated, do not guess
    blindly in code).
  - `alexnet_tv` (torchvision `AlexNet.features`): tap after `features[2]` (post first
    `MaxPool2d`, stride 8) and after `features[12]` (final `MaxPool2d`, stride 32) — torchvision's
    stock AlexNet already spans a wider stride range than the Bottleneck/Fire family, which is
    itself a relevant Phase 7 observation (native pyramid richness differs by backbone, independent
    of kernel size).
  - Append `extra_blocks=2` `_extra_block`s after the deepest tapped feature map in all three cases
    so every backbone contributes the same *number* of pyramid levels to `SSDLiteHead`/`DefaultBoxGenerator`,
    even though their native strides differ — keeps the anchor config identical across backbones,
    isolating the backbone as the only varying factor.
- `forward(x) -> OrderedDict[str, Tensor]` — matches the interface `torchvision.models.detection.SSD`
  expects for its `backbone` argument (confirmed via `SSDLiteFeatureExtractorMobileNet.forward`
  source, Task-prep research: returns `OrderedDict([(str(i), v) for i, v in enumerate(output)])`).
- For segmentation, `DetSegBackbone` in "single-output mode" (`return_layers=[deepest_index]`) feeds
  `DeepLabHead`, which only needs one feature map (typically the stride-8 or stride-16 map, per
  DeepLabv3's convention of not going all the way to stride-32 to preserve spatial detail for
  per-pixel prediction).

**Inputs:** `arch_name` (must be a `MODEL_REGISTRY` key), the model's already-registered `ctor`.

**Outputs:** For detection: `OrderedDict` of 3–4 feature maps at increasing stride/decreasing
channel-resolution. For segmentation: single feature map at ~stride 8–16.

**Dependencies:** `MODEL_REGISTRY` must already have `alexnet_bottleneck`, `alexnet_fire`,
`alexnet_tv` registered (true today per `models/__init__.py` + Phase 3/1 notebooks).

**Deliverables:** `ml/det_seg_models.py` with `DetSegBackbone`, plus a small
`BACKBONE_FEATURE_CONFIG: dict[str, dict]` table (mirrors `fuse_map`'s per-arch hand-written style)
holding each backbone's tap indices and output channel counts (needed by `SSDLiteHead(in_channels=...)`
and `DeepLabHead(in_channels=...)`).

**Pitfalls / Alternatives:**
- If tap-index bookkeeping proves fragile across architectures (e.g. an off-by-one after editing a
  backbone later), fall back to `torchvision.models.feature_extraction.create_feature_extractor`
  (confirmed importable in the installed `torchvision==0.20.1`), which uses `torch.fx` symbolic
  tracing keyed on module *names* instead of list indices — more robust to internal refactors, at
  the cost of a dependency on `torch.fx` successfully tracing these modules (should work: all
  three backbones are static `nn.Sequential`s with no data-dependent control flow; the one thing to
  verify is that `torch.nn.quantized.FloatFunctional.add`/`.cat` — not used by `alexnet_tv`,
  used internally in some *other* Phase 3/4 models but not the three selected here — traces cleanly
  if a future backbone swap ever needs it).
- Channel-count mismatches between tapped backbone stages and `SSDLiteHead`'s expected
  `in_channels` list are a common, silent-until-runtime bug — validate with a single dry-run
  forward pass (`assert [f.shape[1] for f in backbone(dummy_input).values()] == expected_channels`)
  before wiring into the full `SSD` model.

**Validation:** `demo()` — instantiate `DetSegBackbone` for all 3 archs, forward a
`torch.randn(1, 3, 256, 256)`, assert the returned `OrderedDict` has the expected number of levels
and that spatial resolution strictly decreases level-to-level (catches a mis-ordered or duplicated
tap index immediately).

---

## Task 3 — Detection Model Assembly

**What:** `build_ssd_detector(arch_name: str, num_classes: int = 21) -> SSD` composing
`DetSegBackbone` + `DefaultBoxGenerator` + `SSDLiteHead` into a `torchvision.models.detection.SSD`
instance, following the exact pattern `torchvision`'s own `ssdlite320_mobilenet_v3_large` factory
uses internally (confirmed via source read).

**Why:** Reuse the full, tested SSD training/inference machinery (D4) rather than reimplementing
anchor matching, loss, and NMS.

**How:**
```
anchor_generator = DefaultBoxGenerator(
    aspect_ratios=[[2, 3]] * num_pyramid_levels,   # start from SSDLite320's defaults
    min_ratio=0.2, max_ratio=0.95,
)
backbone = DetSegBackbone(arch_name, feature_indices=..., extra_blocks=2)
num_anchors = anchor_generator.num_anchors_per_location()
head = SSDLiteHead(in_channels=BACKBONE_FEATURE_CONFIG[arch_name]["out_channels"],
                    num_anchors=num_anchors, num_classes=num_classes,
                    norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03))
model = SSD(backbone, anchor_generator, size=(256, 256), num_classes=num_classes, head=head)
```
(21 = 20 VOC classes + background, following the SSD/Faster-RCNN convention of a reserved
background class 0.)

**Inputs:** `arch_name`, `num_classes`. **Outputs:** an `SSD` `nn.Module` whose `forward()` in
training mode returns a loss dict (`{"bbox_regression": ..., "classification": ...}`) when given
`(images, targets)`, and returns decoded, NMS'd detections in eval mode — both behaviors come free
from `torchvision.models.detection.SSD`.

**Dependencies:** Task 2 (`DetSegBackbone`).

**Deliverables:** `build_ssd_detector()` in `ml/det_seg_models.py`.

**Pitfalls / Alternatives:**
- **Anchor coverage is a classic silent-failure mode**: if `DefaultBoxGenerator`'s scale/aspect
  configuration doesn't cover VOC's actual object-size distribution at 256×256, mAP silently caps
  far below what the backbone/head are capable of, and this looks identical to "the model is bad"
  in the results table. Before spending full training budget, run an anchor-recall sanity check
  (for every ground-truth box in a training sample, does *some* default anchor achieve IoU > 0.5?
  — a simple offline script, no training needed) and adjust `min_ratio`/`max_ratio`/`aspect_ratios`
  until recall is >95%. This check is cheap (~minutes) and its absence is the single most likely
  cause of a confusing, hard-to-debug low-mAP result across all three backbones simultaneously.
- Keep anchor configuration **identical across all 3 backbones** (only `in_channels` varies, driven
  by each backbone's actual channel widths) — this is what makes the mAP comparison a fair
  backbone ablation rather than a confounded anchor-tuning contest.

**Validation:** Forward one batch through each of the 3 assembled detectors in train mode, assert
the loss dict has finite, non-NaN values; in eval mode, assert output detections have well-formed
`boxes`/`scores`/`labels` shapes. Run the anchor-recall check described above before any real
training run.

---

## Task 4 — Segmentation Model Assembly

**What:** `build_deeplab_segmenter(arch_name: str, num_classes: int = 21) -> nn.Module` composing
`DetSegBackbone` (single-output mode) + `DeepLabHead`, with a final bilinear upsample back to
input resolution — mirrors `torchvision.models.segmentation.deeplabv3.DeepLabV3`'s
`_SimpleSegmentationModel.forward()` pattern (backbone → head → `F.interpolate(..., mode="bilinear",
align_corners=False)`).

**Why:** See D5. One feature map in, per-pixel logits out, at the same spatial size as the input.

**How:**
```
backbone = DetSegBackbone(arch_name, feature_indices=[deepest_index], extra_blocks=0)
head = DeepLabHead(in_channels=BACKBONE_FEATURE_CONFIG[arch_name]["deepest_channels"],
                    num_classes=num_classes)  # atrous_rates default (12, 24, 36)

class Segmenter(nn.Module):
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)["0"]
        logits = self.head(features)
        return F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
```
No extra downsampling blocks here (`extra_blocks=0`) — segmentation wants to stay at the
backbone's native stride (8–16), not push deeper into stride-32 territory, to preserve the spatial
resolution the final upsample has to reconstruct from.

**Inputs/Outputs:** image batch in, `(B, 21, 256, 256)` per-pixel logits out.

**Dependencies:** Task 2.

**Deliverables:** `build_deeplab_segmenter()` in `ml/det_seg_models.py`.

**Pitfalls / Alternatives:**
- ASPP's dilated convs (rate 12/24/36) at a 256×256 input with a stride-8 feature map (32×32
  spatial) push the *effective* dilation footprint close to or past the feature map's own size at
  the largest rate — a known DeepLab pitfall on small feature maps ("gridding artifacts", Chen et
  al. 2017 §4.3). If qualitative outputs look striped/gridded, reduce `atrous_rates` (e.g. `(6, 12,
  18)`, DeepLab's own suggestion for smaller output strides) rather than treating it as a training
  bug.
- Alternative if ASPP dominates latency in a way that muddies the Winograd/kernel-size story
  (H4): swap in a plain 3-stacked-3×3-conv decoder (no dilation) as a documented ablation — same
  parameter budget, isolates "does the decoder's conv type matter" from "does the backbone's kernel
  size matter." Not built by default; only worth the extra implementation time if Task 9's
  profiling shows ASPP is the dominant latency term (see H4).

**Validation:** Forward pass shape assertion (`(B, 21, 256, 256)`); qualitative check — overlay
predicted mask on 4 validation images before training even starts (with random-init weights, the
output will be noise, but this still catches shape/channel-order bugs immediately rather than
after a wasted training run).

---

## Task 5 — Training Infrastructure (`ml/det_seg_trainer.py`)

**What:** `DetectionTrainer(Trainer)` and `SegmentationTrainer(Trainer)` — subclasses overriding
only what differs (loss computation, the train/val step, the metric computed at `evaluate()`),
inheriting everything else (checkpointing, resume, W&B/TensorBoard logging, AMP, the epoch loop,
QAT epoch-callback hook, `benchmark()`) unchanged from `ml/trainer.py`.

**Why:** `Trainer._train_one_epoch`/`_validate`/`evaluate` are hard-coded to
`nn.CrossEntropyLoss(data, target)` and top-1/top-5 classification accuracy — structurally
incompatible with SSD's loss-dict return or DeepLab's per-pixel CE + mIoU. Subclassing (not a
from-scratch trainer) is the smaller diff: `fit()`'s checkpoint/resume/logging machinery
(`ml/checkpoint.py`, W&B config, the `make_qat_callback` hook) is task-agnostic and already
correct — rewriting it would be pure duplication risk for zero benefit. This also matches the
project's own stated principle ("Skip/resume logic lives in the notebook loop, not in a wrapper")
by keeping the *loop* shared and only the *step* specialized.

**How:**
- `DetectionTrainer` overrides `_train_one_epoch`: `loss_dict = model(images, targets); loss =
  sum(loss_dict.values())`, backward as normal (SSD's internal loss already combines box regression
  + classification correctly weighted — do not reweight manually). Overrides `evaluate()` to run
  `torchmetrics.detection.MeanAveragePrecision` over the val set instead of
  `MulticlassAccuracy` (see Task 7 for exact metric wiring).
- `SegmentationTrainer` overrides `_train_one_epoch`: `criterion = nn.CrossEntropyLoss(ignore_index=255,
  label_smoothing=cfg.label_smoothing)`, straightforward per-pixel loss on `(B, 21, H, W)` logits
  vs. `(B, H, W)` mask targets. Overrides `evaluate()` to compute `torchmetrics.segmentation.MeanIoU`.
- Both keep `fit()`, `benchmark()`, and the checkpoint/resume path entirely inherited — no override
  needed there since those operate on `model`/`optimizer`/`scheduler` generically, not on
  loss/metric specifics.
- LR schedule: reuse `TrainerConfig`'s `AdamW` + `CosineAnnealingLR` as-is; SSD/DeepLab training
  in the literature more commonly uses SGD+momentum, but staying with AdamW keeps this consistent
  with every other phase in the project and avoids introducing a second optimizer code path for a
  difference that is secondary to this study's actual question (kernel size / quantization, not
  optimizer choice). Note as a documented simplification, not an oversight.

**Inputs:** Assembled `SSD`/`Segmenter` model, the Task 1 data loaders, a `TrainerConfig` (reused
dataclass, no new config class needed — `epochs`, `lr`, `weight_decay`, `use_amp`,
`early_stopping_patience` all apply unchanged).

**Outputs:** Same `fit()` return shape as base `Trainer` (`best_val_top1` field is repurposed to
hold mAP or mIoU — document this field-name reuse clearly in the run summary rather than
introducing parallel field names that would break `make_run_summary`'s existing shape).

**Dependencies:** Tasks 1, 3, 4.

**Deliverables:** `ml/det_seg_trainer.py`.

**Pitfalls / Alternatives:**
- SSD's loss requires `targets` to always contain at least one box per image in some torchvision
  versions' matcher (an image with zero VOC objects in the crop after augmentation can crash the
  loss). Filter/resample empty-target images in the `Dataset.__getitem__` or `collate_fn`, or
  configure augmentation to guarantee at least one box survives (e.g. `RandomIoUCrop` from
  `transforms.v2`, which is designed exactly for this — it retries crops until a coverage
  constraint is met).
- `torchmetrics.detection.MeanAveragePrecision` accumulates all predictions/targets in memory
  across an epoch before computing (it's a "compute-at-the-end" metric, not a running average) —
  for a ~5K-image VOC07 test set this is fine, but do not call `.compute()` mid-epoch expecting
  cheap incremental results the way `MulticlassAccuracy.update()` behaves in the base `Trainer`.

**Validation:** One-epoch overfit test on a 10-image subset of each dataset — loss should drop
sharply and mAP/mIoU should rise well above random on that same 10-image set within a few epochs;
this is the standard "can the model even learn this data" sanity gate before a full run, and it's
inexpensive (minutes) relative to the cost of discovering a data/loss-wiring bug after a full
multi-hour training run.

---

## Task 6 — QAT Adaptation

**What:** Extend the Phase 1–6 QAT pipeline (`ml/quantization.py`'s `find_fuse_groups`,
`prepare_qat_model`, `build_qat_from_model`, `make_qat_callback`) to the assembled
detector/segmenter models, auditing every new module type introduced by the heads for
fbgemm-QAT compatibility.

**Why:** This project's QAT rules exist because specific ops silently break fbgemm QAT
(`AlexNetSE`'s Sigmoid caused a full training collapse per `TODO.md`'s "Known issues"). SSDLite and
DeepLabHead introduce op types (`ReLU6`, dilated `Conv2d`, `F.interpolate`) never exercised by
Phases 1–6's classification backbones — each needs an explicit compatibility check, not an
assumption of correctness by analogy.

**How, per new op type:**
- **`ReLU6`** (SSDLite's `_extra_block` and prediction-layer activations, confirmed via source
  read in D4): compatible — `ReLU6` fuses into `ConvBnReLU` patterns via
  `torch.ao.nn.intrinsic.qat` the same way plain `ReLU` does; fbgemm supports it natively. No
  special handling needed beyond the existing `inplace=False` rule (verify torchvision's internal
  `_extra_block`/`SSDLiteHead` construct `ReLU6(inplace=False)` — if not, the composed model will
  need child-module surgery analogous to `AlexNetTV`'s existing `_fix_relu_inplace` helper in
  `models/baselines.py`, which already exists precisely for this situation with a pretrained
  torchvision module).
- **Depthwise convs in SSDLite's extra blocks**: already established QAT-compatible in this
  codebase (`AlexNetDepthwiseSep`, Phase 3, full QAT support, `−2.92pp` drop — stable enough to
  ship, just not the *smallest* drop in the family). No new risk category.
- **Dilated `Conv2d` in ASPP** (`atrous_rates=(12,24,36)`): dilation is orthogonal to
  quantization — fbgemm's quantized `Conv2d` kernel supports a `dilation` parameter directly, no
  known incompatibility. This is a genuinely new op combination for this codebase to fuse/quantize,
  so it needs the same treatment as everything else: run `find_fuse_groups()` on the assembled
  `DeepLabHead` and inspect the returned fuse groups before trusting them automatically (ASPP's
  branches are `Sequential(Conv2d, BatchNorm2d, ReLU)` per branch, structurally identical to
  every other Conv-BN-ReLU triple this project already fuses — expected to "just work," verify with
  one test build rather than assuming).
- **`F.interpolate` (bilinear upsample)** in the segmentation head's final step: not a module, not
  quantized — it sits after `DeQuantStub()` in the QAT wrapping (final logits must be dequantized
  before upsampling/loss anyway, exactly how the classification backbones already dequantize before
  their final `Linear`). No new handling needed; just confirm `QuantStub`/`DeQuantStub` placement
  wraps only the conv-heavy backbone+head path, matching every existing model's `forward()` pattern.
- **SSD's `AnchorGenerator`/box-coder/NMS**: never touches quantized tensors — these operate on
  decoded, FP32 box coordinates downstream of the head's dequantized output. No QAT concern.
- Root-level `fuse_root_attr` for `register_model()`: since `DetSegBackbone` wraps the tapped
  backbone inside its own `.features`/`.extra` submodules (not the original backbone's top-level
  `.features` name), the `fuse_root_attr` passed to `register_model()` for these composed models
  needs to point at the *new* wrapper's structure, not reuse the original classification backbone's
  registry entry verbatim — register these as new, separate `MODEL_REGISTRY` entries
  (`ssd_alexnet_bottleneck`, `deeplab_alexnet_fire`, etc.) distinct from the classification-only
  `alexnet_bottleneck` entry, since they are structurally different modules even though they share
  a name-root.

**Inputs:** Assembled FP32-trained `SSD`/`Segmenter` models (Task 5 output).

**Outputs:** QAT-prepared models ready for `Trainer.fit()` with `make_qat_callback(...)`, exactly
following the existing `build_qat → fit(epoch_callback=...) → convert_to_int8` flow.

**Dependencies:** Tasks 3, 4, 5.

**Deliverables:** New `MODEL_REGISTRY` entries (registered in the Phase 7 notebook, matching this
codebase's existing convention of notebook-side `register_model()` calls rather than a central
`models/__init__.py` addition — consistent with how every prior phase's models are registered).

**Pitfalls / Alternatives:**
- **This is the task most likely to surprise**, precisely because it's the first time this
  codebase's QAT pipeline meets box-regression outputs. Box regression is a *continuous* output
  (unlike classification logits, where only the arg-max matters) — INT8 quantization noise on the
  final box-regression conv layer directly perturbs predicted box coordinates, which is a
  qualitatively different failure mode than classification's "occasionally flips the top-1 class."
  There is no existing precedent in this codebase for how much this matters; treat the FP32-vs-INT8
  mAP delta as a genuinely open empirical question this phase answers, not a formality to confirm.
  If INT8 mAP collapses disproportionately (e.g., classification-style ~1pp drops don't hold and
  mAP drops >10pp), the likely culprit is the final box-regression `Conv2d` needing to stay FP32
  (a documented, common practice: quantize the classification head but leave box regression in
  higher precision) — have this fallback ready rather than treating a large drop as a dead end.
- Fusing `SSDLiteHead`'s prediction layers: torchvision's stock `SSDLiteHead` may already be
  written assuming FP32 (its regression/classification heads are typically the last, un-fused
  1×1 convs in the reference implementation) — check whether it needs boundary `QuantStub`/`DeQuantStub`
  placement *around* just the box-regression branch specifically, separate from the classification
  branch, rather than one blanket quant boundary for the whole model.

**Validation:** After QAT `fit()` (even a short 2–3 epoch smoke run), assert
`torch.ao.quantization.convert()` succeeds without error and the INT8 model produces finite,
non-degenerate detections/masks on one validation image — the same category of check
`ml/quantization.py`'s existing `convert_to_int8()` implicitly assumes works, made explicit here
since it's untested on this op combination.

---

## Task 7 — INT8 Evaluation: mAP / mIoU

**What:** Wire `torchmetrics.detection.MeanAveragePrecision` and `torchmetrics.segmentation.MeanIoU`
into `DetectionTrainer.evaluate()`/`SegmentationTrainer.evaluate()` (Task 5), run FP32 and INT8
evaluation exactly like every prior phase's `trainer.evaluate(topk=(1,5))` step.

**Why:** mAP/mIoU are this phase's accuracy metrics — the direct analogue of Phases 1–6's top-1/top-5.

**How:**
```python
from torchmetrics.detection import MeanAveragePrecision
metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
# per batch: metric.update(preds, targets)  where preds/targets are lists of dicts
# {"boxes": ..., "scores": ..., "labels": ...} / {"boxes": ..., "labels": ...}
result = metric.compute()  # {"map", "map_50", "map_75", "map_small", ...}
```
```python
from torchmetrics.segmentation import MeanIoU
metric = MeanIoU(num_classes=21, include_background=False, per_class=False, input_format="index")
# per batch: metric.update(preds.argmax(1), masks)   # masks with 255 filtered/ignored beforehand
```
Both follow torchmetrics' standard `update()`-per-batch, `compute()`-at-epoch-end pattern — same
usage shape as `MulticlassAccuracy` already used in `ml/trainer.py`, so `DetectionTrainer`/`SegmentationTrainer`'s
`evaluate()` override is a small, structurally familiar diff from the base class.

**Inputs:** Model predictions (post-NMS detections for SSD's eval-mode `forward()`; per-pixel
argmax for the segmenter) and ground truth, per validation batch.

**Outputs:** `{"map", "map_50", "map_75", ...}` / `{"miou", "pixel_accuracy"}` dicts — feed
straight into `make_run_summary()`-style JSON, adding new fields
(`fp32_map`, `int8_map`, `map_drop`, or `fp32_miou`/`int8_miou`/`miou_drop`) alongside the existing
schema rather than repurposing the classification-specific `top1`/`top5` field names, to keep
`results/results_aggregate/results_cross_phase.csv`'s cross-phase schema honest about what was actually measured.

**Dependencies:** `pip install pycocotools torchmetrics` (or confirm/upgrade existing
`torchmetrics` install — see D6; **verify the exact installed version supports
`torchmetrics.segmentation.MeanIoU` before relying on it**, since that submodule is a newer
addition than `torchmetrics.classification`, which this codebase has used since Phase 1).

**Deliverables:** `evaluate()` overrides in `ml/det_seg_trainer.py`; `pycocotools`/`torchmetrics`
version pin added to `environment.yml`.

**Pitfalls / Alternatives:**
- `pycocotools` has a known history of finicky compilation on some platforms (a C extension) —
  if installation fails, `torchmetrics.detection.MeanAveragePrecision` supports a
  `backend="faster_coco_eval"` alternative (a pure-Python/Cython-free reimplementation) as a
  documented fallback; check `torchmetrics`'s installed-version docs for the exact `backend=`
  kwarg spelling at implementation time, API surface has changed across `torchmetrics` releases.
- mAP computation is memory-heavier than accuracy (it retains all predictions until `.compute()`);
  for the ~5K-image VOC07 test set this is not a concern on either the RTX 4060 or 4090, but note
  it if this pipeline is ever pointed at a larger eval set later.

**Validation:** Cross-check one model's mAP against a known-reasonable ballpark for a lightweight
SSD-family detector on VOC (published SSDLite/MobileNetV2 numbers on VOC07 are in the 60–70 mAP@.5
range at full training budget — this phase's smaller backbones/shorter budget will land lower, but
a result near 0 or above ~85 on a first successful training run is a red flag worth investigating
before trusting the full comparison table).

---

## Task 8 — Config, CLI, and Cluster Integration

**What:** `configs/detection.yaml`, `configs/segmentation.yaml` (mirroring `configs/data.yaml`'s
shape but for `DetSegDataConfig`), `configs/experiments/phase7_detection.yaml` /
`phase7_segmentation.yaml` (model list: the 3 `ssd_*`/`deeplab_*` registry entries from Task 6,
`stages: [fp32, qat, int8]`), and a new `scripts/train_det_seg.py` CLI driver.

**Why:** Every prior phase runs through `scripts/train.py --experiment <name> --runtime <local|pcad>`
(Phase 6 added `scripts/profile_hardware.py` as a parallel driver rather than overloading
`train.py` with a profiling-specific code path — same reasoning applies here: detection/segmentation's
data loading, loss, and metrics are different enough from classification that bolting a `--task`
flag onto `scripts/train.py` would tangle two different pipelines' branching logic through one
function, `run_experiment()`, which is already ~200 lines. A separate script reusing the same
config/runtime/cluster conventions is the smaller, more honest diff.

**How:** `scripts/train_det_seg.py` mirrors `scripts/train.py`'s structure exactly (`_load_yaml`,
`_maybe_init_wandb`, `_build_data_config`/`_build_trainer_config`, the same `runtime.yaml`
(`local`/`pcad`) loading, the same `_make_model_runs()` directory layout, the same
`create_results_summary`/CSV-append ending) — but calls `create_voc_detection_loaders`/
`create_voc_segmentation_loaders` (Task 1) instead of `create_imagenet_loaders`, and
`DetectionTrainer`/`SegmentationTrainer` (Task 5) instead of the base `Trainer`, selected via a
`--task {detection,segmentation}` flag (this flag is fine at the CLI-arg level, unlike inside
`run_experiment()`'s body, since it only ever dispatches to one of two otherwise-independent
functions — `run_detection_experiment()` / `run_segmentation_experiment()` — rather than
interleaving branches throughout one function body).

Cluster submission: extend `scripts/cluster.py`'s existing pattern (`profile-submit` subcommand,
added in Phase 6) with a `det-seg-submit` subcommand reusing `_build_sbatch_command()`, pointed at
a new `scripts/slurm/det_seg.sbatch` (a one-line copy of `profile.sbatch`/`train.sbatch`, calling
`python -m scripts.train_det_seg "$@"`). Reuse `configs/slurm/tupi_4090.yaml` as-is for GPU
selection — no new SLURM config needed.

**Inputs/Outputs:** Standard experiment-config → results-CSV flow, matching every prior phase.

**Dependencies:** Tasks 1, 5, 6, 7.

**Deliverables:** `configs/detection.yaml`, `configs/segmentation.yaml`,
`configs/experiments/phase7_detection.yaml`, `configs/experiments/phase7_segmentation.yaml`,
`scripts/train_det_seg.py`, `scripts/slurm/det_seg.sbatch`, `det-seg-submit` subcommand in
`scripts/cluster.py`.

**Pitfalls / Alternatives:** None beyond what Tasks 1–7 already surfaced — this task is
integration plumbing, low technical risk, mechanical given the Phase 6 precedent to copy from.

**Validation:** `python -m scripts.train_det_seg --experiment phase7_detection --runtime local
--dry-run` prints a resolved config without error (same smoke check `scripts/train.py --dry-run`
already supports); one short local run (2–3 epochs, `stages: [fp32]` only) completes end-to-end
before submitting the full PCAD job.

---

## Task 9 — Cross-Backbone, Cross-Phase, and Winograd-Latency Analysis

**Status: partially implemented, as a script instead of a notebook, with a narrower scope than
planned below.** `scripts/phase7_analysis.py` exists and does join Phase 3 classification to Phase
7 detection results and test H1–H4 — but it only prints to console (no `phase7_comparison.csv`
saved, no figures, no notebook-based visualizations as this task originally specified). It's also
currently blocked by the anchor-recall issue documented in `ideas/BEST_MODELS.md`'s Phase 7
section — any H1–H4 conclusions it prints are not yet trustworthy. If this task is picked back up,
follow the current repo convention: a notebook (if still wanted) would live under
`notebooks/phase_7_detection_segmentation_analysis/`, not `notebooks/analysis/`.

**What (original plan):** `notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb` — joins Phase 7's mAP/mIoU results to
Phase 3's classification accuracy and Phase 6's latency/Winograd-eligibility data (`model_details.csv`,
`final_comparison.csv`, Phase 6's `{device_tag}_profile.json`) on backbone name, to directly test
H1–H4.

**Why:** The scientific payoff of Phase 7 is cross-phase, not standalone — "does the classification
finding transfer" is a comparison question, not something a single phase's numbers alone answer.

**How:**
- H1/H2: bar charts of classification-top1 vs. detection-mAP vs. segmentation-mIoU per backbone,
  normalized to `alexnet_tv`'s value (ratio plot — directly shows whether the relative ranking
  holds); FP32-vs-INT8 drop comparison table across all three tasks per backbone.
- H3: the normalized-delta calculation specified in H3's "Evidence to Collect," visualized as a
  grouped bar chart (backbone × task, y-axis = accuracy relative to that backbone's own
  classification score) — makes "is this backbone disproportionately worse at localization" a
  direct visual read rather than requiring the reader to mentally normalize raw mAP against raw
  top-1.
- H4: reuse `ml/profiling.py`'s `profile_kernel_trace()`/`profile_model_latency()` (Phase 6, no new
  profiling code) on the 6 assembled Phase 7 models (3 backbones × {detection, segmentation} head),
  run once on whichever GPU is available locally (RTX 4060) — full RTX 4090/PCAD profiling is a
  stretch goal, not blocking, since H4's claim is about head-vs-backbone latency *share*, which is
  expected to be qualitatively consistent across the two GPU classes even if absolute numbers
  differ (Phase 6 already established the 4060/4090 comparison methodology; re-running it fully for
  6 new models is additional wall-clock time better spent validating H1–H3 first).
- Produce one consolidated `phase7_comparison.csv` (mirrors every prior phase's `final_comparison.csv`
  convention) and update `ideas/BEST_MODELS.md`/`TODO.md` with Phase 7's checked-off items and
  headline findings, following the exact pattern Phases 1–6 already use in `TODO.md`.

**Inputs:** `results/results_aggregate/model_details_cross_phase.csv`, `results/phase_4_compression_and_final_architecture_training/final_comparison.csv`,
Phase 6's profiling JSON outputs, Phase 7's own `phase7_detection_comparison.csv`/
`phase7_segmentation_comparison.csv` (Task 8 output).

**Outputs (as planned; not produced by the actual `scripts/phase7_analysis.py` implementation):**
Figures (`results/figures_generated/phase_7_detection_segmentation/phase7_*`),
`results/phase_7_detection_segmentation_analysis/phase7_comparison.csv`, updated
`TODO.md`/`ideas/BEST_MODELS.md` entries.

**Dependencies:** Tasks 1–8 complete with at least FP32+INT8 results for all 3 backbones × 2 tasks.

**Deliverables (as planned):** `notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb`.

**Pitfalls / Alternatives:** With only 3 backbones, any correlation/ranking statistic (Spearman ρ,
etc.) has very limited statistical power — report raw numbers prominently alongside any summary
statistic, and resist over-interpreting a ρ computed on n=3 as if it were a large-sample result
(explicitly flagged already in H2's acceptance criterion).

**Validation:** Every figure/table in this notebook must be traceable to a specific CSV/JSON file
on disk (no numbers computed inline and discarded) — matches the crash-safe,
reproducible-artifact convention every prior phase's analysis notebook already follows.

---

## BLOCKING ISSUES & REQUIRED FIXES

Must be resolved before committing to a full training run (mirrors `PHASE6_PLAN.md`'s
equivalent section):

### 1. Anchor coverage validation (BLOCKING)
Silent, hard-to-diagnose mAP ceiling if `DefaultBoxGenerator`'s scale/aspect config doesn't match
VOC's object-size distribution at 256×256. **Fix:** offline anchor-recall script (Task 3) run
*before* any full training job, for all 3 backbones (their differing native strides mean anchor
coverage could differ even with identical `DefaultBoxGenerator` settings).

### 2. Box-regression precision under INT8 (BLOCKING, open question)
No precedent in this codebase for how INT8 affects continuous regression outputs (Task 6). **Fix:**
treat the first FP32-vs-INT8 mAP comparison as a diagnostic run, not a final result — if the drop
is disproportionate (>5–10pp) compared to Phases 1–6's classification drops (<1–3pp typical),
apply the documented fallback (keep the box-regression head's final conv in FP32) before reporting
final numbers.

### 3. Data pipeline correctness (box/mask-transform alignment) (BLOCKING)
A geometric transform bug here doesn't crash — it silently trains against wrong boxes/masks and
produces a plausible-looking but meaningless mAP/mIoU number. **Fix:** the visualization spot-check
specified in Task 1's Validation step is mandatory, not optional, before any training run.

### 4. VOC download reliability (BLOCKING for reproducibility)
`host.robots.ox.ac.uk` is a single, sometimes-slow academic mirror; a broken download link months
from now would block reproduction the same way an un-versioned Kaggle dataset reference would.
**Fix:** document the exact VOC year/checksum used in this notebook's first cell, matching the
metadata-capture discipline Phase 6 established (`kind: metadata` JSON record) — capture dataset
version/checksum, not just "downloaded from torchvision," so a future re-run can verify it got the
same data.

### 5. `torchmetrics`/`pycocotools` version pinning (BLOCKING)
Confirmed at plan-writing time that `torchmetrics` is listed in `CLAUDE.md`'s Stack but was **not
found installed** in this machine's `.venv` when checked (`ModuleNotFoundError`) — whether this is
a stale local environment or a genuine gap, resolve it before Task 7: install/upgrade
`torchmetrics` to a version with `torchmetrics.segmentation.MeanIoU`, install `pycocotools`, and
add both with explicit version pins to `environment.yml`.

---

## MEDIUM-PRIORITY IMPROVEMENTS

Execute if time allows after the blocking path is validated; not required for a first
publishable result.

### 6. `vgg_style` as a 4th backbone
Adds the "small-kernel but *not* channel-compressed" cell to the 2×2 design implied by D3
(kernel-size × compression-technique) — currently Phase 7 only has 3 of that grid's interesting
cells. Same integration cost as any other backbone once Tasks 2–6's machinery exists (just a new
`BACKBONE_FEATURE_CONFIG` entry and 2 new `MODEL_REGISTRY` registrations).

### 7. Plain-3×3, non-dilated segmentation decoder ablation
Only worth building if Task 9's H4 analysis shows ASPP's dilated convs dominate end-to-end latency
in a way that obscures the backbone comparison (D5's noted trade-off).

### 8. Full RTX 4090/PCAD profiling of the 6 assembled Phase 7 models
Extends H4 from "confirmed on one GPU class" to "confirmed to hold across bandwidth-limited vs.
compute-rich hardware," matching Phase 6's own two-GPU methodology. Reuses `scripts/profile_hardware.py`'s
existing CLI/output-JSON conventions almost unchanged (Task 9) — mostly wall-clock cost, not new
engineering.

---

## REPRODUCIBILITY & VERIFICATION CHECKLIST

Before submitting any full training run:

- [ ] VOC dataset version/checksum captured in notebook metadata (Blocking #4).
- [ ] Data pipeline visualization spot-check passed (Blocking #3 / Task 1 Validation).
- [ ] Anchor-recall check passed (>95%) for all 3 detection backbones (Blocking #1 / Task 3 Validation).
- [ ] 10-image overfit smoke test passed for both detection and segmentation, all 3 backbones
      (Task 5 Validation).
- [ ] `find_fuse_groups()` output inspected (not just trusted) for both new head types — SSDLite
      and DeepLabHead (Task 6).
- [ ] QAT → INT8 conversion succeeds without error and produces non-degenerate output on a smoke
      test, before the full QAT training run is trusted (Task 6 Validation).
- [ ] `pycocotools`/`torchmetrics` versions pinned in `environment.yml` (Blocking #5).
- [ ] `phase7_detection.yaml`/`phase7_segmentation.yaml` `--dry-run` succeeds (Task 8 Validation).
- [ ] `phase7_comparison.csv` populated and cross-referenced against Phase 3/6 CSVs by backbone
      name before any headline claim is written into `TODO.md`/`ideas/BEST_MODELS.md` (Task 9).

---

## SCOPE & EFFORT

Unlike Phase 6 (inference-only, ~2.5–3 hours of pre-execution engineering on top of fully-reused
training infrastructure), Phase 7 builds new data, model, and training-loop layers from scratch.
Rough estimate, engineering time only (excludes actual GPU training wall-clock, which depends on
epoch budget — VOC is small enough that even 50–100 epochs per model is a few hours per run on
either GPU class):

- **Task 1 (data pipeline):** ~3–4 hours (VOC XML parsing, v2 transform wiring, collate_fn,
  visualization check).
- **Task 2 (backbone adapter):** ~2–3 hours (per-arch tap-index bookkeeping + validation).
- **Task 3 (detection assembly):** ~3–4 hours, including the anchor-recall validation script.
- **Task 4 (segmentation assembly):** ~1–2 hours (simpler than detection — one feature map, one head).
- **Task 5 (trainer subclasses):** ~2–3 hours.
- **Task 6 (QAT adaptation):** ~3–5 hours — flagged as the highest-uncertainty task (Blocking #2),
  budget contingency here first if anything overruns.
- **Task 7 (mAP/mIoU wiring):** ~1–2 hours, plus however long `pycocotools` installation friction
  costs (variable, sometimes zero, sometimes a real dependency-resolution fight).
- **Task 8 (config/CLI/cluster):** ~1–2 hours (mechanical, copies Phase 6's pattern).
- **Task 9 (analysis notebook):** ~2–3 hours.

**Total engineering estimate: ~2–3 working days**, before any GPU training wall-clock, and before
the medium-priority stretch items. This is a phase-sized (not profiling-sized) effort — schedule
it accordingly rather than expecting a single-session implementation.

**Manual/out of scope for tooling:** Nsight Compute deep-dive on the assembled detection/segmentation
models (same as Phase 6 — external CLI, run directly if H4's profiling result needs a
kernel-level explanation beyond `torch.profiler`'s trace).
