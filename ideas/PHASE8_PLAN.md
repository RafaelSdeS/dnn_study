# Phase 8 — Efficient Vision Transformers & Hybrid Attention Architectures (Implementation Plan)

**STATUS (2026-08-21):** Tasks 1–6 implemented, training submitted to PCAD; Task 7
(cross-phase analysis) not started, blocked on results. See `docs/PHASE8_LOG.md` for the
build history. One material deviation from this plan: **D6's QAT strategy for
`vit_tiny`/`deit_tiny` changed** — `swap_quantizable_mha()` cannot drive this codebase's
eager-mode `tq.prepare_qat()` (verified; PyTorch's custom-module conversion path crashes
before observers exist). Their `self_attention` is now excluded via `qconfig=None`, the
same treatment D6 already specified for Swin's `ShiftedWindowAttention` — see the amended
D6 section below and `ml/quantization.py`'s `exclude_attention_from_qat` docstring for the
full mechanism. `swap_quantizable_mha()` itself is still correct and kept in the codebase,
just not wired into the QAT path.

Phases 1–4 answered the kernel-restriction question entirely within the convolution paradigm
(shrink the kernel, compensate architecturally). Phase 8 asks a different question: is
convolution — at any kernel size — the right primitive at all, or can **local self-attention**
match small-kernel CNNs' accuracy/efficiency/quantization profile while sidestepping the
receptive-field problem Phase 3 spent an entire phase compensating for? This is the one
`TODO.md` phase that changes the *operator family*, not just the kernel size, so several of
Phase 8's decisions (QAT strategy, Winograd applicability, FLOP accounting) need new reasoning
that Phases 1–7 didn't require.

**Scope framing, stated up front:** the *data* half of Phase 8 is Phase-3-scale; the
*quantization* half is not. Classification stays 200-way on Tiny ImageNet-200 at 64×64 — zero
changes to `ml/data.py`, `ml/config.py`'s `DataConfig`, or `ml/trainer.py`'s core loop, and six
of the seven registry entries train through `scripts/train.py` unmodified. Phase 8 registers
**seven** models from **one** new file (`models/vit_variants.py`): `vit_tiny`, `deit_tiny`,
`swin_pico_{w2,w4,w8}`, `hybrid_bottleneck_swin`, `swin_pico_poolmixer`.

The QAT half is the real cost, and it is **larger than Phase 7's**, not smaller. Torchvision's
`VisionTransformer`/`SwinTransformer` carry no `QuantStub`/`DeQuantStub`, so `convert()` fails
outright on them (verified, see D6); under D6's mixed-precision design every FP32 island
(LayerNorm, attention core) additionally needs an explicit dequant/quant boundary. That is a
per-block `forward()` rewrite, not the one-helper `qconfig = None` walk an earlier draft of this
plan assumed. Budget Phase 8 at **Phase 7 scale or above** (see **SCOPE & EFFORT**), with Task 3
as the dominant line item.

> **Verification status (2026-08-17).** Every empirical claim below was checked against the
> installed stack (`torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`) and against
> `results/results_aggregate/model_details_cross_phase.csv`. Constructors were built and
> forwarded; the QAT path was smoke-tested end to end. Claims marked **(measured)** come from
> that check; claims marked **(unverified)** have not been run. An earlier draft of this plan
> contained wrong parameter counts, a wrong Phase 2 citation, a wrong compression baseline, a
> wrong fvcore diagnosis, and a QAT swap that silently corrupts weights — all corrected in place.

---

## Research Hypotheses

### H1: Local-Attention Window Size Is the Transformer Analogue of Conv Kernel Size
**Claim:** Swin-style windowed self-attention's `window_size` parameter plays the same role for
attention that kernel size plays for convolution — it caps the receptive field per layer — so
sweeping it should reproduce Phase 2's kernel-restriction accuracy curve (`2×2` ≪ `3×3`/`5×5` ≪
unrestricted) in attention terms.

**Expected Outcome:** FP32 top-1 accuracy increases monotonically with `window_size` ∈
`{2, 4, 8}`. Grid geometry (per D3, `patch_size=4` on a 64×64 input): stage 1 is a **16×16**
token grid, stage 2 (after one `PatchMerging`) is **8×8**. `window_size=8` is therefore global
attention *only at stage 2*; at stage 1 it still tiles the grid into 4 windows. It is the
least-restricted point of the sweep, not a literally-global baseline — state it that way in the
write-up.

**On the Phase 2 analogy — read the actual numbers first (measured).** The project's own
classification results do **not** show a clean small-kernel penalty at the point this hypothesis
originally cited:

| Phase 2 model | FP32 top-1 |
|---|---|
| `alexnet_2x2` | 30.02% |
| `alexnet_2x2_gap` | **33.15%** |
| `alexnet_tv` (Phase 1, unrestricted) | 32.89% |

`alexnet_2x2_gap` **beats** unrestricted `alexnet_tv`. (An earlier draft attributed 30.02% to
`AlexNet2x2GAP`; 30.02% is `alexnet_2x2`.) So H1 must not be framed as "reproduce Phase 2's
kernel-restriction penalty in attention terms" — that penalty is not unambiguously present in
Phase 2 once the GAP head is held fixed. Frame H1 as the standalone question it is: *does
capping the attention receptive field cost accuracy at all, on this dataset and at this scale?*
Attention's window covers its whole receptive field in one layer (no compounding through depth,
unlike stacked 3×3 convs per `AlexNetStacked`), so a small or absent penalty is an informative
result, not a null one — and it would be **consistent** with the Phase 2 GAP row above.

**Evidence to Collect:** FP32 top-1/top-5 per `window_size` value; per-window-size FLOPs/params
(windowed attention's compute scales linearly with number of windows, not quadratically with
total tokens — a structural efficiency Phase 2's kernel-restriction study didn't have an
analogue for, worth calling out explicitly in the write-up).

**Acceptance Criterion:** Spearman correlation between `window_size` and FP32 top-1 accuracy > 0.
Directional only (3 points), report raw numbers prominently (same caveat as Phase 7 H2).

---

### H2: Hybrid CNN Stem + Local Attention Beats Both Pure Paradigms at Matched Size
**Claim:** A small-kernel CNN stem (reusing Phase 3's Bottleneck/Fire compensation blocks) doing
the early spatial downsampling, followed by windowed-attention stages doing the later
global-context mixing, combines convolution's cheap local-pattern extraction with attention's
cheap-relative-to-its-receptive-field long-range mixing — beating both `alexnet_bottleneck`/
`alexnet_fire` (pure CNN, Phase 3 Pareto frontier) and the pure-ViT/Swin variants (Phase 8's own
H1) at a comparable parameter/FLOP budget.

**Expected Outcome:** Hybrid FP32 top-1 ≥ `alexnet_fire`'s 43.98% (Phase 3's cheapest Pareto
model) at ≤ 2× `alexnet_fire`'s **0.516M parameters**, and ≥ pure-Swin-Pico's accuracy at the same
window size used in its attention stages.

**⚠ Compare parameters, not `fp32_size_mb` (measured).** `fp32_size_mb` in
`model_details_cross_phase.csv` is `disk_mb()` of the **training checkpoint**, which carries
AdamW's two moment buffers alongside the weights — ≈11.5 bytes/param, not 4. `int8_size_mb` is
the converted model, ≈1 byte/param. The two columns use different conventions, so any
FP32-vs-INT8 or model-vs-model size claim stated in those MB figures is off by ~3×. Reference
points: `alexnet_fire` 0.516M params / 5.99 MB ckpt, `alexnet_bottleneck` 0.385M / 4.49 MB,
`vgg_style` 2.41M / 27.58 MB. Phase 8 reports **params and weights-only MB (`params × 4 B`)**
alongside the legacy columns so the comparison is like-for-like.

At the D3 configs this is already satisfied on the size side (measured): hybrid 0.281M params vs.
`alexnet_fire` 0.516M — the hybrid is *smaller* than both Phase 3 Pareto models, so H2 is a clean
accuracy question with the budget constraint already met, not a size trade-off.

**Evidence to Collect:** FP32 top-1, params, weights-only MB, FLOPs for the hybrid vs.
`alexnet_fire`/`alexnet_bottleneck` (`results/results_aggregate/model_details_cross_phase.csv`)
and vs. Phase 8's own pure-Swin variant.

**Acceptance Criterion:** Hybrid strictly dominates (higher accuracy, ≤ comparable size) at least
one of the two pure paradigms (pure-CNN or pure-attention) it's built from; report the trade-off
plot regardless of which direction it goes, this is the phase's central open question, not a
foregone conclusion.

---

### H3: Attention-Based Models Are Less Quantization-Robust Than Phase 3's Pareto CNNs
**Claim:** Because LayerNorm, softmax, and the internal attention matmuls must stay FP32 (see
Decision Record D6 — no stable INT8 path exists for them in this codebase's eager-mode fbgemm
QAT pipeline), attention-based models end up as **mixed-precision** models where only Linear/Conv
layers convert to INT8. This caps the achievable compression ratio and introduces FP32↔INT8
dequant/requant boundaries at every attention block — a structurally different (and likely worse)
quantization profile than Phase 3's fully-INT8-convertible Bottleneck/Fire. Phase 3's measured
accuracy change on INT8 convert: `alexnet_bottleneck` **−0.08pp**, `alexnet_fire` **+0.33pp**
(`quantization_drop_top1` = +0.084 / −0.328 in
`results/results_aggregate/model_details_cross_phase.csv`; that column is a *drop*, so its sign
is inverted relative to the accuracy-change figures quoted here — state which convention any
plot uses).

**⚠ Do not benchmark against the CSV's `compression_ratio` column (measured).** It is
`fp32_size_mb / int8_size_mb` where the numerator includes AdamW optimizer state and the
denominator does not (see H2), which inflates it ~3×: the CSV reports ≈**10.4×** for
`alexnet_bottleneck` and ≈**10.9×** for `alexnet_fire` (INT8/FP32 ratios 0.096 and 0.092), not
the ~4× a 32-bit→8-bit weight conversion can actually deliver. An earlier draft of this
hypothesis quoted "≈0.25 (4×)" — right about the physics, wrong as a citation of this project's
data. **H3 is therefore defined on weights only**: `theoretical_int8_MB / theoretical_fp32_MB`,
computed from parameter counts and per-module precision, for which a fully-INT8 model floors at
≈0.25 and Phase 3's Bottleneck/Fire sit at that floor.

**Expected Outcome:** the weights-only INT8/FP32 ratio for the Phase 8 models exceeds Phase 3's
≈0.25 floor by an amount proportional to how much of each model stays FP32; accuracy drop may be
small in absolute pp (Linear layers are usually quantization-tolerant) but the **efficiency**
story (size reduction per unit accuracy) is worse.

**Expected split, and it is not uniform across the seven models** — this is the substantive
prediction, and an earlier draft's blanket threshold obscured it. In a transformer block the MLP
holds ≈`8d²` params and attention's qkv+proj ≈`4d²`:
- **ViT/DeiT path** (`QuantizableMHA` swap, D6): qkv and proj *do* quantize, so ≈all params are
  INT8-eligible and the ratio lands near the ≈0.25 floor — i.e. **H3 is expected to be false for
  `vit_tiny`/`deit_tiny` on the size axis**, and their cost shows up as accuracy loss and
  dequant/requant boundary overhead instead.
- **Swin path** (whole-subtree `qconfig = None` on `ShiftedWindowAttention`, D6): attention's
  ≈1/3 of block params stay FP32, predicting a ratio ≈`0.33·1 + 0.67·0.25` ≈ **0.5**.

**Evidence to Collect:** weights-only FP32 vs. INT8 MB, FP32 vs. INT8 top-1, and a per-model
"quantized parameter fraction" (params inside INT8-eligible modules / total params) — this last
number directly explains *why* compression is capped, independent of accuracy. Report the legacy
`fp32_size_mb`/`int8_size_mb` columns too, flagged with the convention caveat, so Phase 8 rows
stay joinable to Phases 1–7.

**Acceptance Criterion:** the three Swin-derived models (`swin_pico_{w2,w4,w8}`) show a
weights-only INT8/FP32 ratio > 0.4; `vit_tiny`/`deit_tiny` are **predicted to fail that bar** and
that is the informative result, not a failed test. The real pass/fail here is whether the
quantized-parameter-fraction accounting reproduces the two predicted ratios above from the actual
converted models — i.e. this hypothesis tests the *accounting*, and separates the "principled but
harder" (MHA) from the "simple but coarser" (Swin) quantization path.

---

### H4: Distillation Recovers Most of the Data-Hungriness Gap on Tiny ImageNet-200
**Claim:** Vision Transformers are known to underperform CNNs on small/medium datasets without
either large-scale pretraining or distillation (Dosovitskiy et al. 2020, "An Image is Worth
16×16 Words," §4.5: ViT underperforms ResNets when trained from scratch on mid-sized data).
Tiny ImageNet-200's ~100K training images is far below the scale ViT was designed around.
DeiT (Touvron et al. 2021, "Training data-efficient image transformers & distillation through
attention") showed hard-label distillation from a strong CNN teacher substantially closes this
gap even without extra data. Distilling from Phase 1's best model (`mobilenetv2`, 57.99% top-1)
should measurably lift DeiT-Tiny above the plain (non-distilled) ViT-Tiny of the same
architecture trained identically otherwise.

**Expected Outcome:** DeiT-Tiny (with distillation) FP32 top-1 > plain ViT-Tiny FP32 top-1 by a
non-trivial margin (DeiT's own paper reports several points on ImageNet-1k; expect a comparable
or larger relative gain here given the dataset is smaller and the teacher/student gap is larger:
teacher `mobilenetv2` at 57.99% vs. an undistilled ViT-Tiny plausibly well below that).

**Evidence to Collect:** FP32 top-1 for ViT-Tiny (undistilled) vs. DeiT-Tiny (same architecture,
identical hyperparameters, distillation loss added) — the only variable is the loss function, so
this is a clean ablation, not a confound-prone architecture comparison like H1–H3.

**⚠ The teacher checkpoint does not exist (measured).** There is no `mobilenetv2_best.pth`
anywhere in the repo — `checkpoints/` contains only `phase_2_kernel_restriction_training/`,
`phase_3_compensation_and_hybrids_training/`, `phase_4_compression_and_final_architecture_training/`
and `final_architecture_phase4/`, and nothing matching under `outputs/`. `mobilenetv2`'s
`int8_top1` is `NaN` in the cross-phase CSV as well, consistent with the artifact never having
been retained. H4 therefore carries an **unbudgeted Phase 1 retrain** (`mobilenetv2`, ~58% top-1)
before any distillation run — GPU wall-clock not counted in SCOPE & EFFORT. Resolve one of:
(a) locate the checkpoint on PCAD and `rsync` it back, (b) retrain `mobilenetv2`, or (c) swap the
teacher to a model whose checkpoint *does* exist locally (e.g. `vgg_style`, 51.81%, or
`alexnet_bottleneck`, 44.62%) and restate H4's expected margin against the weaker teacher.
Decide before Task 4, not during it.

**Acceptance Criterion:** DeiT-Tiny FP32 top-1 − ViT-Tiny FP32 top-1 > 0. Directional (n=1 pair),
report the magnitude honestly rather than assigning it false statistical weight.

---

### H5: Attention Is Structurally Winograd-Incompatible — Even "Local" Attention
**Claim:** Winograd's F(2×2,3×3) transform (Phase 6, H1/H2) accelerates dense, `groups=1`,
stride-1 **convolutions** specifically because convolution has translation-invariant weight
reuse across spatial positions that the transform can exploit. Self-attention — global or
windowed — computes QK^T via batched matmul (GEMM), which has no such structure to transform;
"local" attention only shrinks the GEMM's operand size, it doesn't change the *algorithm*. So
regardless of window size, attention-heavy models should show near-zero Winograd kernel usage
in `profile_kernel_trace()` (Phase 6 infrastructure, reused unchanged), while the CNN
stem/patch-embedding portion of hybrid models remains exactly as Winograd-eligible as it was in
Phase 6's classification of the same conv types.

**Expected Outcome:** Winograd-attributed latency is <10% of total device time in
pure-ViT/Swin/lightweight-attention models — all their compute is Linear/matmul, not `groups=1`
3×3 conv. The hybrid model's Winograd-eligible latency share should be roughly proportional to
how much of its forward pass the CNN stem occupies (expected to shrink as more stages are
attention-based) — directly extending Phase 6's "small kernel ≠ Winograd-compatible" finding
(which was about depthwise convs) to a second, structurally different case (attention).

**⚠ This metric is not computable with Phase 6's code as written (measured).** Two corrections to
an earlier draft:
- **`winograd_speedup_info` does not exist.** It appears nowhere in `ml/` or `scripts/`; there is
  no "dual-signal detector". The only Winograd signal in `profile_kernel_trace()` is the single
  boolean `winograd_trace_detected`.
- **`profile_kernel_trace()` returns no per-kernel timings.** It returns
  `{"winograd_trace_detected": bool, "top_cuda_kernel": str|None, "op_timings": dict}`. The
  per-CUDA-kernel dict needed for a *latency share* is the local `kernel_timings`
  (`ml/profiling.py:429`), which is never returned — a bool cannot produce a "<10% of latency"
  number.

So H5 needs a **small, additive change to `ml/profiling.py`**, not "zero new profiling code": add
`kernel_timings` (and a derived `winograd_device_time_us` / `total_device_time_us` pair) to
`profile_kernel_trace()`'s return dict. Purely additive — existing keys and every Phase 6 call
site keep working. Fold this into Task 7 and its effort estimate.

**Evidence to Collect:** `profile_kernel_trace()` (with the return-dict extension above) and
`profile_model_latency()` on all seven Phase 8 models; per-module (stem vs. attention-stage)
latency breakdown via `torch.profiler(record_shapes=True)`, same technique as Phase 7's H4.

**Acceptance Criterion:** Winograd-attributed device time < 10% for pure-attention models
(ViT-Tiny, DeiT-Tiny, lightweight-attention variant); for the hybrid, the Winograd-attributed
share is strictly less than `alexnet_bottleneck`'s Phase 6-measured share (a pure-CNN model at
comparable depth), confirming attention stages dilute — never add to — Winograd eligibility.
Caveat carried over from Phase 6: cuDNN's Winograd kernel naming is not stable across versions,
so a zero share is weak evidence of absence; report the matched kernel names alongside the
percentage.

---

## Decision Records

### D1 — Dataset & Resolution: Tiny ImageNet-200, 64×64, Unchanged

Unlike Phase 7 (forced to 256×256 by VOC object scale), Phase 8 stays in classification, where
Phases 1–6's 64×64 convention is already sufficient signal (an 8×8 or 16×16 patch grid is enough
tokens for a meaningful attention experiment; going larger would only cost compute without
answering a different question about kernel-vs-attention trade-offs). Reuse
`create_imagenet_loaders(DataConfig(...))` and `ml/config.py`'s existing `DataConfig` **as-is**,
no new dataclass needed.

### D2 — Model Construction: `torchvision.models.vision_transformer`/`swin_transformer`, Not
Hand-Rolled Attention, Not `timm`

Confirmed present in the pinned `torchvision==0.20.1` (checked directly against the installed
`.venv`, not assumed from memory): `torchvision.models.vision_transformer.VisionTransformer`
(configurable `image_size`, `patch_size`, `num_layers`, `num_heads`, `hidden_dim`, `mlp_dim`, and
a `conv_stem_configs` argument for hybrid CNN-stem ViTs) and
`torchvision.models.swin_transformer.SwinTransformer` (configurable `patch_size`, `embed_dim`,
`depths`, `num_heads`, `window_size` — `window_size` is the exact "local attention window" knob
`TODO.md` names). Both are tested, widely used reference implementations — reusing them is the
same reasoning Phase 7's D4/D5 used for `torchvision.models.detection`/`segmentation` heads:
reimplementing multi-head attention, patch embedding, relative position bias, and shifted-window
masking from scratch is a large, error-prone undertaking for zero scientific benefit over using
the tested library version. `timm` (not installed, not in `environment.yml`) is **not** added as
a dependency — everything Phase 8 needs (a plain global-attention ViT for the DeiT-Ti base, and a
windowed-attention Swin for the local-attention variant) already ships inside torchvision.

Reference: Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale." ICLR 2021. Liu, Z. et al. "Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows." ICCV 2021.

### D3 — "Tiny/Pico" Sizing, Not torchvision's Stock Presets

torchvision's stock configs (`vit_b_16`: `hidden_dim=768`, 12 layers; `swin_t`: `embed_dim=96`,
`depths=[2,2,6,2]`, targeting 224×224 ImageNet-1k with 1.28M training images) are grossly
oversized for a 64×64, ~100K-image dataset — both in parameter count (would dwarf every Phase
1–4 model by 100–1000×, breaking size comparability) and in required token-grid depth (Swin's
4-stage design needs the grid to stay divisible by `window_size` at every stage; a 64×64 input
with `patch_size=4` only supports 2 `PatchMerging` stages before the grid drops below any useful
window size). Custom, explicitly-sized configs, hand-picked to keep parameter count in the
same order of magnitude as Phase 3/4's Pareto-frontier models (4–30 MB) rather than chasing
`ViT-Tiny`/`Swin-T`'s literal published hyperparameters, which were tuned for a 3,500× larger
dataset:

**All parameter counts below are measured**, by constructing each model and forwarding a
`(2,3,64,64)` batch — not estimated. An earlier draft's estimates (~5.5M for ViT-Tiny, ~1–2M for
Swin-Pico) were roughly 2× high and 4× high respectively. "Weights MB" is `params × 4 B`;
"ckpt MB" is the ≈11.5 B/param training-checkpoint figure that `fp32_size_mb` actually reports
(see H2), given here only so these rows can be read against the existing CSV.

| Model | Config | Params (measured) | Weights MB | ckpt MB | Notes |
|---|---|---|---|---|---|
| ViT-Tiny (H1 baseline / DeiT-Ti base) | `image_size=64, patch_size=8, hidden_dim=192, mlp_dim=768, num_heads=3, num_layers=6` | **2.758M** | 10.52 | ~30.3 | 64 patches + 1 cls token = 65-token sequence. `num_layers=6`, not DeiT-Ti's 12 — halved given the smaller dataset (ViT is known to *overfit*, not underfit, at this data scale; depth is the more likely lever to cut before width, matching the parameter-efficiency intuition Phase 3's Bottleneck already demonstrated for CNNs). Comparable to `vgg_style` (2.41M) |
| Swin-Pico (H1 window sweep) | `patch_size=[4,4], embed_dim=48, depths=[2,2], num_heads=[2,4], window_size=[w,w]` for `w ∈ {2,4,8}` | **0.321M** (w2) / **0.322M** (w4) / **0.324M** (w8) | 1.23 | ~3.5 | 2 stages only: 16×16 → 8×8 token grid (a 3rd `PatchMerging` stage would drop the grid to 4×4, too small for any `window_size>4` sweep point). Comparable to `alexnet_bottleneck` (0.385M) |
| Hybrid (H2) | `_AlexBottleneck` conv stem (`models/compensation.py`), 3 stride-2 stages 3→32→64→96, → 8×8×96 → 2 `SwinTransformerBlock` (`window_size=4`, provisional pending H1) → LayerNorm → GAP → Linear | **0.281M** | 1.07 | ~3.1 | Smaller than both Phase 3 Pareto models. See Task 1 |
| Lightweight attention (H5 cross-check) | Swin-Pico architecture with `ShiftedWindowAttention` replaced by a parameter-free pooling or depthwise-3×3-conv token mixer | **0.228M** | 0.87 | ~2.5 | Pooling variant measured; the depthwise variant will be larger. See D5/Task 1 |

Note the ~8.5× parameter spread between ViT-Tiny (2.758M) and the pool-mixer (0.228M). That is
*not* a size-matched comparison, so H1 (a sweep within Swin-Pico, all three within 1% of each
other) and H2 (hybrid vs. Phase 3 CNNs, all within 2×) are the size-controlled hypotheses;
any ViT-vs-Swin statement is confounded by scale and must be reported on an accuracy-vs-params
plot rather than as a head-to-head accuracy claim.

Every config is deliberately picked, not tuned via a search — this phase compares *architecture
family*, not a hyperparameter-optimized instance of each; note this limitation explicitly in
final reporting (a fully-tuned ViT might close some of the CNN gap further).

### D4 — Local Attention Window Sweep: `{2, 4, 8}` Patches, Not `{3×3, 5×5}`

`TODO.md` phrases the window size as "3×3 or 5×5 patches." `SwinTransformer`'s `PatchMerging`
halves the grid at each stage, and `window_size` must evenly divide the token grid at every
stage it's applied to — power-of-2 windows (`2, 4, 8`) are the values that divide a
power-of-2 grid (`16×16` → `8×8`) cleanly at every stage without padding logic. All three values
build and forward correctly at 64×64 (measured). `window_size=8` is full/global attention **only
at the 8×8 stage-2 grid**; at stage 1's 16×16 grid it still tiles into 4 windows — so it is the
sweep's least-restricted endpoint, *not* a globally-unrestricted baseline. An earlier draft
overstated this as "degenerates to full/global attention"; the write-up must not claim Phase 8
has a true global-attention Swin control. Document this substitution
explicitly as a deliberate implementation choice driven by Swin's architectural constraint, not
a deviation from the spirit of the TODO item — `window_size=2` (a 2×2-token, i.e. 8×8-pixel
receptive field at `patch_size=4`) is the closest feasible analogue to "3×3 pixel patches" the
architecture supports without hand-modifying `PatchMerging`/attention masking to support
non-power-of-2 grids (not worth the engineering cost for a sweep that already spans the
restrictive-to-unrestrictive range with 3 clean points).

### D5 — Lightweight Attention Variant: Pooling/Depthwise Token-Mixer, Not Linear Attention

`TODO.md` names two options: "linear attention or depthwise attention." Linear attention
(Katharopoulos et al. 2020, "Transformers are RNNs") approximates softmax attention with a
kernel trick to get O(n) instead of O(n²) complexity — a real technique, but it doesn't change
the *op type* (still matmul-based, still Winograd-irrelevant per H5) and adds a second novel,
harder-to-verify-correct implementation on top of everything else Phase 8 already introduces.
**Chosen: depthwise/pooling token-mixer**, following Yu et al. 2022 ("MetaFormer Is Actually
What You Need for Vision," CVPR) — replace `ShiftedWindowAttention`'s token-mixing sub-layer
with a parameter-free 3×3 average-pooling operator (PoolFormer's own choice) or a depthwise 3×3
conv (a documented MetaFormer variant), keeping every other block component (LayerNorm, residual,
MLP) identical to Swin-Pico. This is chosen specifically because it creates a clean, decisive
cross-check for H5: a depthwise-conv token-mixer uses **the exact op category Phase 6's H2
already profiled and found non-Winograd-eligible** (`groups=in_channels`, no dense cross-channel
GEMM to accelerate) — so Phase 8's H5 result for this variant is a directly falsifiable
prediction from Phase 6's own data, not a fresh, unanchored claim. This scientific coherence is
worth more here than linear attention's novelty.

Reference: Yu, W. et al. "MetaFormer Is Actually What You Need for Vision." CVPR 2022.

### D6 — QAT Strategy: Quantize Linear (MLP + patch-embed Conv) Only; Attention Submodule and
All LayerNorms Stay FP32 (Whole-Subtree Exclusion)

**AMENDED (2026-08-21):** the `nn.MultiheadAttention → torch.ao.nn.quantizable.MultiheadAttention`
swap this section originally called a "strict improvement" over Swin's fallback (below) does
**not** work with this codebase's `tq.prepare_qat()`, verified by direct testing, not by reading
docs: `torch.ao.nn.quantizable.MultiheadAttention` is registered in PyTorch's default
`observed_to_quantized_custom_module_class` mapping, and `prepare_qat()`'s first internal step
(`convert()`, which runs *before* `prepare()` attaches any observers) matches on it and calls its
`.from_observed()` classmethod immediately — `AttributeError: 'Linear' object has no attribute
'activation_post_process'`. Setting `qconfig=None` on the swapped module avoids that crash but
also stops `prepare()`/`convert()` from recursing into its own children, so its Linears never
quantize either — the swap buys nothing under this constraint (it would need FX graph-mode
quantization, a different API than the rest of this codebase, to actually work). **Resolution:**
`vit_tiny`/`deit_tiny`'s `self_attention` gets the *same* fallback as `ShiftedWindowAttention`
below — plain `nn.MultiheadAttention`, excluded via `qconfig=None`
(`exclude_attention_from_qat`, extended to match it) — and `models/vit_variants.py`'s
`_QuantizableEncoderBlock` no longer brackets it with Quant/DeQuant stubs (that bracketing
assumed a quantized-input-capable MHA). Net effect: H3's "mixed-precision, capped compression"
prediction now applies uniformly to all seven Phase 8 models, not just the Swin-derived ones —
arguably a cleaner, more comparable result than the original two-tier plan. The rest of this
section is kept as-is below for the reasoning that led here (Swin's fallback was correct on the
first pass; only the ViT/DeiT "strict improvement" half didn't survive contact with
`tq.prepare_qat()`).

This is the single most consequential new decision Phase 8 introduces, and it required checking
this codebase's actual QAT internals (`ml/quantization.py`), not assuming by analogy to Phases
1–7 (all pure Conv-BN-ReLU).

> **Two hard blockers found by smoke-testing this design end to end (measured).** Both invalidate
> an earlier draft's claim that D6 costs "one small helper function". Read these before Task 3.
>
> **(a) Torchvision's ViT/Swin have no `QuantStub`/`DeQuantStub`, so `convert()` fails.** Every
> model in `models/` wraps its forward in stubs (`compensation.py:51-52`, `baselines.py:39-43`);
> torchvision's transformer classes do not. Running the full
> `swap → exclude → prepare_qat → convert → forward` sequence on both:
> ```
> vit :  NotImplementedError: Could not run 'quantized::linear'      with arguments from the 'CPU' backend
> swin:  NotImplementedError: Could not run 'quantized::conv2d.new'  with arguments from the 'CPU' backend
> ```
> A quantized layer received an FP32 tensor because nothing quantized the input. Fixing this needs
> a wrapper module per architecture carrying `self.quant`/`self.dequant` — and, because D6 is
> *mixed* precision by design, a `DeQuantStub`→FP32 island→`QuantStub` boundary around **every**
> `LayerNorm` and **every** `ShiftedWindowAttention`. That is a `forward()` rewrite per block
> type, not a `for m in model.modules(): m.qconfig = None` walk. It is the dominant cost of
> Phase 8.
>
> **(b) `torch.ao.nn.quantizable.MultiheadAttention` is numerically wrong at `batch_first=True`**
> in the installed `torch==2.5.1` — which is exactly the mode torchvision's `EncoderBlock` uses
> (`nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)`).
> With identical weights, no quantization and no dropout:
> ```
> batch_first=True  → max |output difference| = 0.946189
> batch_first=False → max |output difference| = 0.000000
> ```
> The ViT/DeiT path in this decision record is therefore **broken as specified** until the
> quantizable module is driven in sequence-first layout. Fix: wrap it in a small adapter that
> transposes `(B,S,E) → (S,B,E)` on the way in and back on the way out, constructing the
> quantizable module with `batch_first=False`. Verify with the equivalence check in Task 3 before
> anything else. Do **not** take the earlier draft's "train the swapped module from random init"
> fallback — that trains a module with wrong attention semantics and hides the bug rather than
> fixing it.

**What was checked:** `torch.ao.quantization.prepare_qat()` only inserts fake-quant observers on
`nn.Module` instances that (a) have an explicit `.qconfig` set (inherited from the nearest
ancestor unless overridden) and (b) appear in `torch.ao.quantization`'s default QAT module
mapping (`Conv2d`, `Linear`, and their fused variants — confirmed these are the only two types
this codebase's `ml/quantization_advanced.py` already treats as `_QUANTIZABLE`). Bare functional
ops inside a module's `forward()` (`F.softmax`, the `@` matmul in attention, `nn.LayerNorm`'s
functional normalization) are **never** touched by `prepare_qat`/`convert` regardless of
`qconfig`, because they aren't registered `nn.Module`s with a quant-aware replacement — this is a
structural property of PyTorch's eager-mode quantization, not a gap specific to this project.

**Consequence for the two attention implementations Phase 8 uses:**
- **`nn.MultiheadAttention`** (used internally by `torchvision.models.vision_transformer.EncoderBlock`,
  confirmed via source read — this is what ViT-Tiny/DeiT-Ti use): PyTorch ships a purpose-built
  drop-in replacement, `torch.ao.nn.quantizable.MultiheadAttention` (confirmed importable in the
  installed `torch==2.5.1`), which decomposes the internal QKV projection and output projection
  into separately-observable `nn.Linear` calls with explicit `QuantStub`/`DeQuantStub` boundaries
  around the softmax/matmul core — the *documented* PyTorch pattern for quantizing attention
  (`torch.ao.nn.quantizable` module docstring). **Even with this swap, the softmax/scaled-dot-product
  core itself stays FP32** — the quantizable module only makes the surrounding Linears
  observable, consistent with widely-reported findings that attention-internal activations
  (post-softmax, post-LayerNorm) have long-tailed, outlier-heavy distributions that static/QAT
  INT8 handles poorly (Bondarenko et al. 2021, "Understanding and Overcoming the Challenges of
  Efficient Transformer Quantization," EMNLP). Use this swap for ViT-Tiny/DeiT-Ti's attention
  blocks — it is a strict improvement in quantized parameter coverage over the fallback below —
  **subject to blocker (b) above**: it must be constructed `batch_first=False` behind a transpose
  adapter, and it must be built via `QuantizableMHA.from_float(...)`, never a raw
  `load_state_dict`. Measured key overlap between the two modules:
  ```
  nn.MultiheadAttention : in_proj_weight, in_proj_bias, out_proj.{weight,bias}
  QuantizableMHA        : in_proj_weight, in_proj_bias, out_proj.{weight,bias},
                          linear_{Q,K,V}.{weight,bias}
  load_state_dict(..., strict=False) → missing: linear_Q/K/V.{weight,bias}
  ```
  `strict=False` therefore leaves Q/K/V projections **at random init and reports no error**
  (measured output difference 0.88) — precisely the silent corruption this plan warns about.
  `from_float()` slices `in_proj_weight` into the three linears correctly, but requires
  `other.qconfig` to be set first (`assert hasattr(other, "qconfig")`) and returns an
  already-`prepare`d module — see Task 3 for both consequences.
- **`ShiftedWindowAttention`** (Swin's hand-rolled windowed attention, confirmed via source read:
  `qkv`/`proj` are bare `nn.Linear`, but the windowing/masking/relative-position-bias/softmax math
  is all *functional* code inside `forward()`, not decomposed into separate modules the way
  `torch.ao.nn.quantizable.MultiheadAttention` decomposes standard MHA): there is no drop-in
  quantizable replacement for this module in `torch.ao.nn.quantizable`. **Fallback: set
  `module.qconfig = None` on the entire `ShiftedWindowAttention` instance** — this is a documented,
  standard `torch.ao.quantization` mechanism (setting `qconfig = None` on a module excludes its
  entire subtree from `prepare_qat`, leaving it FP32, and `convert()` leaves it as an ordinary
  FP32 submodule inside the otherwise-INT8 model). Its internal `qkv`/`proj` Linears do **not**
  get quantized under this fallback (correctly reflected in H3's expected mixed-precision-ratio
  accounting) — a real accuracy/efficiency cost, stated explicitly rather than glossed over, and
  itself informative: it shows the accuracy/compression trade-off of "principled but harder"
  (MHA path) vs. "simple but coarser" (Swin path) quantization within the very models this phase
  studies.
- **What *does* get quantized in every Phase 8 model**: the MLP block's two `nn.Linear` layers
  (`fc1`/`fc2`, no softmax/attention math involved — ordinary Linear-GELU-Linear, ripe for the
  same standard fbgemm QAT path every prior phase used for `Linear` classifier heads), any
  `nn.Conv2d` in the patch-embedding stem or hybrid model's CNN stem (unchanged from Phase 1–4
  treatment), and `PatchMerging`'s internal `nn.Linear` (Swin's downsampling projection).
- **What never gets quantized in any Phase 8 model**: every `nn.LayerNorm` (also excluded via
  `qconfig = None` — LayerNorm's normalization statistics are computed per-sample at inference
  time, unlike BatchNorm's frozen running stats, and PyTorch's eager-mode QAT has no fused
  quantized LayerNorm counterpart in this codebase's mapping), and — per the two bullets above —
  either the decomposed-but-still-partially-FP32 MHA core, or the wholly-FP32
  `ShiftedWindowAttention` subtree.

**Why not skip QAT/INT8 for Phase 8 entirely** (an alternative considered): `TODO.md` explicitly
asks for FP32/QAT/INT8 comparison and quantization-robustness assessment for every Phase 8
variant, matching every prior phase's pipeline structure — reporting only FP32 results would
leave H3 untestable and break the phase's comparability to Phases 1–7. Mixed-precision QAT,
correctly labeled as such, is the standard way this trade-off is reported in the transformer
quantization literature, not a workaround specific to this codebase.

**What genuinely needs no change, and what does.** The `qconfig` *exclusion mechanism* is sound
and needs no library work (verified): `prepare_qat_model` deep-copies, sets `model.qconfig` on
the root (`ml/quantization.py:51`), then calls `tq.prepare_qat`, and PyTorch's
`propagate_qconfig_` honours an explicitly-set child `qconfig` — including `None` — instead of
overwriting it from the parent. So setting `qconfig = None` on submodules *before*
`build_qat_from_model()` does exclude their subtrees, exactly as designed. `prepare_qat_model`,
`build_qat_from_model`, `convert_to_int8` and `make_qat_callback` need no edits.

What the earlier draft got wrong is the amount of **model-side** code required. Phase 8 needs:
1. A per-architecture wrapper adding `QuantStub`/`DeQuantStub` at the model boundary (blocker (a)).
2. Dequant/quant boundaries around every FP32 island — each `LayerNorm`, each
   `ShiftedWindowAttention` — which means overriding the relevant `forward()`s, not just tagging
   modules. This is the bulk of Task 3.
3. A `batch_first` transpose adapter around `QuantizableMHA` (blocker (b)).
4. The `exclude_attention_from_qat()` / `swap_quantizable_mha()` helpers as originally scoped —
   now the *small* part of the job.

`make_qat_callback`'s `freeze_bn_stats` call is a documented no-op for BatchNorm-free models
(Phase 8 uses LayerNorm throughout except in the hybrid's CNN-stem Conv-BN pairs) — harmless, no
special-casing needed, but worth a one-line note in the notebook so it isn't mistaken for a bug.

**Fallback if Task 3 overruns:** report Phase 8 FP32-only and mark H3 as not-run, rather than
shipping a mixed-precision number whose boundaries were never verified. That is a worse
scientific outcome than the plan intends but a better one than a silently-wrong INT8 table; take
this decision explicitly at the Task 3 checkpoint, not by drift.

### D7 — FLOPs/Params Accounting: Verify `fvcore`'s Attention Coverage Before Trusting It

`ml/reporting.py`'s `compute_flops()` calls `fvcore.nn.FlopCountAnalysis` with
`unsupported_ops_warnings(False)` (`ml/reporting.py:159`) — this **suppresses**, rather than
fixes, the fact that some ops go uncounted. There *is* a real undercount for Phase 8, but it is
in the opposite place from where an earlier draft of this plan predicted.

**Measured** — `analysis.unsupported_ops()` on each model at `(1,3,64,64)`:

| Model | Reported MACs | Uncounted ops |
|---|---|---|
| `vit_tiny` | 0.176 G | `aten::scaled_dot_product_attention` ×6, `gelu` ×6, `mul` ×25, `add` ×13, `div` ×6, `unflatten` ×6 |
| `swin_pico_w4` | 0.032 G | `softmax` ×4, `gelu` ×4, `pad` ×5, `mul` ×12, `add` ×14, `pow` ×4, `rsub` ×8, `fill_` ×18, `sub` ×2, `ne` ×2 |

Corrections to the earlier draft, both material:
- **Swin is fine.** `ShiftedWindowAttention`'s QK^T and softmax·V go through `torch.matmul`, and
  fvcore *does* have a matmul handler — no FLOP-bearing op is missing from the Swin list above
  (only elementwise/normalisation ops, which every phase of this project already omits).
- **ViT is the one that undercounts.** `EncoderBlock` routes through
  `F.scaled_dot_product_attention`, which fvcore has no handler for, so **the entire QK^T +
  softmax·V cost of all 6 layers is missing** from that 0.176 G.

So the fix targets `aten::scaled_dot_product_attention`, on the ViT/DeiT path only — *not*
`aten::matmul` on the Swin path. Register a handler via fvcore's documented extension mechanism,
`FlopCountAnalysis.set_op_handle(op_name, handler_fn)`, returning
`2 · num_heads · seq_len² · head_dim` MACs per call (QK^T plus softmax·V; standard transformer
FLOP-counting convention, e.g. Kaplan et al. 2020 §2.1). Then verify against a hand-computed
figure (Task 5) rather than silently reporting an undercounted number in the final comparison
table. Gate the handler so Phase 1–7's existing FLOPs numbers are untouched.

---

## Task 1 — Model Architectures (`models/vit_variants.py`)

**What:** Four constructors — `vit_tiny()`, `swin_pico(window_size)`, `hybrid_bottleneck_swin()`,
`swin_pico_poolmixer()` — each returning an `nn.Module` ready for `MODEL_REGISTRY`. These yield
**seven** registry entries: `swin_pico` is registered three times (w2/w4/w8) for the H1 sweep, and
`deit_tiny` reuses `vit_tiny()`'s constructor under a distillation training recipe (H4), so it is
a seventh entry with no seventh constructor.

**Why:** See Decision Records D2–D5. All four wrap or lightly modify already-installed
torchvision classes, following the project's existing `models/*.py` convention (`baselines.py`
already wraps torchvision's `AlexNet`/`VGG`/`ResNet18`/`MobileNetV2` — Phase 8's file follows the
exact same "thin wrapper + explicit config" pattern for consistency).

**How:**
- `vit_tiny(num_classes=200)`: `torchvision.models.vision_transformer.VisionTransformer(image_size=64,
  patch_size=8, num_layers=6, num_heads=3, hidden_dim=192, mlp_dim=768, num_classes=200)`,
  per D3's config table. Confirm `_process_input`'s hard `torch._assert(h == self.image_size, ...)`
  matches the actual loader output (64×64) — a shape mismatch here is a loud, immediate crash, not
  a silent bug, low risk.
- `swin_pico(window_size)`: `torchvision.models.swin_transformer.SwinTransformer(patch_size=[4,4],
  embed_dim=48, depths=[2,2], num_heads=[2,4], window_size=[window_size, window_size],
  num_classes=200)`. Register three `MODEL_REGISTRY` entries (`swin_pico_w2`, `swin_pico_w4`,
  `swin_pico_w8`) for the H1 sweep — a `functools.partial`-style factory per window size, matching
  this codebase's existing per-variant-registration convention (Phase 2's per-kernel-size AlexNet
  variants are separate registry entries, not one parameterized constructor called at train time).
- `hybrid_bottleneck_swin()`: compose `_AlexBottleneck` stages (imported from
  `models.compensation`, already used as a plain `nn.Module` building block by
  `models/final_architecture.py` — confirmed reusable pattern) down to an 8×8 feature map, then
  feed into 2 `SwinTransformerBlock` instances (imported directly from
  `torchvision.models.swin_transformer`, confirmed a public class) with `window_size=4` (the H1
  sweep's presumed sweet spot — confirm against H1's actual result before finalizing this choice;
  flag as provisional in code with a one-line comment). No separate `PatchMerging`/patch embedding
  needed — the CNN stem already produces a spatially-downsampled grid, each spatial location
  becomes one "token" directly (channel dimension = `hidden_dim`), avoiding a redundant second
  patchification.
  **Layout: `SwinTransformerBlock` consumes and returns `(B, H, W, C)`, not `(B, C, H, W)`**
  (measured) — the stem's output needs `.permute(0, 2, 3, 1)` going in, and the head pools over
  `dim=(1, 2)`. Easy to get wrong silently, since a transposed tensor of the right rank will
  still forward and merely mix the wrong axes. Reference shape that builds and forwards
  (0.281M params, measured): stem `3→32→64→96` as three stride-2 `_AlexBottleneck` blocks
  (64×64 → 8×8), then 2 blocks at `dim=96, num_heads=4, window_size=[4,4]`, alternating
  `shift_size` `[0,0]` / `[2,2]`, then `LayerNorm(96)` → GAP → `Linear(96, 200)`.
- `swin_pico_poolmixer()`: same `SwinTransformer`-derived structure as `swin_pico`, but with
  `ShiftedWindowAttention` replaced by a custom `_PoolMixer(nn.Module)` (3×3 average pool, or
  depthwise 3×3 conv — implement both, keep whichever trains more stably, document the choice) per
  D5. This is a **constructor argument swap, not a monkeypatch or copy**:
  `block=partial(SwinTransformerBlock, attn_layer=_PoolMixer)`. Both `SwinTransformer`'s `block`
  parameter and `SwinTransformerBlock`'s `attn_layer` parameter are confirmed present, and the
  whole composition builds and forwards to `(2, 200)` at 0.228M params (measured).
  `_PoolMixer.__init__` must accept the signature `SwinTransformerBlock` calls `attn_layer` with —
  `(dim, window_size, shift_size, num_heads, attention_dropout=..., dropout=...)` — even though it
  ignores most of them, and its `forward` takes and returns `(B, H, W, C)`. Follow PoolFormer in
  subtracting the input (`pool(x) - x`) so the block's residual add composes correctly.

**Inputs:** None beyond `num_classes=200` (matches every existing `MODEL_REGISTRY` constructor's
zero-config, defaults-from-`CLAUDE.md`-convention signature).

**Outputs:** Four constructors → seven registry entries (3-way window sweep plus `deit_tiny`,
per **What** above) `nn.Module`
classifiers, each `forward(x: (B,3,64,64)) -> (B,200)` logits — identical I/O contract to every
existing `MODEL_REGISTRY` entry, so nothing downstream (`Trainer`, `reporting.py`,
`quantization.py`) needs to know these are attention-based.

**Dependencies:** None beyond installed `torchvision==0.20.1` (confirmed).

**Deliverables:** `models/vit_variants.py`, updated `models/__init__.py` export list.

**Pitfalls / Alternatives:**
- `VisionTransformer`'s positional embedding is a learned parameter sized to the exact
  `(image_size, patch_size)` pair at construction time — there is no interpolation path needed
  here since Phase 8 never changes resolution after construction (unlike fine-tuning a
  pretrained-at-224 ViT to a new resolution, a common but irrelevant-here complication).
- `SwinTransformer`'s `window_size` must evenly divide the token grid at **every** stage it's
  applied to, not just the first — with only 2 stages (`depths=[2,2]`) and starting grid 16×16 →
  8×8, `window_size=8` only works if applied starting from the 8×8-grid stage; verify
  `SwinTransformer`'s per-stage window handling doesn't require the *same* `window_size` to divide
  the *first* stage's 16×16 grid too (it does, since `window_size` is a single value applied
  across all stages in torchvision's implementation) — `window_size=8` divides both 16 and 8
  cleanly, so no issue for the D4 sweep values `{2,4,8}`, but this constraint is worth a
  one-line assertion in the model constructor (`assert 16 % window_size == 0`) so a future
  edit to `depths`/`patch_size` fails loudly instead of silently miscomputing windows.
- `_AlexBottleneck` **is** cleanly reusable as a stem — its signature is
  `_AlexBottleneck(in_ch, out_ch, stride=1, reduction=4)` with no dependence on
  `AlexNetBottleneck`'s stage sequence (verified; the hybrid reference shape above uses it
  directly). The earlier draft's concern about hardcoded channel counts is unfounded; no fallback
  stem is needed.
- All Conv/BN inside `_AlexBottleneck` already use `ReLU(inplace=False)` and `bias=False`, i.e.
  they already satisfy this project's QAT architecture rules — one less thing to adapt.

**Validation:** `demo()`/assert-based self-check per constructor — forward a
`torch.randn(2, 3, 64, 64)`, assert output shape `(2, 200)`, assert a finite, sane parameter
count. **Sanity bound: 0.15M–5M** (an earlier draft said 0.5M–15M, which would have rejected
four of the seven models — the pool-mixer at 0.228M, the hybrid at 0.281M, and both Swin
variants at ~0.32M). Assert each model's measured count against D3's table within a few percent,
which catches an `embed_dim`/`hidden_dim` typo far more tightly than a wide range does. Run for
all seven registrations before training any of them.

**Status: all seven constructors verified building and forwarding to `(2, 200)`** at the D3
configs, with the parameter counts in D3's table (measured). Task 1 carries no known risk; the
risk is concentrated in Task 3.

---

## Task 2 — `MODEL_REGISTRY` Integration & Fuse-Map Wiring

**What:** `register_model(name, ctor, fuse_map=..., fuse_root_attr=..., lr=...)` calls for all
**seven** Phase 8 registry entries (`vit_tiny`, `deit_tiny`, `swin_pico_w2`, `swin_pico_w4`,
`swin_pico_w8`, `hybrid_bottleneck_swin`, `swin_pico_poolmixer`), in the Phase 8 notebook
(matching this codebase's existing convention of notebook-side registration — Phase 7's D-series
decisions already established this pattern for new phases, not a central `models/__init__.py`
addition). Mirror them into `ml/model_registrations.py` as well, per `CLAUDE.md` — that file is
what `scripts/train.py` reads, so Task 6's CLI path does not work without it.

**⚠ Only `lr` is honoured from registry metadata (measured).** `register_model` accepts arbitrary
`**metadata`, but `scripts/train.py:186` reads exactly one key:
`replace(trainer_cfg, lr=spec.get("lr", trainer_cfg.lr))`. A `register_model(weight_decay=...)`
override is accepted silently and then **ignored** — which matters because Task 4's optimizer
mitigation depends on it. Either extend that line to pull `weight_decay` (and any warmup key) too,
or set the transformer hyperparameters in `configs/experiments/phase8.yaml` instead.

**Why:** Every downstream tool (`Trainer`, `build_qat`, `convert_to_int8`, `make_run_summary`,
`compute_flops`) is keyed off `MODEL_REGISTRY`, exactly as in every prior phase.

**How:** `fuse_map` for each model covers **only** the Conv-BN(-ReLU) pairs in that model's CNN
portion (patch-embedding stem for ViT/Swin variants — `conv_proj`/`PatchMerging`, if
BN-containing; the hand-built `_AlexBottleneck` stem for the hybrid) — run `find_fuse_groups()`
(existing, unmodified) scoped to just that submodule and inspect its output before trusting it
(same discipline Phase 7's Task 6 applied to `DeepLabHead`'s ASPP branches), rather than assuming
it transfers correctly to a novel module tree. Attention/MLP/LayerNorm portions contribute **no**
entries to `fuse_map` — they're handled entirely by Task 3's `qconfig = None` exclusion pass, not
by fusion (LayerNorm has no BN-fusion analogue; Linear-GELU-Linear has no fusable BN either).

**Inputs:** Task 1's constructors.

**Outputs:** Seven live `MODEL_REGISTRY` entries.

**Dependencies:** Task 1.

**Deliverables:** Registration cells in
`notebooks/phase_8_efficient_vit/vit_qat_phase8.ipynb` (see Task 6),
plus the mirrored `ml/model_registrations.py` entries. Note the path: this project stores
notebooks under `notebooks/phase_N_<topic>[_<purpose>]/`, so a bare `notebooks/vit_qat_phase8.ipynb`
(as an earlier draft specified) breaks the convention every other phase follows. Phase 8's
training and analysis notebooks share one `phase_8_efficient_vit/` folder rather than
separate `_training`/`_analysis` folders (the split every other multi-notebook phase uses) —
a deliberate exception, not a naming-convention regression.

**Pitfalls / Alternatives:** If `torchinfo.summary()` (used by every prior phase's per-model
reporting step) doesn't cleanly print a `VisionTransformer`/`SwinTransformer`'s layer table (some
third-party summary tools mishandle attention modules' non-standard `forward()` signatures) — a
plausible but **unverified** risk; check it in Task 1's validation step rather than discovering it
mid-training-run. Raw `sum(p.numel() ...)` is the fallback and is what D3's measured counts use.

**Validation:** `MODEL_REGISTRY[name]["ctor"]()` builds without error for all seven entries;
`find_fuse_groups()`'s output for each model's stem is manually inspected (printed, read, sanity
checked against the actual module tree) at least once before the first training run. Note only
the hybrid has Conv-BN pairs to fuse — the pure ViT/Swin variants' patch embeddings are bare
`Conv2d`/`Linear` with no BatchNorm, so their `fuse_map` is legitimately empty.

---

## Task 3 — QAT Adaptation: Quant Stubs, Mixed-Precision Boundaries, and the MHA Swap
*(the phase's dominant cost — see D6's two confirmed blockers)*

**What:** the mixed-precision QAT plumbing from D6. This is **four** pieces of work, not one
helper — an earlier draft scoped only the last of them:
1. `QuantStubWrapper` — a per-architecture wrapper adding `QuantStub`/`DeQuantStub` at the model
   boundary (D6 blocker (a)).
2. FP32-island boundaries — `DeQuantStub`/`QuantStub` pairs around every `LayerNorm` and every
   `ShiftedWindowAttention`, which requires overriding the enclosing `forward()`s.
3. `BatchFirstMHAAdapter` — a transpose wrapper so `QuantizableMHA` runs sequence-first
   (D6 blocker (b)).
4. `exclude_attention_from_qat()` and `swap_quantizable_mha()` — the small tagging/swap helpers.

**Why:** D6 established that neither this repo nor stock `torch.ao.quantization` handles
LayerNorm/attention exclusion or transformer stub placement automatically, and that torchvision's
transformer classes cannot be `convert()`ed as shipped. Items 1–3 are the price of that; item 4
is the part that really is trivial.

**How:**
```python
def exclude_attention_from_qat(model: nn.Module) -> nn.Module:
    """Set qconfig=None on LayerNorm and any hand-rolled attention submodule (D6).

    Call BEFORE build_qat_from_model(). Leaves Conv2d/Linear elsewhere untouched —
    they still inherit the model-level qconfig set inside prepare_qat_model().
    Verified: prepare_qat's propagate_qconfig_ honours an explicitly-set child
    qconfig (including None) rather than overwriting it from the parent.
    """
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, ShiftedWindowAttention)):
            module.qconfig = None
    return model


class BatchFirstMHAAdapter(nn.Module):
    """Drive a sequence-first QuantizableMHA from batch-first callers (D6 blocker (b)).

    torch 2.5.1's quantizable MultiheadAttention returns wrong values when built
    with batch_first=True (measured: 0.95 max abs error vs. nn.MultiheadAttention
    at identical weights, 0.0 when built batch_first=False). Build it seq-first and
    transpose around it instead.
    """
    def __init__(self, qmha: nn.Module):
        super().__init__()
        self.qmha = qmha

    def forward(self, q, k, v, need_weights=False, **kwargs):
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        out, w = self.qmha(q, k, v, need_weights=need_weights, **kwargs)
        return out.transpose(0, 1), w


def swap_quantizable_mha(model: nn.Module, qengine: str = "fbgemm") -> nn.Module:
    """Replace nn.MultiheadAttention with the quantizable variant (D6, ViT/DeiT path only).

    Uses from_float(), NOT load_state_dict(strict=False): the quantizable module has
    separate linear_Q/K/V submodules that do not exist in the float module's state dict,
    so strict=False leaves them at random init and reports nothing (measured).
    from_float() slices in_proj_weight into the three linears correctly, but asserts
    the source module has a .qconfig, hence the assignment below.
    """
    import torch.ao.quantization as tq
    from torch.ao.nn.quantizable.modules.activation import MultiheadAttention as QuantizableMHA
    for name, child in model.named_children():
        if isinstance(child, nn.MultiheadAttention):
            seq_first = nn.MultiheadAttention(
                child.embed_dim, child.num_heads,
                dropout=child.dropout, batch_first=False,
            )
            seq_first.load_state_dict(child.state_dict())  # strict: same class, keys must match
            seq_first.qconfig = tq.get_default_qat_qconfig(qengine)
            setattr(model, name, BatchFirstMHAAdapter(QuantizableMHA.from_float(seq_first)))
        else:
            swap_quantizable_mha(child, qengine)
    return model
```
Call order in the Phase 8 notebook: `model = load_best_model(...)` → wrap in the
stub-carrying module (item 1/2) → (ViT/DeiT only) `swap_quantizable_mha(model)` →
`exclude_attention_from_qat(model)` → `build_qat_from_model(model, arch_name, device)`
(unchanged) → `fit(epoch_callback=make_qat_callback(...))` (unchanged) → `convert_to_int8(...)`
(unchanged).

**Inputs:** FP32-trained model (Task 4's `fit()` output).

**Outputs:** QAT-prepared model, mixed-precision by construction (D6), ready for the existing
`fit()`/`convert_to_int8()` calls unchanged.

**Dependencies:** Task 1, Task 2.

**Deliverables:** `exclude_attention_from_qat()`, `swap_quantizable_mha()`,
`BatchFirstMHAAdapter` added to `ml/quantization.py`; stub-carrying wrapper modules in
`models/vit_variants.py` (they are architecture-specific, so they belong with the models, not in
the quantization module).

**Pitfalls / Alternatives:**
- The two D6 blockers (missing stubs; `batch_first` numerics) are the substance of this task.
  Neither is hypothetical — both were reproduced. Do not start Task 4 until both checks below pass.
- **`from_float()` returns an already-prepared module.** Its last two lines are
  `observed.eval()` and `torch.ao.quantization.prepare(observed, inplace=True)`, so the returned
  object carries live observers/fake-quant. Two consequences: (i) a naive pre-swap/post-swap
  `torch.allclose` comparison fails *even on a correct swap* — the earlier draft's mandated
  Blocking #1 check could not distinguish success from failure, and would have been "resolved" by
  disabling it; (ii) the swap must happen before `prepare_qat`, and you should confirm the
  double-prepare is a no-op on the already-prepared subtree rather than assuming it.
- The `nn.MultiheadAttention → nn.MultiheadAttention(batch_first=False)` copy in
  `swap_quantizable_mha` uses **strict** `load_state_dict`: same class, so the keys must match
  exactly, and a failure there should be loud. Only the float→quantizable step needs `from_float`.
- **Unverified:** whether `tq.prepare_qat` and `tq.convert` traverse a `QuantizableMHA` nested
  inside `BatchFirstMHAAdapter` correctly (torch maps quantizable→quantized MHA through
  `DEFAULT_STATIC_QUANT_MODULE_MAPPINGS`, but nesting it under a custom parent has not been tested
  here). Check this on a 2-layer ViT before committing to the 6-layer one.

**Validation** (both mandatory, in this order):
1. **`batch_first` equivalence.** Build `nn.MultiheadAttention` and the adapter-wrapped
   `QuantizableMHA` from the same weights, put both in eval, call
   `qmha.apply(torch.ao.quantization.disable_fake_quant)` to take observers out of the numeric
   path, and assert `torch.allclose(a, b, atol=1e-5)`. Expected max abs error is 0.0 — anything
   near 0.9 means the transpose adapter is missing or bypassed. Without `disable_fake_quant` this
   assertion fails for a *correct* swap; that is the trap described above.
2. **End-to-end convert.** `prepare_qat → forward → convert → forward` on a 2-layer version of
   each architecture, asserting a `(B, 200)` output. This is the check that catches missing
   stubs; as of now it **fails** for both ViT and Swin (D6 blocker (a)) and is the definition of
   done for this task.

---

## Task 4 — Training: FP32, QAT, INT8 (Reuse `Trainer` As-Is, Plus One Distillation Subclass)

**What:** For `vit_tiny`, `swin_pico_{w2,w4,w8}`, `hybrid_bottleneck_swin`, `swin_pico_poolmixer`:
standard `Trainer.fit()` → `build_qat` → `fit(epoch_callback=make_qat_callback(...))` →
`convert_to_int8` → `evaluate(topk=(1,5))`, identical call sequence to every prior phase. For
`deit_tiny` (H4): the same architecture as `vit_tiny`, but trained via a new
`DistillationTrainer(Trainer)` subclass.

**Why:** Six of seven Phase 8 models are ordinary 200-way classifiers — the base `Trainer` already
handles this exactly, no subclassing needed (confirmed: `Trainer._train_one_epoch`'s
`nn.CrossEntropyLoss(data, target)` call and `evaluate()`'s `MulticlassAccuracy` are
architecture-agnostic, they only assume `model(images) -> (B, num_classes)` logits, which every
Phase 8 model satisfies). Only DeiT's hard-label distillation loss requires new code, and per
Phase 7's established precedent (subclass only the step, not the loop), this is a minimal diff.

**How:**
```python
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher: nn.Module, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval().to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.alpha = alpha

    def _train_one_epoch(self, model, optimizer, scaler, criterion):
        # Signature matches the base class (ml/trainer.py:313) and MUST return the same
        # 3-tuple it does: (train_loss, train_acc, avg_grad_norm | None) — fit() unpacks
        # three values at ml/trainer.py:123.
        # Override only the loss computation; keep the base class's AMP/grad-clip/logging
        # scaffolding by calling into the same structure it uses (see ml/trainer.py L313-344,
        # AMP/grad-clip branch at L323-339).
        # loss = (1-alpha) * CE(student_logits, labels)
        #      +    alpha  * CE(student_logits, teacher(images).argmax(dim=1))   # hard distillation
        ...
```
Hard-label distillation (student's cross-entropy against the teacher's *argmax* prediction, not a
soft KL term) — the DeiT paper's own finding that hard distillation outperforms the more common
soft-KL approach for this exact setup (Touvron et al. 2021, Table 4). This also avoids needing a
temperature hyperparameter, one fewer knob to tune. `alpha=0.5` (equal weight, DeiT's reported
default) as the starting point; treat as a config value (`DistillationConfig` or a plain kwarg,
not a new dataclass given it's a single float) rather than hardcoding.

Teacher: `load_best_model("mobilenetv2", MODEL_REGISTRY["mobilenetv2"]["ctor"], SAVE_DIR, device)`
— reuses Phase 1's already-trained checkpoint (per `CLAUDE.md`'s Model Inventory, `mobilenetv2`
is Phase 1's best result at 57.99% top-1) — **no new teacher training required**. The checkpoint
actually lives at `outputs/pcad/archive_legacy_phases/phase_4_5_large_scale/mobilenetv2/checkpoints/
mobilenetv2_best.pth` (verified on disk), a different `SAVE_DIR` than Phase 8's own runs will use
— point `load_best_model()` at that path explicitly rather than assuming it's colocated with
Phase 8's checkpoints, and confirm it loads without error before writing the notebook cell that
depends on it.

**Inputs:** Task 1/2/3 outputs; Phase 1's `mobilenetv2` checkpoint (distillation only).

**Outputs:** Same `fit()` return shape (`best_val_accuracy`, `history`, etc.) as every prior
phase; same `{name}_best.pth`/`qat_{name}_best.pth`/`{name}.pth` checkpoint file naming.

**Dependencies:** Tasks 1–3; an existing `mobilenetv2_best.pth` checkpoint from Phase 1.

**Deliverables:** `DistillationTrainer` in `ml/trainer.py` (or a new `ml/distillation_trainer.py`
if keeping `ml/trainer.py` from growing an architecture-specific subclass is preferred —
minor style call, either is a small, contained addition).

**Pitfalls / Alternatives:**
- ViT/Swin models are known to need different optimization hyperparameters than CNNs in the
  literature (AdamW with warmup + cosine decay, higher weight decay, gradient clipping — DeiT's
  own recipe uses `lr=5e-4` with a 5-epoch linear warmup, `weight_decay=0.05`, far from this
  project's CNN-tuned defaults of `lr=3e-4` and `weight_decay=4e-4` in `ml/config.py`
  (`configs/training.yaml` overrides the latter to `5e-4`; the two disagree, worth reconciling
  while you are here), with no warmup. The base `Trainer` already uses `AdamW` (`ml/trainer.py:72`)
  with `CosineAnnealingLR` (`:73`), so only warmup and the hyperparameter values are missing.
  Reusing the defaults unchanged risks slow/unstable convergence purely from an optimizer
  mismatch, which would be mistaken for an architectural finding.
  **Mitigation, corrected:** `TrainerConfig.warmup_epochs` **already exists**
  (`ml/config.py:36`, documented as "linear LR warmup before cosine decay; 0 disables") but
  `ml/trainer.py` never reads it — `fit()` constructs `CosineAnnealingLR` unconditionally, so the
  field is currently dead. The work is to *wire the existing field* (`LinearLR` + `SequentialLR`,
  both stdlib `torch.optim`) and add `warmup_epochs` to `configs/training.yaml`, not to add a new
  config knob. Separately, add a per-model `lr`/`weight_decay` override via
  `register_model(lr=..., weight_decay=...)`'s `**metadata` kwargs (already used by
  `alexnet_fp32.yaml` for a per-model `lr` override, per `CLAUDE.md`'s Key Patterns). Note
  `weight_decay` is only half-wired today: `register_model()` will happily store it, but
  `scripts/train.py` currently only reads `spec.get("lr", ...)` back out — reading `weight_decay`
  the same way is a small, required addition to `scripts/train.py`, not zero new code.
- `use_amp=True` (this project's FP32-training default) interacts with LayerNorm/softmax
  numerics differently than with BatchNorm/ReLU — AMP is generally safe for transformers (it's
  the standard training regime in the literature) but watch for any NaN/inf loss in the first few
  epochs, a known-if-uncommon AMP+attention failure mode, and fall back to `use_amp=False` for
  that specific model if it occurs (same "document the workaround" discipline as
  `AlexNetSE`'s existing precedent).

**Validation:** Standard 2–3 epoch smoke run per model (same discipline as every prior phase's
"local test run before PCAD submission") before committing to a full training budget; for
`DistillationTrainer` specifically, assert the distillation loss term is non-zero and the
teacher's parameters do not change after a training step (`requires_grad_(False)` verification) —
a forgotten-frozen-teacher bug would silently degrade into training two models jointly with no
distillation benefit.

---

## Task 5 — FLOPs/Params Verification (D7)

**What:** A one-time verification script/notebook cell comparing `ml/reporting.py`'s
`compute_flops()` output against a hand-computed attention-FLOPs formula for one Phase 8 model
(`vit_tiny`, the simplest case — standard `nn.MultiheadAttention`, no windowing complexity).

**Why:** See D7 — an uncaught FLOPs undercount would silently corrupt every efficiency comparison
(`param_efficiency_top1_per_m`, MACs-based plots) this phase produces relative to Phases 1–4.

**How:** Manual formula: for one attention layer, `2 · num_heads · seq_len² · head_dim` MACs for
QK^T plus the same for the softmax-weighted value sum (`seq_len=65` for `vit_tiny` including the
cls token, `head_dim = hidden_dim / num_heads = 64`), summed across all 6 layers, plus the MLP
block's `2 · seq_len · hidden_dim · mlp_dim` MACs per layer (ordinary Linear FLOPs, `fvcore`
already handles these correctly) — compare the attention-only component against
`FlopCountAnalysis`'s reported total minus the MLP/Linear-only component computed the same way.
Per D7 the attention component **is** missing for `vit_tiny` (measured: reported total 0.176 G,
with `aten::scaled_dot_product_attention` ×6 uncounted), so register a handler for that op:
```python
def sdpa_flop_handle(inputs, outputs) -> int:
    """MACs for QK^T + softmax·V. Shapes come from the jit graph; read them off
    inputs[0] (query) rather than hardcoding, so this stays correct if the config changes."""
    ...
analysis.set_op_handle("aten::scaled_dot_product_attention", sdpa_flop_handle)
```
**Do not** register an `aten::matmul` handler: fvcore already counts matmul, so adding one
double-counts Swin's attention (whose QK^T/softmax·V go through `torch.matmul` and are already
in its 0.032 G). Confirm with `analysis.unsupported_ops()` per model before adding any handler —
Swin's uncounted list contains no FLOP-bearing op and needs no patch at all.

**Inputs:** `vit_tiny()` constructed instance.

**Outputs:** A verified `compute_flops()` call path (either confirmed already-correct, or patched
via a custom op handle) that the rest of Phase 8's reporting can trust without re-checking.

**Dependencies:** Task 1.

**Deliverables:** A short verification cell in the Phase 8 notebook, plus (if needed) a
`compute_flops`-adjacent helper in `ml/reporting.py` registering the custom `fvcore` op handle for
attention models specifically (gated so it doesn't affect Phase 1–7's existing FLOPs numbers).

**Pitfalls / Alternatives:** `analysis.unsupported_ops_warnings(False)`
(`ml/reporting.py:159`) silences the warning but **not** the diagnostic —
`FlopCountAnalysis.unsupported_ops()` still returns the full uncounted-op dict without
re-enabling the flag. Call it first, per model; it names the ops directly and removes the need to
guess (that is how D7's table was produced). Note this also means every Phase 1–7 FLOPs number in
this project omits elementwise/normalisation ops — consistent across phases, so cross-phase
comparisons stay valid, but worth one sentence in the write-up.

**Validation:** Hand-computed attention FLOPs and `compute_flops()`'s reported value agree within
a documented tolerance (exact match unlikely given rounding/bias-term conventions differ across
FLOP-counting methodologies; document whichever convention is used, consistent with how
`compute_flops()` already documents `flops = macs * 2`).

---

## Task 6 — Config, CLI, and Notebook Integration

**What:** `configs/experiments/phase8.yaml` (model list: the seven registry entries, `stages:
[fp32, qat, int8]`, same shape as every prior phase's experiment config), reuse `scripts/train.py`
**unmodified** (no new CLI driver needed, unlike Phase 7 — Phase 8's models fit the existing
`run_experiment()`'s assumption of `create_imagenet_loaders` + base `Trainer`, except for
`deit_tiny`, which needs a `--distillation` flag or a small notebook-side special case since it's
the only model using `DistillationTrainer`).

**Why:** Six of seven models need zero CLI/pipeline changes — this is where Phase 8's "Phase
3-scale, not Phase 7-scale" framing pays off concretely: `scripts/train.py --experiment phase8
--runtime local` should work today, once Task 2's registrations exist, with no new script.

**How:** `deit_tiny`'s distillation training is driven directly from the Phase 8 notebook (not
`scripts/train.py`) since it's the one model needing a non-standard `Trainer` subclass and an
extra teacher-checkpoint dependency — consistent with the project's stated principle ("Skip/resume
logic lives in the notebook loop, not in a wrapper") of keeping CLI drivers generic and pushing
one-off logic into the notebook.

**Inputs/Outputs:** Standard experiment-config → results-CSV flow.

**Dependencies:** Tasks 1–5.

**Deliverables:** `configs/experiments/phase8.yaml`,
`notebooks/phase_8_efficient_vit/vit_qat_phase8.ipynb` (registration
cells, `DistillationTrainer` training cell for `deit_tiny`, standard FP32/QAT/INT8 loop for the
other six via `scripts/train.py`-equivalent notebook cells, matching every prior phase's notebook
structure). Model the YAML on `configs/experiments/default.yaml` — same
`name`/`models`/`seed`/`stages`/`data`/`training`/`qat` shape, with `models` as the explicit
seven-name list rather than `all`.

**Pitfalls / Alternatives:** `scripts/train.py` reads `ml/model_registrations.py`, not the
notebook, so the CLI path depends on Task 2's mirrored registrations existing. Otherwise nothing
beyond what Tasks 1–5 surfaced.

**Validation:** `python -m scripts.train --experiment phase8 --runtime local --dry-run` resolves
without error for the six non-distillation models (`--dry-run` confirmed present,
`scripts/train.py:375`); one short local run (2–3 epochs, `stages: [fp32]`) completes end-to-end
before a full PCAD submission.

---

## Task 7 — Cross-Phase Analysis Notebook

**What:** `notebooks/phase_8_efficient_vit/phase8_results_analysis.ipynb` — joins Phase 8's results to Phase
2/3's classification results and Phase 6's profiling infrastructure (reused directly on the new
models per H5) to test H1–H5.

**Why:** Same reasoning as Phase 7's Task 9 — the scientific payoff is cross-phase.

**How:**
- H1: FP32 top-1 vs. `window_size` line plot (the direct kernel-size-sweep analogue of Phase 2's
  results table).
- H2: hybrid vs. `alexnet_bottleneck`/`alexnet_fire`/pure-Swin, accuracy-vs-size scatter, same
  Pareto-frontier framing `ideas/BEST_MODELS.md` already uses.
- H3: **weights-only** INT8/FP32 ratio and quantized-parameter-fraction bar chart, all Phase 8
  models vs. Phase 3's Bottleneck/Fire for contrast. Do not plot the CSV's `compression_ratio`
  column against Phase 8 without the optimizer-state caveat from H2 — it is not a like-for-like
  quantity.
- H4: `vit_tiny` vs. `deit_tiny` FP32 top-1, single paired bar; name the teacher actually used.
- H5: `profile_kernel_trace()` (extended to return per-kernel timings, per H5) and
  `profile_model_latency()` on all seven models, on whichever GPU is locally available (same
  "RTX 4090/PCAD full sweep is a stretch goal" reasoning Phase 7's Task 9 used) — per-module
  (stem vs. attention-stage) latency breakdown via `torch.profiler(record_shapes=True)`.
- Produce `results/phase_8_efficient_vit_hybrid_attention_analysis/phase8_comparison.csv` (one
  path, matching the Outputs list below and every prior phase's convention) and update
  `ideas/BEST_MODELS.md`/`TODO.md`.

**Inputs:** `results/results_aggregate/model_details_cross_phase.csv`, Phase 6's profiling JSON, Phase 8's own comparison CSV.

**Outputs:** Figures (`results/figures_generated/phase_8_efficient_vit_hybrid_attention/phase8_*`),
`results/phase_8_efficient_vit_hybrid_attention_analysis/phase8_comparison.csv`, updated
`TODO.md`/`ideas/BEST_MODELS.md`.

**Dependencies:** Tasks 1–6 complete with at least FP32+INT8 results for all seven models.

**Deliverables:** `notebooks/phase_8_efficient_vit/phase8_results_analysis.ipynb`.

**Pitfalls / Alternatives:** With 3–6 points per hypothesis, correlation statistics have limited
power — same caveat Phase 6/7 already state explicitly; report raw numbers prominently.

**Validation:** Every figure/table traceable to a specific CSV/JSON on disk, matching the
crash-safe convention every prior phase's analysis notebook follows.

---

## BLOCKING ISSUES & REQUIRED FIXES

Must be resolved before committing to a full training run. Items 1, 2 and 4 are **confirmed
failing today**, not risks to watch.

### 1. Missing QuantStub/DeQuantStub — INT8 Convert Fails (BLOCKING, CONFIRMED FAILING)
Torchvision's `VisionTransformer`/`SwinTransformer` carry no quant stubs, so `convert()` raises
`NotImplementedError: Could not run 'quantized::linear' / 'quantized::conv2d.new' with arguments
from the 'CPU' backend` on both paths (reproduced). **Fix:** Task 3 items 1–2 — a stub-carrying
wrapper per architecture, plus dequant/quant boundaries around every FP32 island required by D6's
mixed-precision design. Definition of done: Task 3's Validation check 2 passes. This is the
largest single piece of work in Phase 8.

### 2. QuantizableMHA Is Wrong at `batch_first=True` (BLOCKING, CONFIRMED FAILING)
`torch.ao.nn.quantizable.MultiheadAttention` in `torch==2.5.1` returns wrong values in exactly the
layout torchvision's ViT uses — 0.946 max abs error vs. `nn.MultiheadAttention` at identical
weights, 0.0 when built `batch_first=False` (reproduced). **Fix:** the `BatchFirstMHAAdapter` in
Task 3, verified by Task 3's Validation check 1 (with `disable_fake_quant` applied, or the check
fails on correct code). Related: build the quantizable module with `from_float()`, never
`load_state_dict(strict=False)` — the latter leaves `linear_Q/K/V` at random init and reports
nothing.

### 3. FLOPs Undercount for ViT Attention (BLOCKING)
`fvcore` does not count `aten::scaled_dot_product_attention`, so `vit_tiny`/`deit_tiny`'s entire
attention cost is missing from `compute_flops()` (measured). Swin is **not** affected — its
matmuls are counted. **Fix:** Task 5's handler for that one op, plus the hand-computed
cross-check, documented before Task 7's efficiency plots are trusted. Do not add an
`aten::matmul` handler; that would double-count Swin.

### 4. Teacher Checkpoint Does Not Exist (BLOCKING for H4, CONFIRMED FAILING)
`mobilenetv2_best.pth` is absent from `checkpoints/` and `outputs/` (measured), so
`DistillationTrainer`'s dependency cannot be satisfied as written. **Fix:** recover it from PCAD,
retrain `mobilenetv2`, or substitute an existing teacher — decide per H4 before Task 4, and
budget the GPU time if retraining. Keep the fail-fast existence check in the notebook's first cell
regardless.

### 5. Warmup Field Exists But Is Not Wired (BLOCKING for training stability)
`TrainerConfig.warmup_epochs` is defined (`ml/config.py:36`) and never read — `fit()` builds
`CosineAnnealingLR` unconditionally (`ml/trainer.py:73`). ViT divergence in the first few hundred
steps without warmup is a widely-reported failure mode, and reusing CNN-tuned defaults would
conflate an optimizer mismatch with an architectural finding. **Fix:** wire the existing field via
`LinearLR` + `SequentialLR` (stdlib), add `warmup_epochs` to `configs/training.yaml`, and
reconcile the `weight_decay` disagreement between `ml/config.py` (4e-4) and
`configs/training.yaml` (5e-4) while there.

### 6. Registry Metadata Beyond `lr` Is Silently Ignored (BLOCKING for the optimizer fix)
`scripts/train.py:186` reads only `spec.get("lr", ...)`; a `register_model(weight_decay=...)`
override is accepted and discarded (measured). Blocking #5's mitigation depends on it. **Fix:**
extend that line, or set the transformer hyperparameters in `configs/experiments/phase8.yaml`.

### 7. Window-Size / Grid-Divisibility Assertion (BLOCKING for future edits)
All three sweep values `{2,4,8}` build and forward correctly today (measured), so this is a
regression guard rather than a live bug: a later edit to `depths`/`patch_size` could break
window divisibility silently. **Fix:** the `assert 16 % window_size == 0`-style guard from
Task 1's Pitfalls in every Swin-derived constructor.

### 8. `profile_kernel_trace()` Cannot Produce H5's Metric (BLOCKING for H5)
It returns a boolean, not per-kernel timings, and `winograd_speedup_info` does not exist
(measured). **Fix:** the additive return-dict extension described in H5, before Task 7.

---

## MEDIUM-PRIORITY IMPROVEMENTS

Execute if time allows after the blocking path is validated; not required for a first
publishable result.

### 6. `vgg_style` as a Fifth CNN Comparison Point
Phase 6's profiling table already includes `vgg_style` (fully Winograd-eligible, all-dense-3×3) —
adding it to Phase 8's H2/H5 comparison plots (no new training needed; its row is present in
`results/results_aggregate/model_details_cross_phase.csv` at **51.81% FP32 top-1, 2.41M params**,
verified) would sharpen the "does attention beat the *best* CNN, not just the smallest one"
framing. It is also the closest size match to `vit_tiny` (2.758M) in the whole project, which
makes it the one honest head-to-head available for the ViT — worth promoting above
"medium priority" for that reason alone.

### 7. Full RTX 4090/PCAD Profiling of All Seven Phase 8 Models
Extends H5 from "confirmed on one GPU class" to "confirmed across bandwidth-limited vs.
compute-rich hardware," matching Phase 6/7's own two-GPU methodology. Reuses
`scripts/profile_hardware.py`'s existing CLI/output-JSON conventions (add the seven Phase 8
model names to `configs/profiling.yaml`'s `models:` list) — mechanical, mostly wall-clock cost.

### 8. Depthwise-Conv vs. Pooling Token-Mixer Ablation (D5)
D5 leaves both options open ("whichever trains more stably"); running both to completion (not
just picking one) would let H5's depthwise-specific prediction be checked against a second,
independent data point.

---

## REPRODUCIBILITY & VERIFICATION CHECKLIST

Before submitting any full training run:

- [x] All seven model constructors build and forward to `(2, 200)` at the D3 configs, with
      parameter counts matching D3's table (done 2026-08-17).
- [ ] Task 1's `demo()` self-check committed with the corrected 0.15M–5M bound and per-model
      expected counts.
- [ ] `find_fuse_groups()` output manually inspected for each model's CNN stem (Task 2); empty
      `fuse_map` confirmed correct for the BN-free pure ViT/Swin variants.
- [ ] Seven registrations mirrored into `ml/model_registrations.py`, not just the notebook
      (Task 2) — `scripts/train.py` reads that file.
- [ ] **Blocking #1:** `prepare_qat → convert → forward` succeeds on a 2-layer version of each
      architecture (currently fails — missing quant stubs).
- [ ] **Blocking #2:** `BatchFirstMHAAdapter` equivalence check passes at `atol=1e-5` with
      `disable_fake_quant` applied; `from_float()` used, `load_state_dict(strict=False)` not.
- [ ] `exclude_attention_from_qat()` confirmed to set `qconfig=None` on every `LayerNorm`/
      `ShiftedWindowAttention` instance (inspect `module.qconfig` on a few submodules directly,
      don't just trust the function ran).
- [ ] **Blocking #3:** `aten::scaled_dot_product_attention` handler registered for the ViT path
      only, hand-computed cross-check documented; Swin left unpatched.
- [ ] **Blocking #4:** teacher resolved (recovered / retrained / substituted) and H4's expected
      margin restated if the teacher changed.
- [ ] **Blocking #5:** `warmup_epochs` wired into `Trainer.fit()`, added to
      `configs/training.yaml`, `weight_decay` discrepancy reconciled.
- [ ] **Blocking #6:** per-model `weight_decay` actually reaches the optimizer (assert it in the
      dry-run output, don't assume).
- [ ] **Blocking #7:** grid-divisibility assertions added to Swin-derived constructors.
- [ ] **Blocking #8:** `profile_kernel_trace()` returns per-kernel timings; Phase 6 call sites
      still pass.
- [ ] 2–3 epoch smoke run passed for all seven models (Task 4 Validation) before full-budget
      training or PCAD submission.
- [ ] `configs/experiments/phase8.yaml` `--dry-run` succeeds (Task 6 Validation).
- [ ] `phase8_comparison.csv` populated and cross-referenced against Phase 2/3/6 CSVs before any
      headline claim is written into `TODO.md`/`ideas/BEST_MODELS.md` (Task 7).
- [ ] Any size or compression claim uses params / weights-only MB, with the
      optimizer-state caveat stated wherever the legacy CSV columns appear (H2/H3).

---

## SCOPE & EFFORT

Rough estimate, engineering time only (excludes GPU training wall-clock). Revised upward after
the verification pass — the earlier ~1.5–2.5 day figure assumed a one-helper Task 3, which the
D6 blockers disprove.

- **Task 1 (model architectures):** ~2–3 hours — all seven constructors are already verified
  building and forwarding; what remains is packaging them into `models/vit_variants.py` with the
  asserts and `demo()`. Revised *down* from ~4–6 hours.
- **Task 2 (registry integration):** ~1–2 hours — mechanical, plus the `model_registrations.py`
  mirror and the `weight_decay` plumbing fix (Blocking #6).
- **Task 3 (QAT adaptation):** **~2–3 days** — was ~4–6 hours. Two confirmed blockers, a
  stub-wrapper and FP32-island boundaries per architecture (a `forward()` rewrite, not a helper),
  a `batch_first` adapter, and an untested `prepare_qat`/`convert` traversal through nested
  quantizable modules. This is now the phase's dominant cost and its main schedule risk; the
  FP32-only fallback in D6 is the release valve.
- **Task 4 (training incl. distillation):** ~4–6 hours engineering — the `DistillationTrainer`
  itself is modest, but add the warmup wiring (Blocking #5) and, if the teacher must be
  retrained, a full `mobilenetv2` run of GPU wall-clock (Blocking #4). ViT-family models often
  need more epochs to converge than this project's CNN defaults; budget extra GPU time.
- **Task 5 (FLOPs verification):** ~1–2 hours — the diagnosis is already done (D7); what remains
  is writing and checking the one handler.
- **Task 6 (config/CLI/notebook):** ~1–2 hours — mechanical, reuses `scripts/train.py` unchanged
  for 6 of 7 models.
- **Task 7 (analysis notebook):** ~3–4 hours — includes the additive `profile_kernel_trace()`
  extension H5 needs (Blocking #8).

**Total engineering estimate: ~3.5–5 working days**, before GPU training wall-clock and before
medium-priority stretch items — i.e. **larger than Phase 7's ~2–3 days**, not smaller. The data
pipeline is free (no new dataset, loss, or metric infrastructure), but that saving is more than
offset by transformer QAT, which has no precedent anywhere in Phases 1–7 and which torchvision's
models are not built to support. If the schedule cannot absorb Task 3, take the FP32-only
fallback deliberately at the Task 3 checkpoint: H1, H2, H4 and H5 all remain answerable without
INT8; only H3 is lost.

**Manual/out of scope for tooling:** Nsight Compute deep-dive on attention kernels (same
precedent as Phase 6/7 — external CLI, only if H5's profiling result needs a kernel-level
explanation `torch.profiler`'s trace can't provide).
