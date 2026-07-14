# Phase 8 — Efficient Vision Transformers & Hybrid Attention Architectures (Implementation Plan)

Phases 1–4 answered the kernel-restriction question entirely within the convolution paradigm
(shrink the kernel, compensate architecturally). Phase 8 asks a different question: is
convolution — at any kernel size — the right primitive at all, or can **local self-attention**
match small-kernel CNNs' accuracy/efficiency/quantization profile while sidestepping the
receptive-field problem Phase 3 spent an entire phase compensating for? This is the one
`TODO.md` phase that changes the *operator family*, not just the kernel size, so several of
Phase 8's decisions (QAT strategy, Winograd applicability, FLOP accounting) need new reasoning
that Phases 1–7 didn't require.

**Scope framing, stated up front:** Phase 8 is scope-comparable to **Phase 3** (a new model
family plugged into the already-complete `create_imagenet_loaders` → `Trainer` →
`build_qat`/`make_qat_callback` → `convert_to_int8` pipeline), **not** to Phase 7 (new dataset,
new loss, new trainer subclasses, new metrics). Classification stays 200-way on Tiny
ImageNet-200 at 64×64 — zero changes to `ml/data.py`, `ml/config.py`'s `DataConfig`, or
`ml/trainer.py`'s core loop for three of the four model variants. The genuinely new engineering
is narrower and concentrated in three places: (1) four new model files that are mostly thin
wrappers around **already-installed** `torchvision.models.vision_transformer`/`swin_transformer`
classes (confirmed present in the pinned `torchvision==0.20.1`, no new dependency), (2) a
`qconfig = None` submodule-exclusion pattern for LayerNorm/Softmax/attention math that this
codebase has never needed before (every prior model was pure Conv-BN-ReLU), and (3) a
distillation training loop for DeiT-Tiny (one new `Trainer` subclass, following Phase 7's
"subclass only the step, keep the loop" pattern). Budget accordingly — closer to a
multi-day task than Phase 6's afternoon, but well short of Phase 7's multi-day-plus-new-dataset
scope (see **SCOPE & EFFORT**).

---

## Research Hypotheses

### H1: Local-Attention Window Size Is the Transformer Analogue of Conv Kernel Size
**Claim:** Swin-style windowed self-attention's `window_size` parameter plays the same role for
attention that kernel size plays for convolution — it caps the receptive field per layer — so
sweeping it should reproduce Phase 2's kernel-restriction accuracy curve (`2×2` ≪ `3×3`/`5×5` ≪
unrestricted) in attention terms.

**Expected Outcome:** FP32 top-1 accuracy increases monotonically with `window_size` ∈
`{2, 4, 8}` (at fixed `patch_size=8` on a 64×64 input, an 8×8 token grid — `window_size=8`
degenerates to full/global attention, the "unrestricted" analogue). The accuracy gap between
`window_size=2` and `window_size=8` should be directionally similar in shape (not necessarily
magnitude) to Phase 2's `AlexNet2x2GAP` (30.02%) vs. unrestricted `AlexNetTV` (32.89%) gap,
adjusted for the fact that attention's "receptive field" already covers the whole window in one
layer (no compounding-through-depth the way stacked 3×3 convs need, per `AlexNetStacked`'s
Phase 2 result) — so a smaller window-size penalty than the pure-conv case is itself informative,
not a null result.

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
model) at size ≤ 2× `alexnet_fire`'s 5.99 MB, and ≥ pure-Swin-Pico's accuracy at the same window
size used in its attention stages.

**Evidence to Collect:** FP32 top-1, size (MB), params, FLOPs for the hybrid vs. `alexnet_fire`/
`alexnet_bottleneck` (`results/model_details.csv`) and vs. Phase 8's own pure-Swin variant.

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
quantization profile than Phase 3's fully-INT8-convertible Bottleneck/Fire (`−0.08pp`/`+0.33pp`
drop, `results/model_details.csv`).

**Expected Outcome:** INT8 compression ratio (FP32 size / INT8 size) for all four Phase 8 models
is well below Phase 3's Bottleneck/Fire ratios (~4× typical for a fully-INT8 Conv-BN model, since
weights go 32-bit→8-bit); accuracy drop may be small in absolute pp (Linear layers are usually
quantization-tolerant) but the **efficiency** story (size reduction per unit accuracy) is worse.

**Evidence to Collect:** FP32 vs. INT8 size (MB), FP32 vs. INT8 top-1, per-model breakdown of
"quantized parameter fraction" (params inside INT8-eligible modules / total params) — this last
number directly explains *why* compression is capped, independent of accuracy.

**Acceptance Criterion:** All four Phase 8 models show INT8/FP32 size ratio > 0.4 (i.e., less
than ~2.5× compression), vs. Phase 3's Bottleneck/Fire at ratio ≈ 0.25 (4× compression). This is
a mechanical consequence of D6's design choice, not a coin-flip — the acceptance criterion here is
really a confirmation that the mixed-precision accounting is being computed and reported
correctly, not a genuine unknown.

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

**Expected Outcome:** `winograd_trace_detected`/`winograd_speedup_info` (Phase 6's dual-signal
detector) fire for <10% of total latency in pure-ViT/Swin/lightweight-attention models — all
their compute is Linear/matmul, not `groups=1` 3×3 conv. The hybrid model's Winograd-eligible
latency share should be roughly proportional to how much of its forward pass the CNN stem
occupies (expected to shrink as more stages are attention-based) — directly extending Phase 6's
"small kernel ≠ Winograd-compatible" finding (which was about depthwise convs) to a second,
structurally different case (attention).

**Evidence to Collect:** Reuse `ml/profiling.py`'s `profile_kernel_trace()`/`profile_model_latency()`
(Phase 6, no new profiling code) on all four Phase 8 models; per-module (stem vs. attention-stage)
latency breakdown via `torch.profiler(record_shapes=True)`, same technique as Phase 7's H4.

**Acceptance Criterion:** Winograd-attributed latency < 10% for pure-attention models (ViT-Tiny,
DeiT-Tiny, lightweight-attention variant); for the hybrid, Winograd-attributed latency share is
strictly less than `alexnet_bottleneck`'s Phase 6-measured share (a pure-CNN model at comparable
depth), confirming attention stages dilute — never add to — Winograd eligibility.

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

| Model | Config | Params (approx.) | Notes |
|---|---|---|---|
| ViT-Tiny (H1 baseline / DeiT-Ti base) | `image_size=64, patch_size=8, hidden_dim=192, mlp_dim=768, num_heads=3, num_layers=6` | ~5.5M | 64 patches + 1 cls token = 65-token sequence. `num_layers=6`, not DeiT-Ti's 12 — halved given the smaller dataset (ViT is known to *overfit*, not underfit, at this data scale; depth is the more likely lever to cut before width, matching the parameter-efficiency intuition Phase 3's Bottleneck already demonstrated for CNNs) |
| Swin-Pico (H1 window sweep) | `patch_size=[4,4], embed_dim=48, depths=[2,2], num_heads=[2,4], window_size=[w,w]` for `w ∈ {2,4,8}` | ~1–2M | 2 stages only: 16×16 → 8×8 token grid (a 3rd `PatchMerging` stage would drop the grid to 4×4, too small for any `window_size>4` sweep point) |
| Hybrid (H2) | Bottleneck-style conv stem (reuse `_AlexBottleneck` from `models/compensation.py`) → 8×8×C feature map → 2 Swin-style windowed-attention stages (`window_size=4`, the H1 sweep's likely best point, confirm after H1 completes) | ~2–4M | See Task 4 |
| Lightweight attention (H5 cross-check) | Swin-Pico architecture with `ShiftedWindowAttention` replaced by a parameter-free pooling or depthwise-3×3-conv token mixer | ~1–2M | See D5/Task 5 |

Every config is deliberately picked, not tuned via a search — this phase compares *architecture
family*, not a hyperparameter-optimized instance of each; note this limitation explicitly in
final reporting (a fully-tuned ViT might close some of the CNN gap further).

### D4 — Local Attention Window Sweep: `{2, 4, 8}` Patches, Not `{3×3, 5×5}`

`TODO.md` phrases the window size as "3×3 or 5×5 patches." `SwinTransformer`'s `PatchMerging`
halves the grid at each stage, and `window_size` must evenly divide the token grid at every
stage it's applied to — power-of-2 windows (`2, 4, 8`) are the values that divide a
power-of-2 grid (`16×16` → `8×8`) cleanly at every stage without padding logic. `window_size=8`
at an 8×8-token stage is full/global attention (the natural "unrestricted" endpoint for the
sweep, directly analogous to Phase 2's uncompensated baseline). Document this substitution
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

This is the single most consequential new decision Phase 8 introduces, and it required checking
this codebase's actual QAT internals (`ml/quantization.py`), not assuming by analogy to Phases
1–7 (all pure Conv-BN-ReLU).

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
  blocks — it is a strict improvement in quantized parameter coverage over the fallback below, at
  the cost of one explicit module-swap step before `build_qat_from_model()`.
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

**No changes needed to `ml/quantization.py`'s core functions** (`prepare_qat_model`,
`build_qat_from_model`, `convert_to_int8`, `make_qat_callback`) — they already operate generically
on any `nn.Module` tree and already respect per-submodule `qconfig` overrides (standard
`torch.ao.quantization` behavior, not project-specific code). The only new code is (a) a small
helper that walks a model and sets `.qconfig = None` on every `nn.LayerNorm` and
`ShiftedWindowAttention` instance before calling `build_qat_from_model()`, and (b) the
`nn.MultiheadAttention` → `torch.ao.nn.quantizable.MultiheadAttention` module-swap helper for the
ViT/DeiT path. `make_qat_callback`'s `freeze_bn_stats` call is a documented no-op for
BatchNorm-free models (Phase 8 uses LayerNorm throughout except in CNN-stem Conv-BN pairs) —
harmless, no special-casing needed, but worth a one-line note in the notebook so it isn't
mistaken for a bug.

### D7 — FLOPs/Params Accounting: Verify `fvcore`'s Attention Coverage Before Trusting It

`ml/reporting.py`'s `compute_flops()` calls `fvcore.nn.FlopCountAnalysis` with
`unsupported_ops_warnings(False)` — this **suppresses**, rather than fixes, the well-documented
gap that `fvcore` (and most static FLOP counters) undercounts or entirely misses `torch.matmul`/
`@`/`torch.bmm` calls used inside attention's QK^T and softmax·V steps unless they're wrapped in
a module `fvcore` recognizes (it has built-in handlers for `nn.MultiheadAttention` as of recent
`fvcore` versions, but **not** for hand-rolled functional attention like `ShiftedWindowAttention`'s
internal matmuls). This must be verified empirically before trusting any Phase 8 FLOPs number:
manually compute expected attention FLOPs (`2 · num_heads · seq_len² · head_dim` for QK^T,
same again for softmax·V, standard transformer FLOP-counting convention, e.g. Kaplan et al. 2020
§2.1) for one model and compare against `compute_flops()`'s reported value. If they disagree by
more than a rounding-level amount, add a `fvcore.nn.jit_analysis.Handle` custom op handler for
the missing matmul calls (fvcore's documented extension mechanism,
`FlopCountAnalysis.set_op_handle(op_name, handler_fn)`) rather than silently reporting an
undercounted number in the final comparison table.

---

## Task 1 — Model Architectures (`models/vit_variants.py`)

**What:** Four constructors — `vit_tiny()`, `swin_pico(window_size)`, `hybrid_bottleneck_swin()`,
`swin_pico_poolmixer()` — each returning an `nn.Module` ready for `MODEL_REGISTRY`.

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
- `swin_pico_poolmixer()`: same `SwinTransformer`-derived structure as `swin_pico`, but with
  `ShiftedWindowAttention` replaced by a custom `_PoolMixer(nn.Module)` (3×3 average pool, or
  depthwise 3×3 conv — implement both, keep whichever trains more stably, document the choice) per
  D5. Requires subclassing `SwinTransformerBlock` (or copying its ~20-line `forward()` with the
  `attn_layer` swapped — `SwinTransformerBlock.__init__` already accepts an `attn_layer` callable
  argument, confirmed via source read, so this is a **constructor argument swap, not a
  monkeypatch or copy** — pass `attn_layer=_PoolMixer` directly to `SwinTransformer`'s
  `block=partial(SwinTransformerBlock, attn_layer=_PoolMixer)` argument, also confirmed present in
  `SwinTransformer.__init__`'s signature).

**Inputs:** None beyond `num_classes=200` (matches every existing `MODEL_REGISTRY` constructor's
zero-config, defaults-from-`CLAUDE.md`-convention signature).

**Outputs:** Four (six, counting the 3-way window sweep as separate registrations) `nn.Module`
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
- If `_AlexBottleneck` isn't cleanly extractable as a standalone stem (it may assume a fixed
  input/output channel count tuned for `AlexNetBottleneck`'s specific stage sequence), a 2–3 line
  custom `nn.Sequential` of `Conv2d-BatchNorm2d-ReLU` stride-2 blocks is an acceptable, simpler
  fallback stem — the *scientific* point of D2/H2 (small-kernel CNN early layers + attention late
  layers) doesn't require reusing the exact Phase 3 module, only a structurally equivalent one;
  don't force a fragile import if the fit is awkward.

**Validation:** `demo()`/assert-based self-check per constructor — forward a
`torch.randn(2, 3, 64, 64)`, assert output shape `(2, 200)`, assert `torchinfo.summary()` reports
a finite, sane parameter count (sanity bound: 0.5M–15M, catches an accidental
`hidden_dim`/`embed_dim` typo that would otherwise silently build a 100M+-parameter model). Run
this for all four architectures (six, counting the window sweep) before registering any of them
for training.

---

## Task 2 — `MODEL_REGISTRY` Integration & Fuse-Map Wiring

**What:** `register_model(name, ctor, fuse_map=..., fuse_root_attr=..., lr=...)` calls for all six
Phase 8 registry entries, in the Phase 8 notebook (matching this codebase's existing convention
of notebook-side registration — Phase 7's D-series decisions already established this pattern for
new phases, not a central `models/__init__.py` addition).

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

**Outputs:** Six live `MODEL_REGISTRY` entries.

**Dependencies:** Task 1.

**Deliverables:** Registration cells in `notebooks/vit_qat_phase8.ipynb` (see Task 6).

**Pitfalls / Alternatives:** If `torchinfo.summary()` (used by every prior phase's per-model
reporting step) doesn't cleanly print a `VisionTransformer`/`SwinTransformer`'s layer table (some
third-party summary tools mishandle attention modules' non-standard `forward()` signatures) — a
plausible but unconfirmed risk, worth a quick check in Task 1's validation step rather than
discovering it mid-training-run.

**Validation:** `MODEL_REGISTRY[name]["ctor"]()` builds without error for all six entries;
`find_fuse_groups()`'s output for each model's stem is manually inspected (printed, read, sanity
checked against the actual module tree) at least once before the first training run.

---

## Task 3 — QAT Adaptation (`ml/quantization.py` addition: `exclude_attention_from_qat`)

**What:** One new small helper function, `exclude_attention_from_qat(model: nn.Module) -> nn.Module`,
plus the `nn.MultiheadAttention → torch.ao.nn.quantizable.MultiheadAttention` swap for the
ViT/DeiT path, implementing D6.

**Why:** D6 established that no existing code in this repo (or in stock `torch.ao.quantization`)
handles LayerNorm/attention exclusion automatically — this is new, Phase-8-specific glue, but
intentionally the *smallest possible* addition (one function, no new files, no changes to the
existing `build_qat_from_model`/`convert_to_int8`/`make_qat_callback` call sites) given how much
of the existing pipeline already generalizes.

**How:**
```python
def exclude_attention_from_qat(model: nn.Module) -> nn.Module:
    """Set qconfig=None on LayerNorm and any hand-rolled attention submodule (D6).

    Call BEFORE build_qat_from_model(). Leaves Conv2d/Linear elsewhere untouched —
    they still inherit the model-level qconfig set inside prepare_qat_model().
    """
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, ShiftedWindowAttention)):
            module.qconfig = None
    return model


def swap_quantizable_mha(model: nn.Module) -> nn.Module:
    """Replace nn.MultiheadAttention with the quantizable variant (D6, ViT/DeiT path only)."""
    from torch.ao.nn.quantizable.modules.activation import MultiheadAttention as QuantizableMHA
    for name, child in model.named_children():
        if isinstance(child, nn.MultiheadAttention):
            qmha = QuantizableMHA(child.embed_dim, child.num_heads, batch_first=True)
            qmha.load_state_dict(child.state_dict(), strict=False)  # verify param-name overlap first
            setattr(model, name, qmha)
        else:
            swap_quantizable_mha(child)
    return model
```
Call order in the Phase 8 notebook: `model = load_best_model(...)` → (ViT/DeiT only)
`swap_quantizable_mha(model)` → `exclude_attention_from_qat(model)` → `build_qat_from_model(model,
arch_name, device)` (unchanged) → `fit(epoch_callback=make_qat_callback(...))` (unchanged) →
`convert_to_int8(...)` (unchanged).

**Inputs:** FP32-trained model (Task 4's `fit()` output).

**Outputs:** QAT-prepared model, mixed-precision by construction (D6), ready for the existing
`fit()`/`convert_to_int8()` calls unchanged.

**Dependencies:** Task 1, Task 2.

**Deliverables:** `exclude_attention_from_qat()`, `swap_quantizable_mha()` added to
`ml/quantization.py`.

**Pitfalls / Alternatives:**
- `QuantizableMHA.load_state_dict(..., strict=False)` is a real risk point: the quantizable
  variant's internal parameter names may not exactly match stock `nn.MultiheadAttention`'s
  (`in_proj_weight` vs. separate `q_proj`/`k_proj`/`v_proj`, depending on `torch.ao.nn.quantizable`'s
  exact internal structure in this PyTorch version) — **verify the actual parameter name overlap
  by diffing `child.state_dict().keys()` vs. a freshly-constructed `QuantizableMHA(...).state_dict().keys()`
  before trusting `strict=False` to silently do the right thing**; a silent shape/name mismatch
  here would leave the swapped-in attention block at its random-init weights, invisibly corrupting
  every downstream QAT/INT8 number without raising an error. This is the single highest-risk step
  in Task 3 — treat `strict=False` as a starting hypothesis to verify, not a solution.
- If the state-dict swap proves unreliable, the safe fallback is training the QAT-prepared model
  (with `QuantizableMHA` already swapped in, at its own random init) from scratch through the QAT
  fine-tuning schedule directly, skipping the "load FP32 weights first" step for the attention
  submodule specifically (accept a slightly longer QAT fine-tune to let the swapped attention
  weights re-converge) — document whichever path is actually used, don't silently assume the
  weight transfer worked.

**Validation:** After the state-dict transfer, forward an identical input through the pre-swap and
post-swap model in eval mode and assert the top-1 predicted class and logit values are close
(`torch.allclose(..., atol=1e-3)` or tighter) — this is the single check that catches a silent
weight-transfer failure immediately, before any QAT training time is spent on a corrupted model.

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
        # Override only the loss computation; keep the base class's AMP/grad-clip/logging
        # scaffolding by calling into the same structure it uses (see ml/trainer.py L298-334).
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
is Phase 1's best result at 57.99% top-1) — **no new teacher training required**, confirm the
checkpoint file (`checkpoints/mobilenetv2_best.pth` or equivalent per `SAVE_DIR` convention)
actually exists on disk before writing the notebook cell that depends on it.

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
  project's CNN-tuned defaults of `lr=3e-4`, `weight_decay=5e-4`, no warmup). Reusing
  `TrainerConfig`'s defaults unchanged risks slow/unstable convergence purely from an optimizer
  mismatch, which would be mistaken for an architectural finding. **Mitigation:** add a per-model
  `lr`/`weight_decay` override via the existing `register_model(lr=..., weight_decay=...)`
  metadata mechanism (already used by `alexnet_fp32.yaml` for a per-model override, per
  `CLAUDE.md`'s Key Patterns) rather than changing `TrainerConfig`'s global defaults — and
  strongly consider adding a minimal linear-warmup wrapper around the existing
  `CosineAnnealingLR` schedule (a `LinearLR` + `SequentialLR` composition, both stdlib `torch.optim`
  classes, no new dependency) since ViT training divergence in the first few hundred steps
  without warmup is a widely-reported failure mode, not a hypothetical risk.
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
If `fvcore`'s number is missing the attention component (expected, per D7's stated concern), add
a custom handler:
```python
from fvcore.nn.jit_handles import Handle
def matmul_flop_handle(inputs, outputs) -> int: ...
analysis.set_op_handle("aten::matmul", matmul_flop_handle)  # or the relevant aten op name, confirm via analysis.unsupported_ops() output
```

**Inputs:** `vit_tiny()` constructed instance.

**Outputs:** A verified `compute_flops()` call path (either confirmed already-correct, or patched
via a custom op handle) that the rest of Phase 8's reporting can trust without re-checking.

**Dependencies:** Task 1.

**Deliverables:** A short verification cell in the Phase 8 notebook, plus (if needed) a
`compute_flops`-adjacent helper in `ml/reporting.py` registering the custom `fvcore` op handle for
attention models specifically (gated so it doesn't affect Phase 1–7's existing FLOPs numbers).

**Pitfalls / Alternatives:** `analysis.unsupported_ops_warnings(False)` currently silences the
exact diagnostic (`FlopCountAnalysis.unsupported_ops()`, a method that lists which ops weren't
counted) that would make this gap visible without manual verification — call
`analysis.unsupported_ops()` explicitly (it doesn't require re-enabling the warnings flag) as the
first diagnostic step before writing any manual formula, it may directly report which `aten::`
ops need a handler rather than requiring a guess.

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

**Deliverables:** `configs/experiments/phase8.yaml`, `notebooks/vit_qat_phase8.ipynb` (registration
cells, `DistillationTrainer` training cell for `deit_tiny`, standard FP32/QAT/INT8 loop for the
other six via `scripts/train.py`-equivalent notebook cells, matching every prior phase's notebook
structure).

**Pitfalls / Alternatives:** None beyond what Tasks 1–5 already surfaced.

**Validation:** `python -m scripts.train --experiment phase8 --runtime local --dry-run` resolves
without error for the six non-distillation models; one short local run (2–3 epochs,
`stages: [fp32]`) completes end-to-end before a full PCAD submission.

---

## Task 7 — Cross-Phase Analysis Notebook

**What:** `notebooks/analysis/phase8_results_analysis.ipynb` — joins Phase 8's results to Phase
2/3's classification results and Phase 6's profiling infrastructure (reused directly on the new
models per H5) to test H1–H5.

**Why:** Same reasoning as Phase 7's Task 9 — the scientific payoff is cross-phase.

**How:**
- H1: FP32 top-1 vs. `window_size` line plot (the direct kernel-size-sweep analogue of Phase 2's
  results table).
- H2: hybrid vs. `alexnet_bottleneck`/`alexnet_fire`/pure-Swin, accuracy-vs-size scatter, same
  Pareto-frontier framing `ideas/BEST_MODELS.md` already uses.
- H3: INT8/FP32 size ratio and quantized-parameter-fraction bar chart, all Phase 8 models vs.
  Phase 3's Bottleneck/Fire for contrast.
- H4: `vit_tiny` vs. `deit_tiny` FP32 top-1, single paired bar.
- H5: reuse `ml/profiling.py`'s `profile_kernel_trace()`/`profile_model_latency()` (Phase 6,
  zero new profiling code) on all seven models, on whichever GPU is locally available (same
  "RTX 4090/PCAD full sweep is a stretch goal" reasoning Phase 7's Task 9 used) — per-module
  (stem vs. attention-stage) latency breakdown via `torch.profiler(record_shapes=True)`.
- Produce `results/phase8_comparison.csv` (same convention as every prior phase) and update
  `ideas/BEST_MODELS.md`/`TODO.md`.

**Inputs:** `results/model_details.csv`, Phase 6's profiling JSON, Phase 8's own comparison CSV.

**Outputs:** Figures (`results/figures/phase8_*`), `results/phase8_comparison.csv`, updated
`TODO.md`/`ideas/BEST_MODELS.md`.

**Dependencies:** Tasks 1–6 complete with at least FP32+INT8 results for all seven models.

**Deliverables:** `notebooks/analysis/phase8_results_analysis.ipynb`.

**Pitfalls / Alternatives:** With 3–6 points per hypothesis, correlation statistics have limited
power — same caveat Phase 6/7 already state explicitly; report raw numbers prominently.

**Validation:** Every figure/table traceable to a specific CSV/JSON on disk, matching the
crash-safe convention every prior phase's analysis notebook follows.

---

## BLOCKING ISSUES & REQUIRED FIXES

Must be resolved before committing to a full training run:

### 1. Attention-Weight-Transfer Verification (BLOCKING)
The `nn.MultiheadAttention → QuantizableMHA` state-dict swap (Task 3) is the single highest-risk,
least-precedented step in this plan — a silent shape/name mismatch would invisibly corrupt QAT
results. **Fix:** the `torch.allclose()` pre/post-swap output check specified in Task 3's
Validation is mandatory, not optional, before any QAT training time is spent.

### 2. FLOPs Undercount for Attention Ops (BLOCKING)
`fvcore`'s default handler set is confirmed (via documented, general `fvcore` behavior) to miss
some matmul-based ops unless custom handlers are registered — an unverified FLOPs number would
corrupt every efficiency comparison in Task 7. **Fix:** Task 5's verification step, completed and
its outcome (confirmed correct, or patched) documented, before Task 7's plots are trusted.

### 3. Window-Size / Grid-Divisibility Assertion (BLOCKING)
A silent shape mismatch inside `SwinTransformer`'s internal windowing/masking logic (if a future
edit changes `depths`/`patch_size` without re-checking `window_size` divisibility) is a plausible,
hard-to-diagnose failure mode. **Fix:** the `assert 16 % window_size == 0`-style guard specified
in Task 1's Pitfalls, added to every Swin-derived constructor.

### 4. Teacher Checkpoint Existence (BLOCKING for H4)
`DistillationTrainer` hard-depends on `mobilenetv2_best.pth` already existing on disk from Phase
1. **Fix:** verify the file's presence (and that it loads via `load_best_model()` without error)
in the notebook's first cell, before any Phase 8 training begins — fail fast with a clear message
rather than a late, confusing crash mid-distillation-training-loop.

### 5. AdamW Warmup for Transformer Training (BLOCKING for training stability)
Reusing `TrainerConfig`'s CNN-tuned optimizer defaults unchanged risks conflating an optimizer
mismatch with an architectural finding (Task 4's Pitfalls). **Fix:** implement the
`LinearLR + SequentialLR` warmup wrapper (both stdlib `torch.optim`, no new dependency) before the
first full training run, not after observing unexplained instability.

---

## MEDIUM-PRIORITY IMPROVEMENTS

Execute if time allows after the blocking path is validated; not required for a first
publishable result.

### 6. `vgg_style` as a Fifth CNN Comparison Point
Phase 6's profiling table already includes `vgg_style` (fully Winograd-eligible, all-dense-3×3) —
adding it to Phase 8's H2/H5 comparison plots (no new training needed, results already exist in
`results/model_details.csv`) would sharpen the "does attention beat the *best* CNN, not just the
smallest one" framing.

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

- [ ] All six (seven, counting the window-size sweep) model constructors pass Task 1's `demo()`
      shape/param-count self-check.
- [ ] `find_fuse_groups()` output manually inspected for each model's CNN stem (Task 2).
- [ ] Attention-weight-transfer `torch.allclose()` check passed before any QAT training (Blocking #1).
- [ ] `exclude_attention_from_qat()` confirmed to set `qconfig=None` on every `LayerNorm`/
      `ShiftedWindowAttention` instance (inspect `model.qconfig` on a few submodules directly,
      don't just trust the function ran).
- [ ] FLOPs verification (Task 5 / Blocking #2) completed, outcome documented.
- [ ] Grid-divisibility assertions added to Swin-derived constructors (Blocking #3).
- [ ] `mobilenetv2_best.pth` existence verified before `DistillationTrainer` is instantiated
      (Blocking #4).
- [ ] AdamW warmup wrapper implemented and used for all seven models' FP32 training (Blocking #5).
- [ ] 2–3 epoch smoke run passed for all seven models (Task 4 Validation) before full-budget
      training or PCAD submission.
- [ ] `configs/experiments/phase8.yaml` `--dry-run` succeeds (Task 6 Validation).
- [ ] `phase8_comparison.csv` populated and cross-referenced against Phase 2/3/6 CSVs before any
      headline claim is written into `TODO.md`/`ideas/BEST_MODELS.md` (Task 7).

---

## SCOPE & EFFORT

Rough estimate, engineering time only (excludes GPU training wall-clock):

- **Task 1 (model architectures):** ~4–6 hours — four architecturally distinct constructors,
  the hybrid (H2) and pool-mixer (D5) variants being the least precedented.
- **Task 2 (registry integration):** ~1–2 hours — mechanical once Task 1 exists.
- **Task 3 (QAT adaptation):** ~4–6 hours — flagged as the highest-uncertainty task (Blocking #1),
  budget contingency here first if anything overruns, matching Phase 7's Task 6 precedent of
  flagging the QAT-adaptation task as the riskiest one in a new-model-family phase.
- **Task 4 (training incl. distillation):** ~3–4 hours engineering (training wall-clock separate;
  ViT-family models often need more epochs to converge than this project's CNN defaults, budget
  extra GPU time even though extra engineering time is modest).
- **Task 5 (FLOPs verification):** ~1–2 hours.
- **Task 6 (config/CLI/notebook):** ~1–2 hours — mechanical, reuses `scripts/train.py` unchanged
  for 6 of 7 models.
- **Task 7 (analysis notebook):** ~2–3 hours.

**Total engineering estimate: ~1.5–2.5 working days**, before GPU training wall-clock and before
medium-priority stretch items — noticeably smaller than Phase 7's ~2–3 days (no new dataset, no
new data pipeline, no new loss/metric infrastructure), but not a Phase-6-scale afternoon either,
because of the genuinely novel QAT-for-attention engineering (D6/Task 3) this phase requires.

**Manual/out of scope for tooling:** Nsight Compute deep-dive on attention kernels (same
precedent as Phase 6/7 — external CLI, only if H5's profiling result needs a kernel-level
explanation `torch.profiler`'s trace can't provide).
