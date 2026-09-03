# Summary

Results after implementing phases 1–4, 6, 8, and 9. **Most baselines (MobileNetV2, ResNet18, VGGStyle) show superior accuracy to pure AlexNet models, but Phase 2–4 AlexNet variants achieve competitive accuracy at 100–1000× smaller model sizes.** Phase 4's final hybrid architectures push AlexNet-family accuracy past 49% for the first time — within 3pp of VGGStyle — while Phase 9 shows a single residual bypass, with zero added parameters, now *exceeds* that hybrid's own FP32 gain outright (see Phase 9 below; corrected 2026-08-29 after a `Trainer.fit()` checkpoint-restore bug, `ml/trainer.py`). Phase 7 detection has completed a valid retrain (anchor-recall bug fixed) across all 3 backbones — see the Phase 7 section below; segmentation has also completed PCAD runs for all 3 backbones, not yet analyzed. Phase 8 (does local self-attention match small-kernel CNNs?) has results in for all 7 models — see the Phase 8 section below. Phase 5 is this document plus `results/phase_5_cross_phase_results_analysis/`.

>**Size convention (corrected 2026-09-02).** All `Size (MB)` figures are **weights only**
>(the serialized `model_state_dict`). They previously reported the raw `{model}_best.pth`
>training checkpoint, which also carries AdamW's two momentum buffers — ~3x the model — while
>the INT8 column was already weights-only, so every FP32-vs-INT8 size claim was inflated ~3x
>(recorded ~11.9x compression where the true FP32->INT8 ratio is ~4x). Fixed in
>`ml/reporting.py`; data backfilled by `scripts/backfill_model_size.py`. Accuracies, params
>and MACs are unaffected, and no ranking changed.

---

## Overall Rankings — FP32 Top-1 Accuracy

| Rank | Model | Phase | Accuracy | Params (M) | Size (MB) | Efficiency (Acc/MB) |
|------|-------|-------|----------|-----------|-----------|-------------------|
| 1 | **MobileNetV2** | 1 | 57.99% | 2.48 | 9.69 | 5.98 |
| 2 | **ResNet18** | 1 | 53.91% | 11.28 | 43.07 | 1.25 |
| 3 | **VGGStyle** | 1 | 51.81% | 2.41 | 9.21 | 5.63 |
| 4 | **AlexNetFireBypass** | 9 | **50.57%** | 0.52 | 2.01 | 25.16 |
| 5 | **AlexNetFinalFireResidual** | 4 | **49.79%** | 0.70 | 2.71 | 18.34 |
| 6 | **AlexNetResidual** | 3 | **48.01%** | 60.67 | 231.49 | 0.21 |
| 7 | **deit_tiny** | 8 | **46.38%** | 2.76 | 10.55 | 4.39 |
| 8 | **AlexNetSmallKernel** | 2 | **45.84%** | 1.60 | 6.12 | 7.49 |
| 9 | **AlexNetFinalBottleneckResidual** | 4 | **45.10%** | 0.57 | 2.24 | 20.17 |
| 10 | **AlexNetBottleneck** | 3 | **44.62%** | 0.39 | 1.51 | **29.54** |
| 11 | **AlexNetStacked** | 2 | **44.56%** | 60.48 | 230.77 | 0.19 |
| 12 | **AlexNetDepthwiseSep** | 3 | **44.39%** | 0.31 | 1.23 | **36.07** |
| 13 | **AlexNetFire** | 3 | **43.98%** | 0.52 | 2.01 | 21.88 |
| 14 | **AlexNetFinalDepthwiseFire** | 4 | **43.46%** | 0.47 | 1.85 | 23.52 |
| 15 | **AlexNetFinalBottleneckFire** | 4 | **42.29%** | 0.51 | 1.97 | 21.44 |
| 16 | **vit_tiny** | 8 | 40.35% | 2.76 | 10.55 | 3.82 |
| 17 | **hybrid_bottleneck_swin** | 8 | 40.23% | 0.27 | 1.05 | 38.34 |
| 18 | **swin_pico_w8** | 8 | 36.75% | 0.32 | 1.38 | 26.55 |
| 19 | **swin_pico_w4** | 8 | 33.53% | 0.32 | 1.26 | 26.63 |
| 20 | **swin_pico_w2** | 8 | 32.95% | 0.32 | 1.25 | 26.36 |
| 21 | **swin_pico_poolmixer** | 8 | 30.98% | 0.23 | 0.89 | 34.97 |

Rows 4, 5, 9, 14, and 15 are the Phase 4/9 additions; rows 7 and 16–21 are Phase 8's seven
attention models (`deit_tiny`/`vit_tiny` global attention, the rest local/windowed). Row 4's
FP32 accuracy was corrected 2026-08-29 (was 49.03%) — see the Phase 9 section below. **Both the
"Analysis by Dimension" section and the "Recommended Models to Continue Investigating" tiers
below still discuss the Phase 1–3-only picture** (written before Phase 4/8/9 existed) — read
them with the new rows above in mind; Phase 4, 8, and 9 get their own dedicated analysis
further down instead.

---

## Analysis by Dimension

### 1. **Accuracy-First** (Prioritize prediction quality)

**Best models for Winograd-accelerated deployment where accuracy is paramount:**

- **AlexNetResidual** (48.01%): Best AlexNet-family model overall. Residual connections help gradient flow and accuracy recovery. Large model (~60M params).
- **AlexNetSmallKernel** (45.84%): 45.8% at just 1.6M params—competitive with MobileNetV2 at 50× smaller size.
- **AlexNetStacked** (44.56%): Stacked small-kernel layers compensate for kernel restrictions; 60M params.

**Baselines remain best:**
- **MobileNetV2** (57.99%, 2.48M params): Highest accuracy, efficient, well-studied.
- **ResNet18** (53.91%, 11.28M params): Balanced accuracy and moderate size.

---

### 2. **Efficiency-First** (Maximize accuracy per MB)

**Best bang-for-buck in terms of model size:**

| Model | Acc/MB | Accuracy | Size (MB) | Quantization Drop |
|-------|--------|----------|-----------|------------------|
| **AlexNetDepthwiseSep** | 36.07 | 44.39% | 1.23 | –2.92pp ⚠️ |
| **AlexNetBottleneck** | 29.54 | 44.62% | 1.51 | –0.08pp ✓ |
| **AlexNetFire** | 21.88 | 43.98% | 2.01 | +0.33pp ✓ |
| **AlexNetSmallKernel** | 7.49 | 45.84% | 6.12 | –9.89pp ⚠️ |
| **VGGStyle** | 5.63 | 51.81% | 9.21 | –0.63pp ✓ |
| **MobileNetV2** | 5.98 | 57.99% | 9.69 | — |

**Verdict:** **AlexNetBottleneck** and **AlexNetFire** are Pareto-optimal: tiny (~1.5–2 MB), competitive accuracy (43–44%), and **quantization-stable** (minimal INT8 drop).

---

### 3. **Quantization Stability** (Minimal FP32 → INT8 accuracy drop)

**Most robust models under INT8 conversion:**

| Model | FP32 Acc | INT8 Acc | Drop | Status |
|-------|----------|----------|------|--------|
| **AlexNetBottleneck** | 44.62% | 44.54% | –0.08pp | ✓ **Excellent** |
| **AlexNetFire** | 43.98% | 44.30% | +0.33pp | ✓ **Gain** |
| **AlexNetResidual** | 48.01% | 47.27% | –0.74pp | ✓ **Good** |
| **AlexNetMixed** | 38.74% | 37.99% | –0.75pp | ✓ Good |
| **VGGStyle** | 51.81% | 51.19% | –0.63pp | ✓ Good |
| **AlexNetStacked** | 44.56% | 42.79% | –1.77pp | ⚠️ Moderate |
| **AlexNetSmallKernel** | 45.84% | 35.95% | **–9.89pp** | ❌ **Poor** |
| **AlexNet3x3GAP** | 38.74% | 37.60% | –1.14pp | ✓ Good |
| **AlexNet3x3FC** | 35.79% | 36.19% | +0.40pp | ✓ Gain |
| **AlexNet2x2GAP** | 30.02% | 30.89% | +0.87pp | ✓ Gain |

**Key insight:** Small-kernel models (**SmallKernel, 2x2, 3x3**) quantize either poorly or counterintuitively. Compensation mechanisms (**Bottleneck, Fire, Residual**) quantize much better, possibly due to richer feature representations.

---

### 4. **Kernel Restriction Impact** (Phase 2 — AlexNet Variants)

| Kernel(s) | Model | Accuracy | Drop vs Baseline | Size | Notes |
|-----------|-------|----------|------------------|------|-------|
| 3×3 + FC head | **AlexNet3x3FC** | 35.79% | –48.61pp | 659 MB | Large but poor accuracy |
| 3×3 + GAP head | **AlexNet3x3GAP** | 38.74% | –45.48pp | 2.23 MB | GAP head enables size reduction |
| 2×2 + GAP head | **AlexNet2x2GAP** | 30.02% | –49.95pp | 1.04 MB | Severely limited by kernel size |
| 3×3 stacked | **AlexNetStacked** | 44.56% | –26.92pp | 692 MB | Stacked compensates; 60M params |
| Small kernel mixed | **AlexNetSmallKernel** | 45.84% | –26.74pp | 18 MB | **Best small-kernel result** |
| Mixed (var. kernel) | **AlexNetMixed** | 38.74% | –41.34pp | 20 MB | Moderate mixed approach |

**Verdict:** 
- Pure 3×3 or 2×2 restrictions are costly without compensation.
- **Head type matters:** AlexNet3x3GAP (8.79 MB) vs AlexNet3x3FC (219.75 MB) shows GAP's 25× compression benefit.
- **AlexNetSmallKernel** (custom small-kernel design) recovers well and reaches 45.84% at tiny size.
- Stacking small kernels helps but requires more params.

---

### 5. **Compensation Mechanisms** (Phase 3 — Which techniques work best?)

| Mechanism | Model | Accuracy | Size (MB) | Params (M) | Eff. (Acc/MB) | QAT Drop | Status |
|-----------|-------|----------|-----------|-----------|---------------|----------|--------|
| **Bottleneck** | AlexNetBottleneck | 44.62% | 1.51 | 0.39 | 29.54 | –0.08pp | ✓✓ **Best** |
| **Fire (squeezenet)** | AlexNetFire | 43.98% | 2.01 | 0.52 | 21.88 | +0.33pp | ✓✓ **Excellent** |
| **Depthwise Sep** | AlexNetDepthwiseSep | 44.39% | 1.23 | 0.31 | 36.07 | –2.92pp | ⚠️ Unstable |
| **Residual** | AlexNetResidual | 48.01% | 231.49 | 60.67 | 0.21 | –0.74pp | ✓ Best overall accuracy |
| **Factorized** | AlexNetFactorized | 42.89% | 217.73 | 57.07 | 0.20 | –0.29pp | ✓ Good |
| **Group Conv** | AlexNetGroupConv | 29.18% | 213.34 | 55.92 | 0.14 | –1.47pp | ❌ Poor |
| **SE (Squeeze-Excite)** | AlexNetSE | **0.50%** | 219.91 | 57.65 | — | — | ❌ **Failed** |

**Note:** **AlexNet3x3GAP** (moved to Phase 2) is a simple head-type variant (GAP vs FC), not a compensation mechanism; see Phase 2 section above.

**Key findings:**
- **Bottleneck & Fire** are the sweet spot: tiny, competitive accuracy, quantization-stable.
- **Depthwise Separable** is too aggressive; loses 2.92pp to quantization.
- **Residual** recovers the most accuracy but at cost of large model size.
- **SE attention** failed to train (collapsed to 0.5%); likely requires careful initialization.
- **Group Convolution** provided negligible benefit.

---

## Recommended Models to Continue Investigating

### **Tier 1: Production-Ready (Accuracy + Efficiency + Stability)**
1. **MobileNetV2** — Industry standard; 57.99% accuracy, proven quantization.
2. **AlexNetBottleneck** — Tiny (1.5 MB), 44.62% accuracy, rock-solid QAT (–0.08pp drop).
3. **AlexNetFire** — Tiny (2 MB), 43.98% accuracy, quantization *gain* (+0.33pp).
4. **AlexNetSmallKernel** — Efficient (45.84%, 6.1 MB), but watch QAT stability (–9.89pp drop).

### **Tier 2: High-Accuracy Alternatives**
5. **AlexNetResidual** — Best AlexNet variant (48.01%), good QAT stability (–0.74pp), but large (231 MB).
6. **ResNet18** — Balanced baseline (53.91%), moderate size, baseline QAT reference.

### **Tier 3: Exploratory / Requires Tuning**
7. **AlexNetDepthwiseSep** — Highest efficiency (36.07 Acc/MB) but poor quantization drop (–2.92pp); needs QAT tuning.
8. **AlexNetStacked** — Good accuracy (44.56%) via stacking, but large and moderate QAT drop (–1.77pp).
9. **AlexNetMixed** — Balanced compromise (38.74%, 6.7 MB), stable QAT (–0.75pp).
10. **VGGStyle** — Competitive baseline (51.81%), stable QAT (–0.63pp).

### **Not Recommended**
- **AlexNetSE** — Training failure; do not pursue without debugging.
- **AlexNet3x3FC / AlexNet2x2GAP** — Poor accuracy (30–35%) from naive kernel restriction without compensation.
- **AlexNetGroupConv** — Negligible benefit, poor QAT.

---

## Next Steps

1. **Investigate AlexNetSmallKernel QAT** — Diagnosed: fuse_map is correct and QAT fake-quant
   training tracks FP32 closely (no drop during training); the damage is isolated to the real
   INT8 `convert()` step. The model has no BatchNorm (by design, for a controlled Phase 2
   comparison), so fbgemm's per-tensor activation observer likely has no bounded range to
   calibrate against. `AlexNetSmallKernelWithBN` (`models/compensation.py`) already exists to
   test this and is now registered as `alexnet_small_kernel_with_bn` in
   `notebooks/phase_3_compensation_and_hybrids_training/compensation_qat.ipynb` (Section 11) —
   not yet trained.
2. **Debug AlexNetSE** — Diagnosed: collapse is immediate and total from epoch 1 (loss pinned at
   exactly `ln(200)`, never moves) — not init (identical default init to every sibling), not LR
   (matches the best-performing sibling). Root cause hypothesis: no BatchNorm + 5 serially-chained
   saturating Sigmoid SE gates crush gradient flow through the trunk from the first step.
   `AlexNetResidual(use_se=True)` already has the same `_SEBlock` on a BN-backed scaffold and was
   only ever trained with `use_se=False`; now registered as `alexnet_residual_se` in the same
   notebook (Section 11) to test whether BN fixes it — not yet trained.
3. ~~**Benchmark Winograd compatibility** — Verify that Bottleneck & Fire leverage small-kernel acceleration on actual hardware.~~ **Done — see Phase 6.**
4. **Architecture search** — AutoML over compensation mechanisms for Pareto-optimal size/accuracy/quantization trade-offs. Not started (see "Phase 10" in `TODO.md`, contingent on Phase 8).
5. ~~**Task transfer** — Test best models on object detection and semantic segmentation.~~ **Detection done (valid A4 retrain, all 3 backbones) — see Phase 7. Segmentation PCAD runs also done, all 3 backbones.**
6. **Fine-tune Tier 1 models** for deployment scenarios (mobile, edge, server). Not started.
7. **Extend the H1–H4 analysis notebook to segmentation** now that results are on disk.

---

## Phase 6 — Hardware Profiling (Winograd Efficiency)

Measured on a real RTX 4090 (PCAD `tupi5`), batch=1, 64×64 input — the first phase to check hardware
behavior directly instead of relying on FLOPs/params as a proxy. Full methodology, statistical tests, and
data-quality corrections in `notebooks/phase_6_hardware_profiling_analysis/hardware_profiling_phase6.ipynb`;
hypotheses/acceptance-criteria source in `ideas/PHASE6_PLAN.md`.

| Model | Winograd-eligible | FP32 Latency (ms) | INT8 Latency (ms) | FP32 GFLOP/s | FP32 Top-1 | Efficiency (Acc/ms) |
|---|---|---|---|---|---|---|
| `alexnet_tv` | ⚠️ Only trailing 3×3 layers | 0.49 | 0.98 | 390.1 | 32.89% | 67.6 |
| `alexnet_depthwisesep` | ❌ None (depthwise) | 0.96 | 0.94 | 42.7 | 44.39% | 46.3 |
| `alexnet_bottleneck` | ✅ Dense 3×3 branch | 1.54 | 0.86 | 52.7 | 44.62% | 28.9 |
| `alexnet_final_bottleneck_residual` | ✅ Dense 3×3 branch | 1.93 | 1.27 | 57.8 | 45.10% | 23.4 |
| `alexnet_final_fire_residual` | ✅ Fire expand branch | 1.81 | 1.14 | 61.9 | 49.79% | 27.5 |
| `vgg_style` | ✅ Fully (every conv) | 1.15 | 1.29 | 405.5 | 51.81% | 45.2 |
| `mobilenetv2` | ❌ None (depthwise) | 1.47 | 4.24 | 37.3 | 57.99% | 39.4* |
| `alexnet_fire` | ✅ Fire expand branch | 1.65 | 0.79 | 218.1 | 43.98% | 26.6 |

\*`mobilenetv2` INT8 top-1 accuracy is unmeasured (Phase 1–4 training gap, unrelated to Phase 6) — its
INT8 latency (4.24ms) is unusually high, plausibly an uncalibrated-quantization fusion penalty specific
to its inverted-residual/depthwise structure; efficiency column is FP32-only for this model.

**Hypothesis results** (full statistical detail in the notebook):

- **H1 (dense 3×3 gets Winograd acceleration): PARTIAL.** A paired Wilcoxon test across 32 layer-config
  groups confirms 3×3 is *systematically* faster than 5×5 (p=4.7e-10) — a real, structural effect, not
  noise. But the stricter per-group signal (3×3 beats 2×2 *and* clears a 1.8× margin vs. 5×5) only holds
  in 31% of groups, well under a 60% bar — the effect is real but inconsistent across batch/resolution
  combinations, likely because larger batches/resolutions become memory-bound rather than compute-bound.
  The kernel-name trace detector (`profile_kernel_trace`) never fired at all — best-effort as documented,
  provides no confirmation either way.
- **H2 (depthwise doesn't benefit): PASS (n=2, case-study level).** `alexnet_depthwisesep`/`mobilenetv2`
  median 40.0 GFLOP/s vs. 91.2 GFLOP/s for the dense-conv group.
- **H3 (Pareto frontier beats baseline on accuracy/latency): 4/5 PASS.** `alexnet_bottleneck`,
  `alexnet_final_bottleneck_residual`, `alexnet_final_fire_residual`, and `vgg_style` all beat
  `alexnet_tv`'s accuracy/latency ratio. `mobilenetv2` does not — `alexnet_tv`'s FP32 latency turned out
  to be the fastest of all 16 profiled configs (0.49ms despite a 221MB model), and `mobilenetv2`'s INT8
  accuracy can't be checked (unmeasured). Not smoothed over — see notebook for full discussion.
- **H4 (INT8 preserves latency ranking vs. FP32): PASS, unambiguous.** Spearman ρ=0.9999 (n=96 layer
  configs) — FP32 latency is a reliable proxy for INT8 latency ranking.

**Practical takeaway:** kernel size alone doesn't predict Winograd-friendliness — `alexnet_depthwisesep`
and `mobilenetv2` use 3×3 kernels but see none of the acceleration `vgg_style`/the bottleneck-fire hybrids
show, because Winograd requires *dense* (`groups=1`) convolutions. `alexnet_fire` (INT8: 456 GFLOP/s) and
`alexnet_tv` (FP32: 390 GFLOP/s) post the highest raw compute throughput in this sweep, though `vgg_style`
is the most consistently Winograd-eligible architecture (100% dense 3×3).

**Known limitations:** single-GPU (RTX 4090 only, no RTX 4060 cross-comparison yet); uncalibrated INT8
observers (±5–10% latency bias per `PHASE6_PLAN.md`); `winograd_speedup_info` in the raw JSONL is not
per-kernel-size (see notebook Phase 1 for the correction applied). Full details in the notebook's
Limitations section.

---

## Phase 4 — Final Architecture & Compression

Combines Phase 3's best mechanisms (Bottleneck, Fire, Residual, Depthwise-Separable) into four
hybrid architectures, then separately tests how far each can be compressed below plain INT8. Full
data: `results/phase_4_compression_and_final_architecture_training/`; notebooks:
`notebooks/phase_4_compression_and_final_architecture_training/`.

### Final hybrid architectures — FP32 vs INT8

| Model | Mechanisms combined | FP32 Top-1 | INT8 Top-1 | QAT Δ | Size (MB) | Acc/MB |
|---|---|---|---|---|---|---|
| **AlexNetFinalFireResidual** | Fire + residual shortcuts | 49.79% | 49.20% | –0.59pp | 2.71 | 18.34 |
| AlexNetFinalBottleneckResidual | Bottleneck + residual shortcuts | 45.10% | 45.98% | +0.88pp ✓ | 2.24 | 20.17 |
| AlexNetFinalDepthwiseFire | Depthwise-separable stem + Fire body | 43.46% | 42.79% | –0.67pp | 1.85 | 23.52 |
| AlexNetFinalBottleneckFire | Bottleneck stem + Fire body | 42.29% | 44.00% | +1.71pp ✓✓ | 1.97 | 21.44 |

### Ablation — does combining mechanisms help, or does one alone explain the gain?

Each hybrid compared against its two single-mechanism Phase 3 parents (no retraining, direct lookup):

| Hybrid | Hybrid FP32 | Best single-mechanism parent | Parent FP32 | Δ vs best parent |
|---|---|---|---|---|
| **AlexNetFinalFireResidual** | 49.79% | AlexNetResidual | 48.01% | **+1.78pp — only combo that beats both parents** |
| AlexNetFinalBottleneckFire | 42.29% | AlexNetBottleneck | 44.62% | –2.34pp — combining hurts |
| AlexNetFinalBottleneckResidual | 45.10% | AlexNetResidual | 48.01% | –2.91pp — combining hurts |
| AlexNetFinalDepthwiseFire | 43.46% | AlexNetDepthwiseSep | 44.39% | –0.93pp — combining hurts |

**Only Fire+Residual beats its best single-mechanism parent; the other three combinations
underperform their own ancestor.** Combining mechanisms isn't automatically additive, and can
actively hurt accuracy (Bottleneck+Residual: –2.91pp; Bottleneck+Fire: –2.34pp) — see Phase 9 below
for how much of Fire+Residual's win is attributable to the bypass alone vs. its added stem.

### Compression (Phase 4.1) — how far below plain INT8 can these go?

5 models (`mobilenetv2`, `alexnet_bottleneck`, `alexnet_fire`, `alexnet_depthwisesep`,
`alexnet_final_fire_residual`) pushed through INT8/INT4/INT2/ternary/binary, both PTQ and QAT, plus
per-layer mixed precision. Mean across all 5 models per method:

| Method | Mean Top-1 | Mean Drop (pp) | Mean Ratio | Mean Acc/MB |
|---|---|---|---|---|
| INT8 (anchor) | 47.40% | 0.76pp | 3.97× | 90.8 |
| **Mixed INT4/8 (per-layer)** | 44.52% | **–0.19pp (gain)** | 7.05× | 215.3 |
| INT4 QAT | 43.31% | 1.02pp | 7.86× | 230.2 |
| INT4 PTQ | 25.31% | 22.85pp | 7.86× | 106.2 |
| Ternary QAT | 26.06% | 22.10pp | 15.42× | 205.4 |
| INT2 QAT | 17.40% | 30.75pp | 15.42× | 145.4 |
| Binary QAT | 10.73% | 37.43pp | 29.70× | 164.8 |

**Verdict:** Mixed-precision INT4/8 is the actual Pareto winner among the aggressive options — it
beats plain INT8 on *both* accuracy (slight gain) and size (7.05× vs. 3.97×) by keeping the
sensitive ~30% of layers at INT8 and pushing the rest to INT4. Pure INT4 QAT is close behind and
simpler (uniform bit-width, no sensitivity analysis needed). Below INT4, retraining recovers
meaningfully more than PTQ at the same bit-width (ternary QAT 26.1% vs. INT4 PTQ 25.3%, despite
ternary being *more* aggressive) — but ternary/binary still cost 22–37pp and aren't viable without
further work. Full per-model breakdown: `.../pareto_frontier.csv`.

---

## Phase 7 — Detection: A4 retrain complete (valid numbers); Segmentation: PCAD runs complete, not yet analyzed

**Root cause found and fixed** (`docs/PHASE7_LOG.md` Stage 9): anchor recall was originally
0.76–0.80 for all 3 backbones (well under the 95% acceptance bar), caused by a tap-index bug
producing duplicate/degenerate pyramid levels plus `DefaultBoxGenerator`'s `min_ratio`/`max_ratio`
linear scale interpolation badly mismatched to VOC's actual box-size distribution. Fixed via
corrected tap indices + explicit percentile-matched anchor scales; recall is now 0.991/0.991/0.932
(bottleneck/fire/alexnet_tv) at 512px. The A4 retrain (FP32 → QAT → INT8, all 3 backbones, plain
+ ImageNet-pretrained-backbone variants) has since completed on PCAD against the fixed config —
these are real, citable numbers:

| Model | Variant | FP32 mAP | QAT mAP | INT8 mAP | Size FP32/INT8 (MB) |
|---|---|---|---|---|---|
| alexnet_bottleneck | scratch | 20.98% | 21.00% | 20.85% | 5.96 / 3.35 |
| alexnet_bottleneck | pretrained | 20.83% | 20.15% | 19.79% | 5.97 / 3.35 |
| alexnet_fire | scratch | 8.94% | 6.38% | 6.40% | 6.46 / 3.47 |
| alexnet_fire | pretrained | 8.93% | 7.79% | 7.54% | 6.46 / 3.47 |
| alexnet_tv | scratch | 16.94% | 14.73% | 14.50% | 223.4 / 221.5 |
| alexnet_tv | pretrained | 18.10% | 17.24% | 16.96% | 223.4 / — |

**Reading it:** bottleneck is both the best detector and quantization-stable (FP32→INT8 within
0.2pp, matching its classification-phase behavior); fire is the worst detector and takes a real
QAT hit (–2.6pp scratch) that backbone pretraining partly recovers; tv (the largest backbone by
far, ~223MB vs. ~6MB) doesn't convert its size advantage into a proportional mAP lead over
bottleneck. Backbone pretraining helps tv and fire's QAT/INT8 stages but has ~no effect on
bottleneck. This is a first read of the raw numbers, not yet the H1–H4 hypothesis analysis — see
`notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb` for that.

Segmentation (Part B) has a fully implemented model/trainer/CLI (`build_deeplabv3_segmenter`,
`SegmentationTrainer`, `run_segmentation` — no longer placeholders) and PCAD runs are now complete
for all 3 backbones × FP32/QAT/INT8 (`outputs/detection_segmentation/phase7/seg_*`); results are
on disk but not yet folded into the H1–H4 analysis notebook.

---

## Phase 9 — Bypass Ablation: does Phase 4's whole architecture change matter, or just the bypass?

Phase 4's `AlexNetFinalFireResidual` (49.79% FP32) beats Phase 3's `AlexNetFire` (43.98% FP32) by
+5.81pp — but it changes two things at once versus Fire: it adds a 3×3 stride-2 stem *and* wraps
every Fire stage in a residual shortcut. `AlexNetFireBypass` isolates the shortcut alone (Fire's
exact stem, one identity bypass added, zero extra parameters) to find out which change is doing the
work. Full plan: `ideas/PHASE9_PLAN.md`.

| Run | Epochs | FP32 Top-1 | INT8 Top-1 | Quant. Δ |
|---|---|---|---|---|
| Initial (PCAD job 806654) | 66 | 47.05% | 47.16% | +0.11pp (gain) |
| **Large-scale (final, corrected)** | **152** | **50.57%** | **49.86%** | **–0.71pp (drop)** |

**Corrected 2026-08-29:** the large-scale run's FP32 number was originally read as 49.03% —
`Trainer.fit()` returned the last epoch's weights instead of reloading the best checkpoint, so
FP32 (measured post-fit) and INT8 (measured from a QAT run that correctly started from the best
checkpoint) were scored on different weights. `ml/trainer.py` is fixed and this run's FP32 was
re-evaluated from the surviving best checkpoint (`scripts/backfill_best_epoch_eval.py`); INT8 is
unchanged (the QAT stage already started from the correct checkpoint). See `docs/PHASE8_LOG.md`
Stage 11 and `report/ic_report.tex` Eixo 4.

**The bypass-alone fraction of Phase 4's total gain grew substantially with more training — and
the correction pushes it further still.** The original ~55% estimate (git commit `924d553`) came
from the 66-epoch initial run. The 152-epoch large-scale run's isolated shortcut now *exceeds* the
full hybrid's FP32 gain outright: +6.59pp over `AlexNetFire` vs. the full hybrid's +5.81pp
(previously read as closing ~87% of that gap, not surpassing it). It also still beats the hybrid
on raw INT8 accuracy (49.86% vs. 49.20%). But unlike the pre-correction reading, it does **not**
keep Fire's quantization-gain property — corrected FP32 is *higher* than INT8, a –0.71pp drop,
close to the full hybrid's own –0.59pp drop (see Phase 4 table above). Zero added parameters
(0.516M, identical to `AlexNetFire`) either way.

**Training-budget caveat, now more load-bearing:** `AlexNetFireBypass` ran 152 epochs vs.
`AlexNetFinalFireResidual`'s 77 (PCAD's larger epoch budget vs. what was available locally) — part
of the larger FP32 gain may reflect that longer training, not the residual shortcut alone.

**Practical implication:** the stem change in `AlexNetFinalFireResidual` contributes comparatively
little once training is run to completion — a plain Fire backbone plus one residual shortcut gets
most or all of the benefit at Fire's exact parameter count — but, corrected, it no longer quantizes
better doing it; both the isolated shortcut and the full hybrid lose a comparable amount under QAT.

**Task 2 — structured channel pruning** (`scripts/phase9/prune_channels.py`, `alexnet_bottleneck`,
ratio 0.4): 385,000 → 207,399 params (53.9%), forward-passes cleanly, every remaining `Conv2d`
stays `groups=1` (Winograd-eligible by construction). Unfine-tuned accuracy collapses as expected
(top1=0.50%, top5=2.35%) — this is a mechanics/Winograd-eligibility check, not a competitive
pruned-accuracy result. `scripts/phase9/prune_channels.py --finetune-epochs` already implements the
fine-tuning loop (full FP32→QAT→INT8, mirrors `scripts/train.py`) but every PCAD submission
attempt (jobs 812276-812278) crashed before Python ran — `scripts/slurm/prune_channels.sbatch`
used a stale `TRAIN_REPO_ROOT`/conda-activate pattern that broke in a non-interactive batch shell.
Fixed to match `det_seg.sbatch`'s proven `git rev-parse`+`.venv` pattern; not yet resubmitted.

**Task 3 — compression headroom** (`scripts/phase9/measure_compression.py`, `alexnet_fire`): real INT8
weights use 7.19 bits/weight (vs. 8.00 nominal); k-means weight-sharing at 16/32/64 clusters
(4/5/6-bit) all beat the real on-disk gzip ratio (1.18×), with up to ~2.2× headroom at 16 clusters
(weights-only) — gzip's DEFLATE is only capturing a fraction of the achievable compression on top
of INT8. H3's acceptance criterion is met; per D6 no changes were made to `ml/checkpoint.py`, this
is a measurement-only signal that a real weight-sharing pipeline would be worth building.

Full tables and methodology: `ideas/PHASE9_PLAN.md` (Tasks 1–3). Reproducible analysis notebook:
`notebooks/phase_9_pcad_bypass_ablation_analysis/phase9_ablation_analysis.ipynb`.

---

## Phase 8 — Efficient Local Attention: does windowed self-attention match small-kernel CNNs?

Tests whether local self-attention (windowed Swin / ViT) reproduces small-kernel CNNs'
accuracy/efficiency/quantization profile, and whether it's structurally Winograd-incompatible
regardless of window size. 7 models (`models/vit_variants.py`); hypotheses H1–H5 and build
history: `ideas/PHASE8_PLAN.md`, `docs/PHASE8_LOG.md`.

**Note on corrected data:** FP32 accuracy for 5 of these 7 models (all but `vit_tiny`/`deit_tiny`,
which used the notebook's `load_best_model()` path and were unaffected) was corrected 2026-08-29
after finding `Trainer.fit()` returned the last epoch's weights instead of reloading the best
checkpoint — see the Phase 9 section above and `scripts/backfill_best_epoch_eval.py`. INT8 numbers
for these 5 could not be rebuilt (no full-precision QAT-best checkpoint survived) and are left as
previously measured.

| Model | Params (M) | MACs (M) | FP32 Top-1 | INT8 Top-1 | ΔQAT | Size (MB) | Acc/MB |
|---|---|---|---|---|---|---|---|
| **deit\_tiny** | 2.76 | 185.5 | **46.38%** | 47.66% | +1.28pp | 10.55 | 4.39 |
| vit\_tiny | 2.76 | 185.5 | 40.35% | 42.44% | +2.10pp | 10.55 | 3.82 |
| hybrid\_bottleneck\_swin | 0.27 | 30.3 | 40.23% | 40.18% | –0.06pp | 1.05 | **38.34** |
| swin\_pico\_w8 | 0.32 | 35.3 | 36.75% | 36.99% | +0.24pp | 1.38 | 26.55 |
| swin\_pico\_w4 | 0.32 | 31.8 | 33.53% | 33.89% | +0.36pp | 1.26 | 26.63 |
| swin\_pico\_w2 | 0.32 | 30.9 | 32.95% | 33.25% | +0.31pp | 1.25 | 26.36 |
| swin\_pico\_poolmixer | 0.23 | 21.2 | 30.98% | 30.76% | –0.22pp | 0.89 | 34.97 |

**Hypothesis results** (full detail in `ideas/PHASE8_PLAN.md`):

- **H1 (window size is attention's kernel-size analogue): CONFIRMED.** Accuracy grows
  monotonically with window ∈ {2,4,8} in both FP32 (32.95→33.53→36.75%) and INT8
  (33.25→33.89→36.99%) — restricting the local receptive field costs accuracy here too.
- **H2 (CNN stem + local attention beats both pure paradigms): PARTIAL.**
  `hybrid_bottleneck_swin` beats max-window Swin (`swin_pico_w8`) by +3.19pp INT8 at similar size
  (0.542 vs. 0.803 MB), but doesn't close the gap to compensation CNNs at equal size —
  `AlexNetFireBypass` occupies nearly the same INT8 footprint (~0.55 MB) and hits 49.86% vs. the
  hybrid's 40.18%.
- **H3 (attention is less quantization-robust than Phase 3's patterns): REJECTED, opposite of
  expected.** 5 of 7 models *gain* accuracy under INT8 (+0.24 to +2.10pp: the three
  `swin_pico_{w2,w4,w8}`, `vit_tiny`, `deit_tiny`); only `swin_pico_poolmixer` (–0.22pp) and
  `hybrid_bottleneck_swin` (–0.06pp, near-zero) drop, and only slightly. Plausible mechanism:
  keeping LayerNorm/attention in FP32 and converting only Linear/Conv turns QAT fine-tuning into
  extra regularization on the converted fraction — speculative, and doesn't explain the two
  non-gaining models.
- **H4 (DeiT distillation recovers most of the data-hungriness gap): CONFIRMED.** `deit_tiny`
  beats `vit_tiny` (identical architecture/hyperparameters, distillation loss only) by +6.04pp
  FP32 / +5.22pp INT8; the gain from a frozen `MobileNetV2` teacher survives quantization, making
  `deit_tiny` this axis's strongest model.
- **H5 (attention is structurally Winograd-incompatible): CONFIRMED, but not attention-specific.**
  No model triggers a Winograd kernel in any config — expected, since none contain a stride-1 3×3
  conv (the only structure that triggers `F(2×2,3×3)`; patch-embed convs use stride = kernel
  size). Doesn't isolate attention as the cause: pure CNNs without stride-1 3×3
  (`AlexNetBottleneck`, `ResNet18`) also never trigger Winograd.

**Size-based Pareto frontier:** `hybrid_bottleneck_swin` is the steepest single step on the whole
7-model FP32 frontier (+9.25pp for +0.47MB over `swin_pico_poolmixer`) — ahead of even the final
jump to `MobileNetV2` (+6.18pp). Under INT8 the frontier shrinks from 7 to 5 models and only
`swin_pico_poolmixer` survives as a local-attention point; `hybrid_bottleneck_swin` gets dominated
by `AlexNetDepthwiseSep` (smaller *and* more accurate). `swin_pico_poolmixer` is no longer the
worst model in the study — it edges out Phase 2's `AlexNet2x2`/`AlexNet2x2-FC` — but still trails
every other `swin_pico` variant, including the most window-restricted one.

**Key caveat:** `vit_tiny`/`deit_tiny` carry 2.76M params — ~9× the `swin_pico` family
(0.23–0.32M) and close in scale to `VGGStyle` (2.41M) — 5–7× the Pareto-optimal compensation CNNs
(`AlexNetBottleneck` 0.39M, `AlexNetFire`/`AlexNetFireBypass` 0.52M) they compete with on
accuracy. `deit_tiny`'s headline number should be read against that size, not as a free win.

Full methodology, hypothesis acceptance criteria, and build history: `ideas/PHASE8_PLAN.md`,
`docs/PHASE8_LOG.md`. Cross-phase comparison data:
`results/phase_8_efficient_vit_hybrid_attention_analysis/phase8_comparison.csv`.