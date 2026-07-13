# Experimental Phases

Each experiment follows the same pipeline:

1. Train the model in **FP32**.
2. Fine-tune using **Quantization-Aware Training (QAT)**.
3. Convert to **INT8**.
4. Evaluate both FP32 and INT8 models.
5. Compare accuracy, efficiency, model size, and quantization impact.

---

# High Priority

## Notebook Output

- ✅ Reduce notebook output volume.
  - ✅ Remove or greatly reduce per-batch `print()` statements, but keep it trackable so I know it is running.
  - ✅ Use `tqdm` progress bars instead of creating new output lines.

- ✅ Separate logs from notebook output.
  - ✅ Save training logs to a `.log` file — `Trainer.__init__(log_file=...)` writes timestamped logs.
  - ✅ Continue logging metrics to Weights & Biases — now includes LR, epoch time, peak GPU memory, `grad_norm` (when `grad_clip_norm` is set).
  - ✅ Save the final run summary to a JSON or CSV file — per-model summary JSON via `make_run_summary()`.

- ✅ Keep notebooks lightweight.
  - ✅ Avoid storing thousands of lines of training logs in the notebook.

- ✅ Improve experiment reporting.
  - ✅ Save a concise `run_summary` (Top-1, Top-5, loss, training time, etc.) as a standalone file.
  - ✅ Aggregate all experiment summaries into a single CSV for easy comparison — comparison CSV still exists; individual summaries are JSON for crash-safety.

- ✅ Verify notebook execution.
  - ✅ Ensure execution errors are written to the log file and are easy to locate — `log_file` handler captures all logging.
  - ✅ Ensure the final run summary is always printed and saved, even if notebook output is collapsed — logged to file + console + W&B.

- ✅ Weights & Biases (W&B)
  - ✅ Always initialize W&B runs with clear, descriptive names and complete metadata (project, group, tags, config with num_classes/img_size/dataset, model, params).
  - ✅ Use clear, descriptive, and consistent names for all W&B artifacts so they are easy to identify and distinguish.

---

## Metrics to Add

### Efficiency

- ✅ Report FP32 model size (MB) — `disk_mb(SAVE_DIR / f"{name}_best.pth")`.
- ✅ Report INT8 model size (MB) — `disk_mb(SAVE_DIR / f"{name}.pth")`.
- ✅ Measure inference latency (ms/image) — `Trainer.benchmark()` returns `latency_ms_per_image`.
- ✅ Measure throughput (images/s) — `Trainer.benchmark()` returns `throughput_img_per_s`.
- ✅ Measure memory usage (GPU) — peak GPU memory per epoch tracked in `history["peak_gpu_mem_mb"]`, logged to W&B.
- ✅ Report MACs/FLOPs — `compute_flops(model)` via fvcore, includes in `make_run_summary()`.
- ✅ Report parameter count — via `torchinfo.summary()`, included in per-model summaries.

### Architecture Comparison

- [ ] Estimate receptive field — requires manual calculation per architecture, not yet automated.
- ✅ Compute parameter efficiency (Top-1 accuracy per million parameters) — `param_efficiency_top1_per_m` in `make_run_summary()`.
- ✅ Compute accuracy drop from FP32 → INT8 (Top-1 and Top-5) — `quantization_drop_top1`, `int8_top1_top5_gap` in summary.
- ✅ Report Top-1 / Top-5 gap — `fp32_top1_top5_gap`, `int8_top1_top5_gap` in summary.

---

## Run Summary

✅ Per-model summaries implemented via `make_run_summary()`. Includes all fields below, saved to `{RESULTS_DIR}/{name}_summary.json`:

- ✅ Model name
- ✅ Training mode (FP32 / QAT / INT8 evaluation)
- ✅ Number of epochs
- ✅ Best validation Top-1
- ✅ Best validation Top-5
- ✅ Final validation Top-1
- ✅ Final validation Top-5
- ✅ Best validation loss
- ✅ Final training loss
- ✅ FP32 model size (MB)
- ✅ INT8 model size (MB)
- ✅ Number of parameters (M)
- ✅ Compression ratio (FP32 size / INT8 size)
- ✅ MACs/FLOPs (via fvcore)
- ✅ Inference latency (ms/image, FP32 GPU + INT8 CPU)
- ✅ Throughput (images/s, FP32 GPU + INT8 CPU)
- ✅ Epoch time (average per epoch, seconds)
- ✅ Total training time (seconds)
- ✅ Peak GPU memory (MB)
- ✅ FP32 vs INT8 Top-1 accuracy drop
- ✅ Top-1 / Top-5 gap (both FP32 and INT8)
- ✅ Parameter efficiency (Top-1 per M params)

---

## Analysis

- ✅ Generate a single CSV summarizing every experiment — `final_comparison.csv` per phase.
- ✅ Cross-phase figures in `results/figures/` (accuracy_vs_macs, fp32_vs_int8_bar, efficiency_accuracy_per_mb, training curves, etc.).
- [ X ] Full cross-phase ranking once Phase 3 results are in:
  - [ X ] Top-1 accuracy ranking
  - [ X ] Top-5 accuracy ranking
  - [ X ] Model size ranking
  - [ X ] Inference latency ranking
  - [ X ] Parameter efficiency ranking
  - [ X ] Accuracy loss after quantization ranking
  - [ X ] Overall efficiency (accuracy vs compute).

---

## Phase 1 — Reference Architectures ✅

Establish baseline performance using representative CNN families.

Models:

- ✅ AlexNet (pretrained)
- ✅ VGG-style CNN
- ✅ ResNet-18 (pretrained)
- ✅ MobileNetV2 (pretrained)

For each model:

- ✅ FP32 training — results in `results/baselines_qat_phase1/final_comparison.csv`
- ✅ QAT fine-tuning (partial — MobileNetV2, ResNet18 skipped; VGGStyle, AlexNetTV completed)
- ✅ INT8 conversion
- ✅ FP32 vs INT8 evaluation
- ✅ Record latency, model size, and quantization accuracy drop

**Results:** MobileNetV2 (57.99%), ResNet18 (53.91%), VGGStyle (51.81%), AlexNetTV (32.88%)

---

## Phase 2 — Kernel Restriction Study ✅

Investigate the impact of restricting convolution kernels while keeping the architecture as controlled as possible.

Models:

- ✅ AlexNet (3×3)
- ✅ AlexNet (2×2)
- ✅ AlexNet (stacked 3×3)
- ✅ AlexNet (mixed kernels)
- ✅ AlexNetSmallKernel (compact all-small-kernel)

For each variant:

- ✅ FP32 training
- ✅ QAT fine-tuning
- ✅ INT8 conversion
- ✅ FP32 vs INT8 comparison
- ✅ Compare against the original AlexNet

Results in `results/alexnet_qat_phase2/` (57 epochs, full QAT + INT8).

---

## Phase 3 — Compensation Mechanisms ✅

Evaluate architectural modifications that compensate for the reduced receptive field caused by small kernels.

Implemented variants:

- ✅ Bottleneck — **Best**: 44.62% FP32, 9.93 Acc/MB, –0.08pp QAT drop
- ✅ Factorized convolutions — 42.89% FP32, stable QAT (–0.29pp)
- ✅ Group convolutions — Poor results (29.18% FP32)
- ✅ Depthwise separable convolutions — 44.39% FP32, high efficiency (12.15 Acc/MB) but QAT unstable (–2.92pp)
- ✅ Residual connections — **Best accuracy**: 48.01% FP32, but large (694 MB)
- ✅ Fire modules — **Excellent**: 43.98% FP32, quantization gain (+0.33pp), tiny (6 MB)
- ✅ Global Average Pooling — Moderate (38.74% FP32)
- ✅ Squeeze-and-Excitation (SE) — **Failed** training (collapsed to 0.5%)

For each variant:

- ✅ FP32 training
- ✅ QAT fine-tuning
- ✅ INT8 conversion
- ✅ FP32 vs INT8 comparison

**Results:** See `ideas/BEST_MODELS.md` for comprehensive analysis. Bottleneck & Fire are Pareto-optimal (tiny, competitive accuracy, quantization-stable).

---

## Phase 4 — Final Architecture

Combine the most successful architectural ideas from previous phases into a final AlexNet-inspired network.

Tasks:

- [ X ] Design the final architecture
- [ X ] FP32 training
- [ X ] QAT fine-tuning
- [ X ] INT8 conversion
- [ X ] Compare with all previous models
- [ X ] Perform complete ablation study

---

## Phase 5 — Final Analysis

Summarize the complete experimental study.

Produce:

- [ X ] FP32 ranking
- [ X ] INT8 ranking
- [ X ] Quantization accuracy-drop analysis
- [ X ] Model size comparison
- [ X ] Latency comparison
- [ X ] Winograd compatibility discussion
- [ X ] Final conclusions and recommendations

---

## Phase 6 — Hardware Profiling & Winograd Efficiency Validation

Measure actual latency, memory bandwidth, and power consumption on **RTX 4060 (laptop, bandwidth-limited)** and **RTX 4090 (PCAD tupi nodes, compute-rich)** to empirically validate Winograd acceleration claims across contrasting hardware. Compare theoretical vs real-world efficiency gains across kernel sizes, and identify whether small-kernel gains hold on both GPU classes or only on the bandwidth-limited one.

Benchmarks (run on both RTX 4060 and RTX 4090):

- [ ] Profile single-layer latency for Conv(k=2×2), Conv(k=3×3), Conv(k=5×5) on RTX 4060 and RTX 4090
- [ ] Measure memory bandwidth utilization (NVIDIA Nsight Compute or similar) on both GPUs
- [ ] Profile full forward pass for Phase 1–5 reference architectures (AlexNet, MobileNetV2, TinyHybridNet, etc.) on both GPUs
- [ ] Profile INT8 vs FP32 inference latency on CPU (fbgemm backend)
- [ ] Identify which layers contribute most to latency (are Winograd gains concentrated or distributed?) and whether this differs between the two GPUs
- [ ] Measure Winograd reordering overhead (input/output packing costs) on both GPUs
- [ ] Profile power consumption under sustained inference (if HW supports NVIDIA RAPL or equivalent)

Outputs:

- [ ] Latency heatmap: kernel size vs layer depth, per GPU
- [ ] Speedup ratio: (Conv 5×5 time) / (Conv 3×3 time), RTX 4060 vs RTX 4090
- [ ] Winograd feasibility threshold: break-even point for reordering overhead, per GPU
- [ ] CPU INT8 latency ranking (identify bottleneck layers)
- [ ] Cross-GPU comparison: does small-kernel efficiency hold on compute-rich hardware (RTX 4090), or only on bandwidth-limited hardware (RTX 4060)?
- [ ] Revised efficiency recommendations based on empirical data

---

## Phase 7 — Detection & Segmentation Kernel Study

Extend the kernel-restriction findings (Phases 2–3) to object detection and semantic segmentation, testing whether the accuracy/efficiency trade-off observed in classification holds for denser prediction tasks. Directly addresses the research objective's detection/segmentation scope, which Phases 1–6 (classification only) do not cover.

Models: reuse Phase 3's Pareto-optimal backbones (Bottleneck, Fire) as feature extractors, paired with a lightweight detection head (e.g., SSD-lite/YOLO-tiny) and segmentation head (e.g., small U-Net/DeepLab-lite decoder). Compare against a large-kernel baseline backbone (AlexNetTV or VGGStyle).

- [ ] Pick a detection/segmentation dataset compatible with small-scale training (e.g., Pascal VOC subset, COCO subset, or Cityscapes)
- [ ] FP32 training (backbone + head)
- [ ] QAT fine-tuning
- [ ] INT8 conversion
- [ ] FP32 vs INT8 evaluation (mAP for detection, mIoU for segmentation)
- [ ] Compare small-kernel vs large-kernel backbones on mAP/mIoU, latency, and model size
- [ ] Quantization robustness comparison (does the QAT-stability ranking from Phase 3 transfer to detection/segmentation heads?)
- [ ] Determine whether the classification kernel-size trade-off transfers to dense prediction tasks

---

## Phase 8 — Efficient Vision Transformers & Hybrid Attention Architectures

Explore whether attention-based models can match or exceed CNN efficiency within Winograd constraints. Investigate local-attention Vision Transformers as an alternative paradigm to small-kernel CNNs.

Models:

- [ ] Vision Transformer (ViT-Tiny) with local attention windows (e.g., 3×3 or 5×5 patches, local self-attention)
- [ ] DeiT-Tiny (Knowledge-distilled ViT)
- [ ] Hybrid CNN-Transformer (small-kernel CNN stem + local transformer blocks)
- [ ] Lightweight attention variant (linear attention or depthwise attention)

For each variant:

- [ ] FP32 training
- [ ] QAT fine-tuning
- [ ] INT8 conversion
- [ ] FP32 vs INT8 evaluation
- [ ] Compare accuracy, latency, model size, and quantization robustness vs Phase 5–6 CNNs
- [ ] Assess Winograd deployment feasibility (attention ops, memory layout)
- [ ] Determine whether attention can substitute for large receptive fields in resource-constrained inference

---

## Phase 9 — Extended Architecture Search (Future)

If Phase 8 yields promising hybrid results, consider automated architecture search (NAS or evolutionary search) to discover optimal kernel-size, depth, width, and attention-ratio combinations under Winograd constraints.

### Deployment Fine-Tuning (NEW)

Optimize Tier 1 models for real-world deployment scenarios.

- [ ] **Mobile (resource-constrained):** Export AlexNetBottleneck/AlexNetFire to TFLite, measure on-device latency, tune for <50ms/image, <100mA
- [ ] **Edge (moderate constraints):** Profile AlexNetSmallKernel QAT drop (–9.89pp); investigate per-channel quantization, mixed-precision INT8
- [ ] **Server (throughput-focused):** Batch optimization for MobileNetV2, measure throughput vs latency trade-off, optional knowledge distillation
- [ ] **Format conversion:** ONNX, TensorRT, Core ML, ONNX-RT exports for each Tier 1 model

**Expected outcome:** Production-ready model exports and deployment tuning for three scenarios.
