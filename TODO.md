# Research Roadmap

## Objective

Design an **AlexNet-inspired CNN** using **3×3 (or smaller) kernels** that provides the best trade-off between:

- Accuracy
- Model size
- Training speed
- Inference speed
- Quantization performance
- Winograd compatibility

---

## Research Strategy

The project will proceed in three stages:

1. **Baseline evaluation of classic CNNs.**  
   Establish reference performance using standard architectures (e.g., AlexNet, VGG, ResNet, MobileNet) under identical training conditions to measure accuracy, compute cost, and inference speed.

2. **Controlled AlexNet-derived ablations that isolate single architectural choices.**  
   Systematically modify AlexNet one component at a time (kernel size, bottlenecks, grouped convolutions, residuals, factorization, pooling changes) to identify the isolated impact of each design decision.

3. **Quantization analysis (PTQ and QAT).**  
   Evaluate model compression and hardware efficiency using post-training quantization and quantization-aware training, measuring trade-offs between INT8/FP32 accuracy, model size reduction, and latency.

4. **Specialized efficient CNNs that introduce modern design ideas for comparison.**  
   Benchmark lightweight and modern architectures (e.g., MobileNet, ShuffleNet, EfficientNet, ConvNeXt-Tiny, GhostNet) to compare against the AlexNet-derived family under efficiency-oriented constraints.

The main scientific goal is to understand which architectural choices preserve accuracy while improving efficiency and hardware compatibility.

---

## Architectures to Explore

## Stage 1 — Baselines (reduced)

Keep a minimal set of reference architectures covering classic, deep, and efficient CNN design spaces:

- **AlexNet (fine-tuned)** — Classic reference CNN under modern training conditions.  
- **StrongCNN** — High-capacity conventional CNN used as an upper baseline for classical architectures.  
- **VGG-style CNN** — Deep stacked 3×3 convolution baseline for small-kernel behavior.  
- **ResNet-18** — Standard residual learning baseline for stable deep training.  
- **TinyHybridNet** — Lightweight hybrid CNN used as a low-resource performance baseline.  

---

## Stage 2 — AlexNet-Derived Variants (core ablation set)

Replace many incremental variants with grouped architectural axes:

- **AlexNet (3×3 uniform)** — Tests impact of enforcing small kernels across all layers.  
- **AlexNet (stacked 3×3)** — Tests whether depth can recover receptive field lost from kernel reduction.  
- **AlexNet (kernel-restricted / small-kernel variant)** — Fully constrained AlexNet variant enforcing small convolution kernels.  
- **AlexNet (factorized convolutions)** — Tests spatial decomposition via 1×k + k×1 convolutions.  
- **AlexNet (bottleneck + global average pooling)** — Tests parameter reduction and classifier simplification.  
- **AlexNet (residual + optional SE blocks)** — Tests optimization stability and attention-based refinement.  

---

## Stage 3 — Efficient CNNs (modern comparison set)

Use a small but representative set of modern architectures:

- **MobileNetV2** — Depthwise separable convolution efficiency baseline.  
- **ShuffleNetV2** — Efficient channel mixing with hardware-aware design.  
- **EfficientNet-B0** — Strong accuracy/efficiency trade-off via compound scaling.  
- **ConvNeXt-Tiny** — Modern CNN design inspired by transformer-era improvements.  

---

## Architectural Ideas To Isolate

Each idea should be tested in a controlled way, ideally by modifying only one model family at a time.

### Kernel Design
- 3×3 kernels
- 2×2 kernels
- Stacked 3×3 convolutions
- Mixed-kernel architectures

### Convolution Variants
- Standard convolutions
- Depthwise separable convolutions
- Group convolutions
- Factorized convolutions

### Network Blocks
- Residual connections
- Fire modules
- Inverted residual blocks
- Squeeze-and-Excitation blocks
- Dense connections
- Channel shuffle

### Hardware / Efficiency Ideas
- Winograd-compatible kernel restriction
- Quantization-aware training
- Post-training quantization
- Structured pruning
- Global average pooling instead of large FC heads

---

## Training Techniques

Use the same training recipe whenever possible so architecture is the main variable.

- Better data augmentation
- Label smoothing
- Exponential Moving Average (EMA)
- Cosine learning-rate scheduling
- Mixed Precision (AMP)

---

## Model Optimization

These should be applied after the architecture study or in a separate comparison pass.

- Quantization-Aware Training (QAT)
- Post-Training Quantization (PTQ)
- Structured pruning
- Knowledge distillation
- Winograd-compatible kernel design

---

## Evaluation Metrics

Evaluate every architecture using:

- Top-1 / Top-5 Accuracy
- Parameters
- FLOPs / MACs
- Model size
- FP32 vs INT8 accuracy
- Quantization accuracy drop
- Training time
- CPU latency
- INT8 inference latency

---

## Recommended Priority Order

1. AlexNet baseline.
2. AlexNet-3×3.
3. AlexNet-Stacked.
4. AlexNet-Bottleneck.
5. AlexNet-Grouped.
6. AlexNet-Fire.
7. AlexNet-Residual.
8. AlexNet-SE.
9. AlexNet-GAP.
10. SqueezeNet.
11. MobileNetV1.
12. MobileNetV2.
13. ShuffleNetV1 / V2.
14. GhostNet.
15. ResNet-18.
16. DenseNet-121.
17. RegNet-Small.
18. EfficientNet-B0.
19. EfficientNet-V2-S.
20. ConvNeXt-Tiny.

---

## Final Goal

Develop a lightweight **AlexNet-like architecture** composed primarily of **3×3 (or smaller) kernels**, inspired by ideas from **ResNet**, **SqueezeNet**, and **MobileNet**, that is:

- Accurate
- Lightweight
- Fast to train
- Fast to infer
- Quantization-friendly
- Compatible with Winograd-based accelerators

## Hybrid CNN Proposals (Experimental Extensions)

These combine multiple architectural principles from classic and modern CNNs. Each variant should be treated as a controlled ablation extension of the AlexNet-focused ladder.

---

### H1. Residual–VGG Hybrid
- VGG-style stacked 3×3 convolutions with residual skip connections
- Global average pooling (GAP) head instead of fully connected layers  
- Tests depth vs optimization stability under small kernels  
- Based on VGG-style CNNs and residual learning (ResNet)

---

### H2. AlexNet–Inception Hybrid
- Replace AlexNet convolution blocks with multi-branch modules:
  - 1×1 convolution branch
  - 3×3 convolution branch
  - stacked 3×3 branch (larger effective receptive field)
- Concatenate outputs across branches  
- Tests multi-scale feature extraction in AlexNet-style backbones  
- Inspired by Inception / GoogLeNet design

---

### H3. MobileNet–Residual Hybrid
- Depthwise separable convolutions
- Residual connections every 1–2 blocks  
- Tests whether residual learning improves optimization in lightweight CNNs  
- Combines MobileNet efficiency with ResNet stability

---

### H4. Fire–Residual Hybrid
- SqueezeNet Fire modules:
  - squeeze (1×1 conv)
  - expand (1×1 and 3×3 conv)
- Add residual connections between Fire modules  
- Tests whether compression bottlenecks benefit from skip connections  
- Combines SqueezeNet compression with ResNet-style training stability

---

### H5. Factorized AlexNet (Kernel-Centric Model)
- Replace k×k convolutions with:
  - 1×k followed by k×1 factorized convolutions
- Optional 1×1 bottleneck layers
- Optional residual connections  
- Tests pure impact of kernel factorization on accuracy and efficiency  
- Based on AlexNet and Inception-style factorization ideas

---

### H6. Large-Kernel vs Small-Kernel Hybrid
- Parallel convolution paths:
  - stacked 3×3 convolutions (small kernel path)
  - 7×7 or 9×9 depthwise convolutions (large kernel path)
- Fuse outputs via sum or concatenation  
- Tests trade-off between receptive field size and efficiency  
- Inspired by VGG and modern large-kernel CNNs (e.g., RepLKNet)

---

### H7. Shuffle–Residual Hybrid
- Grouped convolutions for efficiency
- Channel shuffle for cross-group information mixing
- Residual connections between blocks  
- Tests whether channel mixing or kernel size is the main bottleneck  
- Based on ShuffleNet and ResNet principles

---

### H8. ConvNeXt-Lite Hybrid
- Depthwise 7×7 convolutions
- Inverted bottleneck structure
- Layer normalization
- Residual connections  
- Tests modern CNN design under strict efficiency constraints  
- Inspired by ConvNeXt and MobileNetV2

---

### H9. Kernel Constraint Network (Core Research Model)
- Strict convolution constraints:
  - only 1×1 and 3×3 convolutions allowed
- Compensatory mechanisms:
  - deeper stacking
  - residual connections
  - multi-branch modules (Inception-lite)
  - bottlenecks  
- Tests whether restricted kernels can match modern CNN performance

---

### H10. Attention-Augmented CNN
- Base CNN (AlexNet/VGG-style or kernel-constrained variant)
- Add lightweight attention modules:
  - Squeeze-and-Excitation (SE)
  - optional spatial attention  
- Tests whether channel/spatial attention compensates for kernel limitations  
- Focus on efficiency vs accuracy trade-off