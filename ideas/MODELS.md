# Model Catalog

This document provides a comprehensive overview of all models in the alexnet_rafael project, organized by research phase. Each model includes its architecture, inspiration, and rationale for inclusion.

---

## Phase 1: Reference Architectures (Baselines)

These classical and modern architectures establish performance baselines for the kernel restriction study.

### AlexNetTV

**File:** `models/baselines.py`

**Architecture:**
- 5 convolutional stages with large kernels: 11×11 (stride=4), 5×5, 3×3, 3×3, 3×3
- Max pooling between stages
- 3 fully connected layers (4096, 4096, 200)
- Input: 224×224 → rescaled to 64×64 for Tiny ImageNet
- Pretrained on ImageNet, fine-tuned for 200 classes

**Inspiration:**
- Original AlexNet architecture from Krizhevsky et al. (2012)
- Uses torchvision's pretrained IMAGENET1K_V1 weights

**Why Chosen:**
- Classical deep learning benchmark with proven generalization
- Large kernel sizes (11×7×5) establish the "efficiency ceiling" for Winograd accelerators
- Pretrained weights provide strong feature initialization
- Serves as the theoretical maximum accuracy in Phase 1

**Key Paper:**
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NIPS* [[PDF](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)]

**Expected Performance:** 24–30% top-1 (pretrained)  
**Model Size:** 220 MB FP32 / 55 MB INT8

---

### VGGStyleCNN

**File:** `models/baselines.py`

**Architecture:**
- 5 stages of stacked 3×3 convolutions (2 convs per stage)
- Max pooling between stages (stride=2)
- Channel progression: 32→32→64→64→128→128→256→256→256→256
- Global average pooling followed by single linear layer
- No pretrained weights

**Inspiration:**
- VGG network design (Simonyan & Zisserman, 2014) — demonstrates that depth via stacking small kernels is effective
- Simplified version optimized for 64×64 images

**Why Chosen:**
- Pure 3×3 baseline with no residuals, demonstrating depth-only compensation
- Small parameter count (2.41M) while maintaining depth
- Establishes VGG-style design as a feasible small-kernel alternative
- Fully QAT-compatible with Conv-BN-ReLU fusion throughout

**Key Paper:**
- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR* [[PDF](https://arxiv.org/abs/1409.1556)]

**Expected Performance:** 35–45% top-1 (trained from scratch)  
**Model Size:** 5 MB FP32 / 1.3 MB INT8

---

### ResNet18TV

**File:** `models/baselines.py`

**Architecture:**
- 4 residual layer groups with 2 BasicBlocks each (9 layers total)
- 7×7 stem with stride=2, followed by max pooling (reduces 64×64 → 8×8 early)
- Skip connections with identity or 1×1 projection
- Global average pooling → linear classifier
- Pretrained on ImageNet, fine-tuned for 200 classes

**Inspiration:**
- ResNet (He et al., 2015) — residual learning enables training of very deep networks
- Modern architecture with skip connections and normalized initialization
- Uses torchvision's IMAGENET1K_V1 weights

**Why Chosen:**
- Lightweight modern baseline (11.28M params)
- Residual connections enable efficient optimization
- Demonstrates the impact of skip connections on small-kernel networks
- Pretrained weights provide competitive accuracy reference

**Key Papers:**
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Deep Residual Learning for Image Recognition." *CVPR* [[PDF](https://arxiv.org/abs/1512.03385)]

**Expected Performance:** 46–65% top-1 (pretrained)  
**Model Size:** 44 MB FP32 / 11 MB INT8

---

### MobileNetV2TV

**File:** `models/baselines.py`

**Architecture:**
- Inverted residual blocks (expand→depthwise sep→project pattern)
- Channel progression from 32 to 1280, then projection to 200 classes
- Linear bottlenecks: ReLU6(x+3)/6 approximation at output layers
- Depthwise separable convolutions (3×3 depthwise + 1×1 pointwise)
- Pretrained on ImageNet (IMAGENET1K_V2), fine-tuned for 200 classes

**Inspiration:**
- MobileNetV2 (Sandler et al., 2018) — inverted residuals and linear bottlenecks for efficient mobile inference
- Gold standard for efficiency-aware mobile CNN design
- Uses torchvision's pretrained IMAGENET1K_V2 weights

**Why Chosen:**
- Smallest Phase 1 baseline (2.48M params, 3.5 MB INT8)
- Depthwise separable convolutions reduce FLOPs 8–9× vs standard conv
- Demonstrates modern efficiency techniques (linear bottlenecks, inverted residuals)
- Establishes the efficiency ceiling for Phase 2 comparisons
- Represents state-of-the-art mobile architecture

**Key Paper:**
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR* [[PDF](https://arxiv.org/abs/1801.04381)]

**Expected Performance:** 45–65% top-1 (pretrained)  
**Model Size:** 14 MB FP32 / 3.5 MB INT8

---

## Phase 2: Kernel Restriction Study

These models isolate kernel size as the experimental variable, all based on AlexNet architecture.

### AlexNet3x3

**File:** `models/alexnet_variants.py`

**Architecture:**
- 5 convolutional stages, all 3×3 kernels
- Identical channel widths to AlexNetTV: 64→192→384→256→256
- No BatchNorm (matches AlexNetTV for clean kernel-size comparison)
- First conv: stride=2, padding=1
- Subsequent convs: stride=1, padding=1
- AdaptiveAvgPool(6×6) + 3-layer FC (4096→4096→200)

**Inspiration:**
- AlexNet architecture with kernel sizes uniformly reduced to 3×3
- Controlled experiment isolating kernel size as single variable
- Enables direct comparison with AlexNetTV

**Why Chosen:**
- Pure kernel-size ablation: everything identical to AlexNetTV except kernel sizes
- Tests whether classical AlexNet can work with only 3×3 kernels
- Establishes the lower bound of receptive field while maintaining other design choices
- Winograd-compatible kernel size (3×3 is a standard Winograd size)

**Expected Performance:** 8–12% top-1 (from scratch)  
**Model Size:** 220 MB FP32 / 55 MB INT8

---

### AlexNet2x2

**File:** `models/alexnet_variants.py`

**Architecture:**
- 5 convolutional stages, all 2×2 kernels with stride=1
- No padding → spatial reduction of 1 per layer
- Spatial progression: 64 → 32 → 16 → 15 → 7 → 6 → 5 → 4
- AdaptiveAvgPool(1×1) absorbs non-standard spatial sizes
- Single linear classifier (256→200)
- Channel widths: 64→192→384→256→256

**Inspiration:**
- Tests the extreme lower bound of even-kernel convolutions
- 2×2 kernels: smallest even kernel, no center pixel, incompatible with Winograd
- Explores ultra-compact receptive fields

**Why Chosen:**
- Tests whether even smaller kernels (2×2 vs 3×3) reduce parameter efficiency
- Smallest receptive field: two stacked 2×2 = effective RF ~3×3
- Establishes practical lower limit for Winograd restrictions
- GAP head reduces overfitting on 200-class task despite small RF

**Expected Performance:** 5–9% top-1 (from scratch)  
**Model Size:** 3 MB FP32 / 1 MB INT8

---

### AlexNetStacked

**File:** `models/alexnet_variants.py`

**Architecture:**
- 5 stages with 2 stacked 3×3 convolutions per stage (10 conv total)
- BatchNorm after each conv, ReLU(inplace=False)
- Same channel widths as AlexNet3x3
- Two stacked 3×3 convs: combined RF ≈ 5×5 (matches original AlexNet 5×5)
- 3-layer FC head (4096→4096→200)
- AdaptiveAvgPool(6×6) feature pooling

**Inspiration:**
- VGG design principle: depth via stacking compensates for smaller kernels
- Tests whether doubling convolutions recovers receptive field lost by smaller kernels
- Approximates AlexNetTV's mixed-kernel design using only 3×3

**Why Chosen:**
- Direct test of "depth vs width" trade-off for kernel restriction
- Two stacked 3×3s achieve same effective RF as one 5×5
- Isolates whether receptive field or kernel diversity matters more
- BatchNorm addition tests its impact on small-kernel networks

**Expected Performance:** 12–18% top-1 (from scratch)  
**Model Size:** 220 MB FP32 / 55 MB INT8

---

### AlexNetMixed

**File:** `models/alexnet_variants.py`

**Architecture:**
- 5 stages alternating 3×3 and 2×2 kernels: 3×3 → 2×2 → 3×3 → 2×2 → 3×3
- 2×2 convs use no padding, 3×3 uses padding=1
- Spatial dims: 64 → 32 → 16 → 15 → 7 → 7 → 6 → 6 → AdaptiveAvgPool(1)
- GAP + single linear classifier (256→200)
- Channel widths: 64→192→384→256→256

**Inspiration:**
- Tests heterogeneous kernel strategies within a single model
- Combines receptive field diversity: large jumps from 3×3, compression from 2×2
- Practical design for systems that support variable kernel sizes

**Why Chosen:**
- Evaluates whether kernel mixing outperforms uniform restriction
- Bridges AlexNet3x3 and AlexNet2x2 design spaces
- Tests if alternating kernel costs differ from pure 2×2 or pure 3×3
- Lightweight head (GAP) avoids FC bottleneck

**Expected Performance:** 7–12% top-1 (from scratch)  
**Model Size:** 3 MB FP32 / 1 MB INT8

---

### AlexNetSmallKernel

**File:** `models/alexnet_variants.py`

**Architecture:**
- 5 convolutional stages, all 3×3 kernels
- **Narrow channels:** 64→128→256→256→256 (vs 64→192→384→256→256 in AlexNet3x3)
- No BatchNorm
- AdaptiveAvgPool(1×1) + single linear classifier
- ~36× fewer parameters than AlexNet3x3

**Inspiration:**
- Combines kernel restriction (3×3) with channel reduction for extreme efficiency
- Minimal design: removes parameters from both kernel size *and* channel width
- Demonstrates parameter efficiency from GAP + narrow channels

**Why Chosen:**
- The most parameter-efficient Phase 2 model (1.6M params)
- Tests whether narrow channels compensate for small kernels
- Best candidate for Winograd acceleration: small kernels + small footprint
- Shows the limits of ultra-compact CNN design on 200-class task

**Expected Performance:** 8–10% top-1 (from scratch)  
**Model Size:** 6 MB FP32 / 1.6 MB INT8

---

## Phase 3a: Compensation Mechanisms

These models add exactly one architectural mechanism to AlexNet3x3, isolating each technique's impact.

### AlexNetBottleneck

**File:** `models/compensation.py`

**Architecture:**
- 5 stages of bottleneck blocks: 1×1 (squeeze) → 3×3 → 1×1 (expand)
- Squeeze ratio: 1/4 (mid_ch = out_ch // 4)
- BatchNorm and ReLU(inplace=False) after each conv
- GAP(1×1) + single linear classifier
- Channel widths: 3→64→192→384→256→256

**Inspiration:**
- Inception/ResNet bottleneck design (Szegedy et al., 2015)
- Compress spatial information before 3×3, then expand channel-wise
- Parameter efficiency via squeeze-expand pattern

**Why Chosen:**
- Tests whether bottleneck parameterization compensates for small kernels
- Reduction=4 significantly cuts parameters per stage while keeping 3×3 core
- Isolates the bottleneck mechanism as a standalone compensation technique
- Lightweight head reduces overfitting on 200 classes

**Key Paper:**
- Szegedy, C., Liu, W., Jia, Y., et al. (2015). "Going Deeper with Convolutions." *CVPR* [[PDF](https://arxiv.org/abs/1409.4842)]

**Expected Performance:** 12–18% top-1 (from scratch)  
**Model Size:** 50 MB FP32 / 13 MB INT8

---

### AlexNetFactorized

**File:** `models/compensation.py`

**Architecture:**
- 5 stages replacing each 3×3 conv with 1×3 + 3×1 factorized pairs
- Same channel progression as AlexNet3x3 (64→192→384→256→256)
- BatchNorm and ReLU after each factorized conv
- AdaptiveAvgPool(6×6) + 3-layer FC (4096→4096→200)

**Inspiration:**
- Inception module spatial factorization (Szegedy et al., 2014)
- Asymmetric convolutions reduce FLOPs while maintaining spatial coverage
- ~33% parameter reduction compared to square 3×3 kernels

**Why Chosen:**
- Tests spatial factorization as a parameter-efficiency technique
- 1×3 + 3×1 has same effective RF as 3×3 with fewer parameters
- Explores asymmetric kernel patterns compatible with Winograd (each factorized piece)
- Maintains depth while reducing computational cost

**Key Paper:**
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlyan, J., & Wojna, Z. (2014). "Rethinking the Inception Architecture for Computer Vision." *CVPR* [[PDF](https://arxiv.org/abs/1512.00567)]

**Expected Performance:** 10–15% top-1 (from scratch)  
**Model Size:** 200 MB FP32 / 50 MB INT8

---

### AlexNetGroupConv

**File:** `models/compensation.py`

**Architecture:**
- 5 stages with grouped 3×3 convolutions (groups=4 for all but stage 1)
- Stage 1 uses groups=1 (3 input channels not divisible by 4)
- Channel widths 64, 192, 384, 256 (all divisible by 4)
- BatchNorm and ReLU after each conv
- AdaptiveAvgPool(6×6) + 3-layer FC (4096→4096→200)

**Inspiration:**
- AlexNet original design used grouped convolutions (Krizhevsky et al., 2012)
- Groups reduce parameter count and cross-channel computation by ~4×
- Modern grouped convolutions are fully supported in quantization

**Why Chosen:**
- Tests whether group-based channel separation compensates for small kernels
- Groups=4 splits channels into independent streams, reducing mixing
- Direct hardware efficiency gain: modern accelerators optimize grouped operations
- Isolates group convolution as a standalone compensation mechanism

**Expected Performance:** 10–18% top-1 (from scratch)  
**Model Size:** 60 MB FP32 / 15 MB INT8

---

### AlexNetDepthwiseSep

**File:** `models/compensation.py`

**Architecture:**
- 5 stages of depthwise separable convolutions (3×3 depthwise + 1×1 pointwise)
- Depthwise: 3×3 with groups=input_channels
- Pointwise: 1×1 with groups=1 for channel mixing
- BatchNorm and ReLU after each operation
- AdaptiveAvgPool(1×1) + single linear classifier
- Channel widths: 64→192→384→256→256

**Inspiration:**
- MobileNet design principle (Howard et al., 2017)
- Separates spatial filtering (depthwise) from channel mixing (pointwise)
- ~8–9× FLOPs reduction compared to standard 3×3 conv

**Why Chosen:**
- Most efficient Phase 3a model (8 MB FP32 / 2 MB INT8)
- Demonstrates MobileNet-style factorization on restricted kernels
- Depthwise 3×3 is Winograd-compatible and highly optimized
- Tests efficiency via operational decomposition rather than depth/width trade-offs

**Key Paper:**
- Howard, A. G., Zhu, M., Chen, B., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." *arXiv* [[PDF](https://arxiv.org/abs/1704.04861)]

**Expected Performance:** 10–15% top-1 (from scratch)  
**Model Size:** 8 MB FP32 / 2 MB INT8

---

### AlexNetResidual

**File:** `models/compensation.py`

**Architecture:**
- 5 residual blocks, each with 2 consecutive 3×3 convs + BN + ReLU + skip add
- FloatFunctional() for quantization-friendly skip addition
- Optional SE blocks (Squeeze-Excitation) after each residual block (disabled for QAT due to Sigmoid)
- Channel widths: 64→192→384→256→256
- AdaptiveAvgPool(6×6) + 3-layer FC (4096→4096→200)

**Inspiration:**
- ResNet residual learning (He et al., 2015)
- Skip connections enable stable training of deeper networks with small kernels
- Empirically the single most impactful compensation mechanism for small CNNs

**Why Chosen:**
- **Most research-relevant Phase 3a model** — residuals are the single strongest small-kernel compensator
- Skip connections directly address the gradient flow problems of small-kernel networks
- Expected 20–30% accuracy improvement vs plain AlexNet3x3
- FloatFunctional skip-add is fully QAT-compatible (INT8 residuals work correctly)
- Optional SE blocks provide channel attention for further global context

**Key Paper:**
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Deep Residual Learning for Image Recognition." *CVPR* [[PDF](https://arxiv.org/abs/1512.03385)]

**Expected Performance:** 20–30% top-1 (from scratch)  
**Model Size:** 220 MB FP32 / 55 MB INT8

---

### AlexNetFire

**File:** `models/compensation.py`

**Architecture:**
- 5 Fire modules (SqueezeNet design), each: 1×1 squeeze → parallel [1×1 expand, 3×3 expand] → concatenate
- Squeeze: compress to 1/4 of output channels
- Expand: split into pure channel mixing (1×1) and spatial mixing (3×3)
- BatchNorm and ReLU after each operation
- AdaptiveAvgPool(1×1) + single linear classifier
- Channel widths: 3→64→192→384→256→256

**Inspiration:**
- SqueezeNet Fire module (Iandola et al., 2016)
- Aggressive channel compression + multi-branch expand
- Combines bottleneck squeezing with multi-scale receptive fields

**Why Chosen:**
- Tests SqueezeNet-style aggressive compression + multi-path design
- Fire modules reduce parameters while adding expressivity via dual expand paths
- One of the smallest Phase 3a models (5 MB FP32)
- Parallel 1×1 and 3×3 expands provide multi-scale feature mixing
- Fully QAT-compatible (torch.cat is quantization-safe)

**Key Paper:**
- Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). "SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and <0.5MB Model Size." *arXiv* [[PDF](https://arxiv.org/abs/1602.07360)]

**Expected Performance:** 10–16% top-1 (from scratch)  
**Model Size:** 5 MB FP32 / 1.5 MB INT8

---

### AlexNetGAP

**File:** `models/compensation.py`

**Architecture:**
- Identical backbone to AlexNet3x3 (5× 3×3 conv, no BN, same channels)
- AdaptiveAvgPool(1×1) replacing the 6×6 pooling
- Single linear classifier (256→200) replacing 3-layer FC (4096→4096→200)
- No other changes from AlexNet3x3

**Inspiration:**
- Global Average Pooling principle (Lin et al., 2013)
- GAP enforces spatial invariance and eliminates unnecessary FC parameters
- Network-in-Network design philosophy

**Why Chosen:**
- **Isolates head architecture as a single variable** (holds backbone identical to AlexNet3x3)
- Tests whether GAP's regularization effect (spatial averaging, no learnable params) compensates for small kernels
- Extreme parameter reduction: 3 MB FP32 vs 220 MB for AlexNet3x3 FC head
- Distinguishes backbone capacity from classification head design

**Key Paper:**
- Lin, M., Chen, Q., & Yan, S. (2013). "Network in Network." *ICLR* [[PDF](https://arxiv.org/abs/1312.4400)]

**Expected Performance:** 8–12% top-1 (from scratch)  
**Model Size:** 3 MB FP32 / 1 MB INT8

---

### AlexNetSE

**File:** `models/compensation.py`

**Architecture:**
- Identical backbone to AlexNet3x3 (5× 3×3 conv, no BN, channels: 64→192→384→256→256)
- Squeeze-Excitation blocks after each conv stage
- SE block: Global avg pool → FC squeeze (1/16) → ReLU → FC expand → Sigmoid scale
- AdaptiveAvgPool(6×6) + 3-layer FC (4096→4096→200)

**Inspiration:**
- Squeeze-and-Excitation Networks (Hu et al., 2018)
- Channel attention mechanism: recalibrate channel responses based on global context
- Compensates for local-only 3×3 receptive field with global channel weighting

**Why Chosen:**
- Tests global channel attention as a compensation for small kernels
- SE blocks add global context to local 3×3 receptive fields
- Modest parameter overhead (~2–5%) with measurable accuracy gains
- Isolates channel attention as a standalone mechanism
- **QAT limitation:** Sigmoid is not fbgemm-compatible; use_se=False by default for INT8

**Key Paper:**
- Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." *CVPR* [[PDF](https://arxiv.org/abs/1709.01507)]

**Expected Performance:** 15–22% top-1 (from scratch)  
**Model Size:** 220 MB FP32 / INT8 not supported (Sigmoid limitation)

---

## Phase 3b: Efficient Hybrids

These models combine multiple compensation mechanisms and modern design principles for practical efficiency.

### TinyHybridNet

**File:** `models/tinyhybridnet.py`

**Architecture:**
- Stem: 3×3 conv (3→32) + BN + ReLU
- 6 FireMobileResidual blocks:
  - Block structure: 1×1 squeeze → 3×3 depthwise + BN + ReLU → 1×1 expand + BN
  - Residual skip add (FloatFunctional) with stride projection
  - Channels: 32→64→64→128→128→256→256
  - Strides: [1, 1, 2, 1, 2, 1]
- AdaptiveAvgPool(1×1) + single linear classifier
- All ReLU(inplace=False), BN throughout

**Inspiration:**
- Fire module squeeze-expand from SqueezeNet
- Residual learning from ResNet
- Depthwise separable convolutions from MobileNet
- Hybrid approach combining three modern efficiency techniques

**Why Chosen:**
- Combines Fire squeeze-expand + depthwise separable + residuals in one model
- Practical design balancing efficiency (small kernels, separable) and accuracy (residuals)
- All 3×3 kernels are Winograd-compatible
- Compact model (reasonable parameters) with strong accuracy potential
- Demonstrates integration of Phase 3a techniques into cohesive architecture

**Expected Performance:** 35–45% top-1 (from scratch)  
**Model Size:** ~6 MB FP32 / ~1.5 MB INT8

---

### InvertedResidual

**File:** `models/tinyhybridnet.py`

**Architecture:**
- Linear bottleneck structure: expand (optional) → 3×3 depthwise → project (1×1)
- Expand ratio: 1–6 (configurable, default 6)
- Residual skip when stride=1 and in_ch=out_ch
- FloatFunctional skip add for QAT compatibility
- All operations: Conv2d + BN + ReLU(inplace=False)

**Inspiration:**
- MobileNetV2 inverted residual block (Sandler et al., 2018)
- Inverted bottleneck: expand middle channel width before depthwise to increase receptive field
- Linear projection output (no activation) prevents information loss

**Why Chosen:**
- Sub-component of TinyMobileNetV2 (see below)
- Represents modern mobile CNN building block
- Inverted design (expand → depthwise → project) is proven efficient
- FloatFunctional skip-add ensures full QAT compatibility

**Key Paper:**
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR* [[PDF](https://arxiv.org/abs/1801.04381)]

---

### TinyMobileNetV2

**File:** `models/tinyhybridnet.py`

**Architecture:**
- Stem: 3×3 conv (3→32) + BN + ReLU
- 6 InvertedResidual blocks:
  - Block 1: expand_ratio=1 (linear, no expand)
  - Blocks 2–6: expand_ratio=6
  - Strides: [1, 1, 2, 1, 2, 1]
  - Channels: 32→64→64→128→128→256→256
- AdaptiveAvgPool(1×1) + single linear classifier
- All operations: Conv2d + BN + ReLU(inplace=False), FloatFunctional skip-add

**Inspiration:**
- MobileNetV2 inverted residual architecture for mobile/efficient inference
- Adapts MobileNetV2 to 64×64 images and 200-class classification
- Modern state-of-the-art for mobile CNNs

**Why Chosen:**
- Directly comparable to MobileNetV2TV pretrained baseline
- Tests whether trainable inverted residuals match pretrained performance
- Lightweight design: all kernels 3×3 or 1×1, depthwise-separable core
- **Most practical Phase 3b model** for production Winograd deployment
- Inverted residuals provide strong optimization stability for small image size

**Key Paper:**
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR* [[PDF](https://arxiv.org/abs/1801.04381)]

**Expected Performance:** 35–50% top-1 (from scratch)  
**Model Size:** ~6 MB FP32 / ~1.5 MB INT8

---

### FireMobileResidual

**File:** `models/tinyhybridnet.py`

**Architecture:**
- Fire squeeze: 1×1 conv → BN → ReLU (reduces to ~25% of output channels)
- Depthwise mobile: 3×3 depthwise → BN → ReLU (spatial filtering, no channel mixing)
- Expand: 1×1 conv → BN (projects to target channels)
- Residual skip: identity or 1×1 conv projection (stride, channel mismatch handling)
- FloatFunctional skip add with optional stride support

**Inspiration:**
- Fire module squeeze-expand from SqueezeNet
- Depthwise separable convolutions from MobileNet
- Residual learning from ResNet
- Hybrid design combining three complementary techniques

**Why Chosen:**
- Sub-component of TinyHybridNet (see above)
- Isolates Fire-Mobile-Residual fusion pattern for study
- Combines aggressive squeeze (1/4), efficient spatial filtering (depthwise), and residual stability
- QAT-fully compatible: no unsupported operations

---

## Model Selection Summary

| Phase | Model | Params | FP32 Size | Key Property |
|-------|-------|--------|-----------|--------------|
| **1** | AlexNetTV | 57.82M | 220 MB | Pretrained baseline, large kernels |
| **1** | VGGStyleCNN | 2.41M | 5 MB | Pure 3×3, stacked depth |
| **1** | ResNet18TV | 11.28M | 44 MB | Residual learning, pretrained |
| **1** | MobileNetV2TV | 2.48M | 14 MB | Depthwise separable, pretrained |
| **2** | AlexNet3x3 | 57.61M | 220 MB | Uniform 3×3 kernel restriction |
| **2** | AlexNet2x2 | 1.05M | 12 MB | Ultra-small 2×2 kernels |
| **2** | AlexNetStacked | 60.48M | 230 MB | Double 3×3 stacking |
| **2** | AlexNetMixed | 1.75M | 20 MB | Alternating 3×3 and 2×2 |
| **2** | AlexNetSmallKernel | 1.60M | 18 MB | Narrow channels, 3×3, GAP |
| **3a** | AlexNetBottleneck | ↓ | 50 MB | Bottleneck parameterization |
| **3a** | AlexNetFactorized | ↓ | 200 MB | 1×3 + 3×1 asymmetric kernels |
| **3a** | AlexNetGroupConv | ↓ | 60 MB | Grouped 3×3 convolutions |
| **3a** | AlexNetDepthwiseSep | ↓ | 8 MB | Depthwise separable convolutions |
| **3a** | AlexNetResidual | ↓ | 220 MB | **Residual skip connections** |
| **3a** | AlexNetFire | ↓ | 5 MB | Fire squeeze-expand modules |
| **3a** | AlexNetGAP | ↓ | 3 MB | Global average pooling head |
| **3a** | AlexNetSE | ↓ | 220 MB | Squeeze-excitation channel attention |
| **3b** | TinyHybridNet | ↓ | 6 MB | Fire + depthwise + residual hybrid |
| **3b** | TinyMobileNetV2 | ↓ | 6 MB | Inverted residuals (MobileNetV2-style) |

---

## Key References

### Foundational Deep Learning
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NIPS* [[PDF](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)]
- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR* [[PDF](https://arxiv.org/abs/1409.1556)]
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Deep Residual Learning for Image Recognition." *CVPR* [[PDF](https://arxiv.org/abs/1512.03385)]

### Mobile & Efficient CNNs
- Howard, A. G., Zhu, M., Chen, B., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." *arXiv* [[PDF](https://arxiv.org/abs/1704.04861)]
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR* [[PDF](https://arxiv.org/abs/1801.04381)]
- Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). "SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and <0.5MB Model Size." *arXiv* [[PDF](https://arxiv.org/abs/1602.07360)]

### Modern Attention & Architectural Components
- Szegedy, C., Liu, W., Jia, Y., et al. (2015). "Going Deeper with Convolutions." *CVPR* [[PDF](https://arxiv.org/abs/1409.4842)]
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlyan, J., & Wojna, Z. (2014). "Rethinking the Inception Architecture for Computer Vision." *CVPR* [[PDF](https://arxiv.org/abs/1512.00567)]
- Lin, M., Chen, Q., & Yan, S. (2013). "Network in Network." *ICLR* [[PDF](https://arxiv.org/abs/1312.4400)]
- Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." *CVPR* [[PDF](https://arxiv.org/abs/1709.01507)]

---

**Last Updated:** June 2026  
**Project:** alexnet_rafael (Winograd-accelerated CNN research)
